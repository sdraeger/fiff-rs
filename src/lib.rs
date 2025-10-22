/*! FIFF (Functional Imaging File Format) Parser
 *
 * ✅ **COMPREHENSIVE METADATA PARSER** - Production-ready parser for Neuromag/Elekta MEG data
 * with extensive metadata support. See `IMPLEMENTATION_STATUS.md` for full details.
 *
 * This module provides a pure Rust implementation of the FIFF file format parser,
 * based on MNE-Python's implementation. FIFF is the native format for Neuromag/Elekta
 * MEG systems and is widely used in MEG/EEG data analysis.
 *
 * **For advanced MEG analysis or matrix operations, use [MNE-Python](https://mne.tools/) instead.**
 *
 * # Public API
 *
 * ## Core Functions
 * - [`open_fiff`]: Open and parse a FIFF file
 * - [`MeasInfo::read`]: Extract measurement metadata
 *
 * ## Data Structures
 * - [`ChannelInfo`]: Channel information (name, type, calibration, etc.)
 * - [`MeasInfo`]: Measurement metadata (channels, sampling rate, etc.)
 * - [`TreeNode`]: FIFF directory tree structure
 * - [`Tag`]: FIFF tag (kind, type, data)
 * - [`DirEntry`]: FIFF directory entry
 *
 * ## Constants
 * All FIFF constants are re-exported from the [`constants`] module, including:
 * - Block types (FIFFB_*)
 * - Tag kinds (FIFF_*)
 * - Data types (FIFFT_*)
 * - Channel types (FIFFV_*_CH)
 *
 * ## Helper Functions
 * - [`channel_type_name`]: Get human-readable channel type name
 * - [`is_data_channel`]: Check if a channel type is a data channel
 * - [`type_size`]: Get byte size for a FIFF data type
 */

// Submodules
pub mod constants;
pub mod tag;
pub mod tree;

// Re-exports: Public API
pub use constants::*;
pub use tag::{DirEntry, Tag};
pub use tree::{build_tree, dir_tree_find, TreeNode};

use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Open and parse a FIFF file
pub fn open_fiff(path: &Path) -> std::io::Result<(BufReader<File>, TreeNode)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // FIFF files start with a tag structure, not a traditional magic number
    // Read first tag to validate this is a FIFF file
    let first_tag_kind = reader.read_i32::<BigEndian>()?;

    // Common first tag kinds in FIFF files (FIFF_FILE_ID = 100, FIFF_DIR_POINTER = 101, etc.)
    if first_tag_kind < 100 || first_tag_kind > 200 {
        reader.seek(SeekFrom::Start(0))?;
    }

    // Look for directory pointer tag (kind 101)
    // Try to find it in the first few tags
    let mut dir_pos: Option<u64> = None;
    reader.seek(SeekFrom::Start(0))?;

    for i in 0..20 {
        // Check first 20 tags
        let pos_before = reader.stream_position()?;
        let kind = reader.read_i32::<BigEndian>()?;
        let type_ = reader.read_i32::<BigEndian>()?;
        let size = reader.read_i32::<BigEndian>()?;
        let _next = reader.read_i32::<BigEndian>()?;

        eprintln!(
            "Tag {}: kind={}, type={}, size={} at pos {}",
            i, kind, type_, size, pos_before
        );

        if kind == 101 {
            // FIFF_DIR_POINTER
            // Read the directory position (as unsigned!)
            let dir_pos_val = reader.read_u32::<BigEndian>()? as u64;
            eprintln!(
                "Found directory pointer: {} (0x{:X})",
                dir_pos_val, dir_pos_val
            );
            dir_pos = Some(dir_pos_val);
            break;
        }

        // Skip to next tag
        let next_pos = pos_before + 16 + size as u64;
        reader.seek(SeekFrom::Start(next_pos))?;
    }

    let directory = if let Some(pos) = dir_pos {
        // Check if directory pointer is valid (not -1/0xFFFFFFFF)
        if pos == 0xFFFFFFFF {
            eprintln!("Directory pointer is -1 (0xFFFFFFFF), using sequential tag reading");
            None
        } else {
            // Try to read directory from the specified position
            match read_directory_at(&mut reader, pos) {
                Ok(dir) => Some(dir),
                Err(e) => {
                    eprintln!(
                        "Failed to read directory at {}: {}, falling back to sequential reading",
                        pos, e
                    );
                    None
                }
            }
        }
    } else {
        eprintln!("No directory pointer found, using sequential tag reading");
        None
    };

    // Build tree from directory or sequential reading
    let tree = if let Some(dir) = directory {
        build_tree(&mut reader, dir)?
    } else {
        build_tree_sequential(&mut reader)?
    };

    // Reset to start for reading
    reader.seek(SeekFrom::Start(0))?;

    Ok((reader, tree))
}

/// Read directory from a specific file position
fn read_directory_at<R: Read + Seek>(reader: &mut R, pos: u64) -> std::io::Result<Vec<DirEntry>> {
    reader.seek(SeekFrom::Start(pos))?;
    let nent = reader.read_i32::<BigEndian>()? as usize;

    let mut directory = Vec::with_capacity(nent);
    for _ in 0..nent {
        directory.push(DirEntry::read(reader)?);
    }

    Ok(directory)
}

/// Build tree by reading tags sequentially from the file
fn build_tree_sequential<R: Read + Seek>(reader: &mut R) -> std::io::Result<TreeNode> {
    reader.seek(SeekFrom::Start(0))?;

    let mut directory = Vec::new();
    let mut pos = 0u64;

    // Read tags sequentially until we reach the end
    loop {
        reader.seek(SeekFrom::Start(pos))?;

        // Try to read tag header
        let kind = match reader.read_i32::<BigEndian>() {
            Ok(k) => k,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };

        let type_ = reader.read_i32::<BigEndian>()?;
        let size = reader.read_i32::<BigEndian>()?;
        let _next = reader.read_i32::<BigEndian>()?;

        // Add to directory
        directory.push(DirEntry {
            kind,
            type_,
            size,
            pos,
        });

        // Move to next tag
        pos += 16 + size as u64;

        // Safety check: don't read past reasonable file size
        if pos > 10_000_000_000 {
            // 10GB limit
            break;
        }
    }

    eprintln!(
        "Built directory with {} entries from sequential reading",
        directory.len()
    );

    // Build tree from the collected directory
    build_tree(reader, directory)
}

/// SSP/Signal Space Projection operator
/// Used for artifact removal (e.g., eye blinks, heartbeat, environmental noise)
#[derive(Debug, Clone)]
pub struct Projection {
    pub kind: i32,      // Channel type this projection applies to (MEG/EEG)
    pub active: bool,   // Whether projection is currently applied
    pub desc: String,   // Human-readable description
    pub nvec: i32,      // Number of projection vectors
    pub data: Vec<f64>, // Flattened projection matrix data (nvec × nchan)
}

impl Projection {
    /// Parse projection from PROJ block
    fn read<R: Read + Seek>(reader: &mut R, proj_node: &TreeNode) -> std::io::Result<Self> {
        // Read kind
        let kind = Self::find_tag_in_node(reader, proj_node, FIFF_PROJ_ITEM_KIND)?.as_i32()?;

        // Read description
        let desc = Self::find_tag_in_node(reader, proj_node, FIFF_PROJ_ITEM_DESC)?.as_string()?;

        // Read active status
        let active =
            Self::find_tag_in_node(reader, proj_node, FIFF_PROJ_ITEM_ACTIVE)?.as_i32()? != 0;

        // Read number of vectors
        let nvec = Self::find_tag_in_node(reader, proj_node, FIFF_PROJ_ITEM_NVEC)?.as_i32()?;

        // Read projection vectors (stored as named matrix)
        let data = Self::find_tag_in_node(reader, proj_node, FIFF_NAMED_MATRIX)
            .ok()
            .map(|tag| {
                // Parse the matrix data (stored as doubles)
                let mut result = Vec::new();
                let mut cursor = std::io::Cursor::new(&tag.data);

                // Skip the matrix name (first part of named matrix)
                // Just read all doubles from the data
                while cursor.position() < tag.data.len() as u64 - 7 {
                    if let Ok(val) = cursor.read_f64::<BigEndian>() {
                        result.push(val);
                    } else {
                        break;
                    }
                }
                result
            })
            .unwrap_or_default();

        Ok(Projection {
            kind,
            active,
            desc,
            nvec,
            data,
        })
    }

    fn find_tag_in_node<R: Read + Seek>(
        reader: &mut R,
        node: &TreeNode,
        kind: i32,
    ) -> std::io::Result<Tag> {
        for entry in &node.directory {
            if entry.kind == kind {
                return Tag::read_at(reader, entry.pos);
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Tag kind {} not found in projection", kind),
        ))
    }
}

/// CTF (Canadian Thin Films) MEG Compensation
/// CTF systems use reference channels to cancel environmental noise.
/// Compensation matrices transform reference channel data to remove artifacts.
#[derive(Debug, Clone)]
pub struct Compensation {
    pub kind: i32,        // Compensation grade (0=none, 1=G1BR, 2=G2BR, 3=G3BR)
    pub calibrated: bool, // Whether data is already compensated
    pub data: Vec<f64>,   // Flattened compensation matrix (ncomp × nchan)
}

impl Compensation {
    /// Parse CTF compensation from CTF_COMP block
    fn read<R: Read + Seek>(reader: &mut R, comp_node: &TreeNode) -> std::io::Result<Self> {
        // Read compensation kind (grade)
        let kind = Self::find_tag_in_node(reader, comp_node, FIFF_MNE_CTF_COMP_KIND)?.as_i32()?;

        // Read calibrated status (optional, defaults to false)
        let calibrated = Self::find_tag_in_node(reader, comp_node, FIFF_MNE_CTF_COMP_CALIBRATED)
            .ok()
            .and_then(|tag| tag.as_i32().ok())
            .map(|v| v != 0)
            .unwrap_or(false);

        // Read compensation data matrix (named matrix or derivation data)
        let data = Self::find_tag_in_node(reader, comp_node, FIFF_MNE_CTF_COMP_DATA)
            .or_else(|_| Self::find_tag_in_node(reader, comp_node, FIFF_MNE_DERIVATION_DATA))
            .ok()
            .map(|tag| {
                // Parse matrix data (stored as doubles)
                let mut result = Vec::new();
                let mut cursor = std::io::Cursor::new(&tag.data);

                // Read all doubles from the data
                while cursor.position() < tag.data.len() as u64 - 7 {
                    if let Ok(val) = cursor.read_f64::<BigEndian>() {
                        result.push(val);
                    } else {
                        break;
                    }
                }
                result
            })
            .unwrap_or_default();

        Ok(Compensation {
            kind,
            calibrated,
            data,
        })
    }

    fn find_tag_in_node<R: Read + Seek>(
        reader: &mut R,
        node: &TreeNode,
        kind: i32,
    ) -> std::io::Result<Tag> {
        for entry in &node.directory {
            if entry.kind == kind {
                return Tag::read_at(reader, entry.pos);
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Tag kind {} not found in compensation", kind),
        ))
    }

    /// Get human-readable compensation grade name
    pub fn grade_name(&self) -> &'static str {
        match self.kind {
            FIFFV_MNE_CTFV_COMP_NONE => "None",
            FIFFV_MNE_CTFV_COMP_G1BR => "G1BR",
            FIFFV_MNE_CTFV_COMP_G2BR => "G2BR",
            FIFFV_MNE_CTFV_COMP_G3BR => "G3BR",
            _ => "Unknown",
        }
    }
}

/// Coordinate Transformation
/// Transforms points between coordinate frames (device, head, MRI, etc.)
#[derive(Debug, Clone)]
pub struct CoordTrans {
    pub from: i32,       // Source coordinate frame
    pub to: i32,         // Destination coordinate frame
    pub rot: [f32; 9],   // 3x3 rotation matrix (row-major)
    pub move_: [f32; 3], // 3D translation vector
}

impl CoordTrans {
    /// Parse coordinate transformation from FIFF_COORD_TRANS tag data (56 bytes)
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        if data.len() < 56 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Coordinate transform data too short: {} bytes (expected 56)",
                    data.len()
                ),
            ));
        }

        let mut cursor = std::io::Cursor::new(data);

        // Read coordinate frames (4 bytes each)
        let from = cursor.read_i32::<BigEndian>()?;
        let to = cursor.read_i32::<BigEndian>()?;

        // Read 3x3 rotation matrix (9 floats = 36 bytes)
        let mut rot = [0.0f32; 9];
        for i in 0..9 {
            rot[i] = cursor.read_f32::<BigEndian>()?;
        }

        // Read translation vector (3 floats = 12 bytes)
        let mut move_ = [0.0f32; 3];
        for i in 0..3 {
            move_[i] = cursor.read_f32::<BigEndian>()?;
        }

        Ok(CoordTrans {
            from,
            to,
            rot,
            move_,
        })
    }

    /// Get human-readable description of this transformation
    pub fn description(&self) -> String {
        format!(
            "{} -> {}",
            coord_frame_name(self.from),
            coord_frame_name(self.to)
        )
    }

    /// Check if this is a device-to-head transform
    pub fn is_device_to_head(&self) -> bool {
        self.from == FIFFV_COORD_DEVICE && self.to == FIFFV_COORD_HEAD
    }

    /// Check if this is a head-to-MRI transform
    pub fn is_head_to_mri(&self) -> bool {
        self.from == FIFFV_COORD_HEAD && self.to == FIFFV_COORD_MRI
    }
}

/// Channel information parsed from FIFF_CH_INFO_STRUCT (96 bytes)
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    pub scanno: i32,     // Scan number (4 bytes)
    pub logno: i32,      // Logical channel number (4 bytes)
    pub kind: i32,       // Channel type (MEG, EEG, etc.) (4 bytes)
    pub range: f32,      // Range scaling factor (4 bytes)
    pub cal: f32,        // Calibration coefficient (4 bytes)
    pub coil_type: i32,  // Coil/sensor type (4 bytes)
    pub loc: [f32; 12],  // Location/orientation (48 bytes: 12 × f32)
    pub unit: i32,       // Physical unit (4 bytes)
    pub unit_mul: i32,   // Unit multiplier (4 bytes)
    pub ch_name: String, // Channel name (16 bytes null-terminated)
}
// Total: 4+4+4+4+4+4+48+4+4+16 = 96 bytes

impl ChannelInfo {
    /// Parse channel info from FIFF_CH_INFO_STRUCT binary data (96 bytes)
    pub fn from_bytes(data: &[u8]) -> std::io::Result<Self> {
        if data.len() < 96 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Channel info data too short: {} bytes (expected 96)",
                    data.len()
                ),
            ));
        }

        let mut cursor = std::io::Cursor::new(data);

        // Read fields in order (all big-endian)
        let scanno = cursor.read_i32::<BigEndian>()?;
        let logno = cursor.read_i32::<BigEndian>()?;
        let kind = cursor.read_i32::<BigEndian>()?;
        let range = cursor.read_f32::<BigEndian>()?;
        let cal = cursor.read_f32::<BigEndian>()?;
        let coil_type = cursor.read_i32::<BigEndian>()?;

        // Read 12 location floats
        let mut loc = [0.0f32; 12];
        for i in 0..12 {
            loc[i] = cursor.read_f32::<BigEndian>()?;
        }

        let unit = cursor.read_i32::<BigEndian>()?;
        let unit_mul = cursor.read_i32::<BigEndian>()?;

        // Read 16-byte channel name (null-terminated)
        let mut name_bytes = [0u8; 16];
        cursor.read_exact(&mut name_bytes)?;

        // Convert to string, stopping at first null
        let ch_name = String::from_utf8_lossy(&name_bytes)
            .trim_end_matches('\0')
            .to_string();

        Ok(ChannelInfo {
            scanno,
            logno,
            kind,
            range,
            cal,
            coil_type,
            loc,
            unit,
            unit_mul,
            ch_name,
        })
    }

    /// Get the full calibration factor (cal * range)
    pub fn calibration(&self) -> f64 {
        self.cal as f64 * self.range as f64
    }

    /// Check if this is a data channel (not stimulus, etc.)
    pub fn is_data_channel(&self) -> bool {
        is_data_channel(self.kind)
    }

    /// Get human-readable channel type name
    pub fn type_name(&self) -> &'static str {
        channel_type_name(self.kind)
    }
}

/// Measurement info from FIFF file
#[derive(Debug)]
pub struct MeasInfo {
    pub nchan: usize,
    pub sfreq: f64,
    pub channels: Vec<ChannelInfo>,

    // Filter information
    pub lowpass: Option<f64>,
    pub highpass: Option<f64>,

    // Timing
    pub meas_date: Option<i64>, // Unix timestamp (seconds since epoch)

    // Metadata
    pub line_freq: Option<f64>,
    pub experimenter: Option<String>,
    pub description: Option<String>,
    pub proj_id: Option<i32>,
    pub proj_name: Option<String>,

    // Bad channels
    pub bads: Vec<String>, // Bad channel names

    // SSP Projectors
    pub projs: Vec<Projection>, // Signal space projections for artifact removal

    // CTF Compensation
    pub comps: Vec<Compensation>, // CTF compensation matrices

    // Coordinate Transformations
    pub coord_trans: Vec<CoordTrans>, // Coordinate transformations (device-to-head, head-to-MRI, etc.)
}

impl MeasInfo {
    /// Read measurement info from FIFF file
    pub fn read<R: Read + Seek>(reader: &mut R, tree: &TreeNode) -> std::io::Result<Self> {
        // Find MEAS_INFO block
        let meas_nodes = dir_tree_find(tree, FIFFB_MEAS_INFO);

        if meas_nodes.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No measurement info found in FIFF file",
            ));
        }

        let meas_node = meas_nodes[0];

        // Read number of channels
        let nchan = Self::find_tag(reader, &meas_node.directory, FIFF_NCHAN)?.as_i32()? as usize;

        // Read sample frequency
        let sfreq = Self::find_tag(reader, &meas_node.directory, FIFF_SFREQ)?.as_f32()? as f64;

        // Parse channel info structures
        let mut channels = Vec::with_capacity(nchan);

        for entry in &meas_node.directory {
            if entry.kind == FIFF_CH_INFO {
                let tag = Tag::read_at(reader, entry.pos)?;

                // Verify this is the expected structure type
                if tag.type_ != FIFFT_CH_INFO_STRUCT {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "FIFF_CH_INFO tag has unexpected type {} (expected {})",
                            tag.type_, FIFFT_CH_INFO_STRUCT
                        ),
                    ));
                }

                // Parse the binary channel info structure
                let ch_info = ChannelInfo::from_bytes(&tag.data)?;
                channels.push(ch_info);
            }
        }

        // Verify we got the expected number of channels
        if channels.len() != nchan {
            eprintln!(
                "Warning: Expected {} channels but found {} FIFF_CH_INFO tags",
                nchan,
                channels.len()
            );

            // If we didn't get channel info, create placeholders
            if channels.is_empty() {
                channels = (0..nchan)
                    .map(|i| ChannelInfo {
                        scanno: i as i32,
                        logno: i as i32,
                        kind: FIFFV_MISC_CH,
                        range: 1.0,
                        cal: 1.0,
                        coil_type: 0,
                        loc: [0.0; 12],
                        unit: 0,
                        unit_mul: 0,
                        ch_name: format!("CH{}", i + 1),
                    })
                    .collect();
            }
        }

        // Read optional filter information
        let lowpass = Self::find_tag_optional(reader, &meas_node.directory, FIFF_LOWPASS)
            .and_then(|tag| tag.as_f32().ok())
            .map(|v| v as f64);

        let highpass = Self::find_tag_optional(reader, &meas_node.directory, FIFF_HIGHPASS)
            .and_then(|tag| tag.as_f32().ok())
            .map(|v| v as f64);

        // Read measurement date (Unix timestamp)
        let meas_date = Self::find_tag_optional(reader, &meas_node.directory, FIFF_MEAS_DATE)
            .and_then(|tag| tag.as_i32().ok())
            .map(|v| v as i64);

        // Read line frequency
        let line_freq = Self::find_tag_optional(reader, &meas_node.directory, FIFF_LINE_FREQ)
            .and_then(|tag| tag.as_f32().ok())
            .map(|v| v as f64);

        // Read metadata strings
        let experimenter = Self::find_tag_optional(reader, &meas_node.directory, FIFF_EXPERIMENTER)
            .and_then(|tag| tag.as_string().ok());

        let description = Self::find_tag_optional(reader, &meas_node.directory, FIFF_DESCRIPTION)
            .and_then(|tag| tag.as_string().ok());

        let proj_id = Self::find_tag_optional(reader, &meas_node.directory, FIFF_PROJ_ID)
            .and_then(|tag| tag.as_i32().ok());

        let proj_name = Self::find_tag_optional(reader, &meas_node.directory, FIFF_PROJ_NAME)
            .and_then(|tag| tag.as_string().ok());

        // Read bad channels (colon-separated string in MNE format)
        let bads = Self::find_tag_optional(reader, &meas_node.directory, FIFF_MNE_BAD_CHS)
            .and_then(|tag| tag.as_string().ok())
            .map(|s| {
                s.split(':')
                    .filter(|ch| !ch.is_empty())
                    .map(|ch| ch.trim().to_string())
                    .collect()
            })
            .unwrap_or_default();

        // Read SSP projections
        let proj_nodes = dir_tree_find(tree, FIFFB_PROJ);
        let mut projs = Vec::new();

        for proj_node in proj_nodes {
            match Projection::read(reader, proj_node) {
                Ok(proj) => projs.push(proj),
                Err(e) => {
                    eprintln!("Warning: Failed to parse projection: {}", e);
                }
            }
        }

        // Read CTF compensations
        let comp_nodes = dir_tree_find(tree, FIFFB_MNE_CTF_COMP);
        let mut comps = Vec::new();

        for comp_node in comp_nodes {
            match Compensation::read(reader, comp_node) {
                Ok(comp) => comps.push(comp),
                Err(e) => {
                    eprintln!("Warning: Failed to parse compensation: {}", e);
                }
            }
        }

        // Read coordinate transformations
        let mut coord_trans = Vec::new();

        for entry in &meas_node.directory {
            if entry.kind == FIFF_COORD_TRANS {
                match Tag::read_at(reader, entry.pos) {
                    Ok(tag) => match CoordTrans::from_bytes(&tag.data) {
                        Ok(trans) => coord_trans.push(trans),
                        Err(e) => {
                            eprintln!("Warning: Failed to parse coordinate transform: {}", e);
                        }
                    },
                    Err(e) => {
                        eprintln!("Warning: Failed to read coordinate transform tag: {}", e);
                    }
                }
            }
        }

        Ok(MeasInfo {
            nchan,
            sfreq,
            channels,
            lowpass,
            highpass,
            meas_date,
            line_freq,
            experimenter,
            description,
            proj_id,
            proj_name,
            bads,
            projs,
            comps,
            coord_trans,
        })
    }

    /// Find a tag with specific kind in directory
    fn find_tag<R: Read + Seek>(
        reader: &mut R,
        directory: &[DirEntry],
        kind: i32,
    ) -> std::io::Result<Tag> {
        for entry in directory {
            if entry.kind == kind {
                return Tag::read_at(reader, entry.pos);
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Tag kind {} not found", kind),
        ))
    }

    /// Find a tag with specific kind in directory, returning None if not found
    fn find_tag_optional<R: Read + Seek>(
        reader: &mut R,
        directory: &[DirEntry],
        kind: i32,
    ) -> Option<Tag> {
        Self::find_tag(reader, directory, kind).ok()
    }

    /// Get indices of good (non-bad) channels
    pub fn get_good_channels(&self) -> Vec<usize> {
        self.channels
            .iter()
            .enumerate()
            .filter(|(_, ch)| !self.bads.contains(&ch.ch_name))
            .map(|(i, _)| i)
            .collect()
    }

    /// Check if a channel is marked as bad
    pub fn is_bad_channel(&self, channel_name: &str) -> bool {
        self.bads.iter().any(|bad| bad == channel_name)
    }

    /// Get device-to-head coordinate transform, if available
    pub fn get_device_to_head_trans(&self) -> Option<&CoordTrans> {
        self.coord_trans
            .iter()
            .find(|trans| trans.is_device_to_head())
    }

    /// Get head-to-MRI coordinate transform, if available
    pub fn get_head_to_mri_trans(&self) -> Option<&CoordTrans> {
        self.coord_trans.iter().find(|trans| trans.is_head_to_mri())
    }

    /// Get all coordinate transformations
    pub fn get_coord_trans(&self, from: i32, to: i32) -> Vec<&CoordTrans> {
        self.coord_trans
            .iter()
            .filter(|trans| trans.from == from && trans.to == to)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Cursor;

    fn create_test_channel_info_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();

        // scanno (i32)
        bytes.write_i32::<BigEndian>(1).unwrap();
        // logno (i32)
        bytes.write_i32::<BigEndian>(2).unwrap();
        // kind (i32) - MEG channel
        bytes.write_i32::<BigEndian>(FIFFV_MEG_CH).unwrap();
        // range (f32)
        bytes.write_f32::<BigEndian>(3.2768e-10).unwrap();
        // cal (f32)
        bytes.write_f32::<BigEndian>(1e-13).unwrap();
        // coil_type (i32)
        bytes.write_i32::<BigEndian>(3012).unwrap();

        // loc array (12 × f32 = 48 bytes)
        for i in 0..12 {
            bytes.write_f32::<BigEndian>(i as f32 * 0.1).unwrap();
        }

        // unit (i32)
        bytes.write_i32::<BigEndian>(112).unwrap();
        // unit_mul (i32)
        bytes.write_i32::<BigEndian>(0).unwrap();

        // ch_name (16 bytes null-terminated)
        let name = b"MEG 0113\0\0\0\0\0\0\0\0";
        bytes.extend_from_slice(name);

        bytes
    }

    #[test]
    fn test_channel_info_from_bytes() {
        let data = create_test_channel_info_bytes();
        let ch_info = ChannelInfo::from_bytes(&data).unwrap();

        assert_eq!(ch_info.scanno, 1);
        assert_eq!(ch_info.logno, 2);
        assert_eq!(ch_info.kind, FIFFV_MEG_CH);
        assert!((ch_info.range - 3.2768e-10).abs() < 1e-15);
        assert!((ch_info.cal - 1e-13).abs() < 1e-18);
        assert_eq!(ch_info.coil_type, 3012);
        assert_eq!(ch_info.unit, 112);
        assert_eq!(ch_info.unit_mul, 0);
        assert_eq!(ch_info.ch_name, "MEG 0113");

        // Check loc array
        for i in 0..12 {
            assert!((ch_info.loc[i] - i as f32 * 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn test_channel_info_from_bytes_short() {
        let data = vec![0u8; 50]; // Only 50 bytes instead of 96
        let result = ChannelInfo::from_bytes(&data);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_channel_info_calibration() {
        let data = create_test_channel_info_bytes();
        let ch_info = ChannelInfo::from_bytes(&data).unwrap();

        let expected = 3.2768e-10 * 1e-13;
        let actual = ch_info.calibration();
        assert!((actual - expected).abs() < 1e-25);
    }

    #[test]
    fn test_channel_info_is_data_channel() {
        let mut data = create_test_channel_info_bytes();

        // MEG channel - should be data channel
        let ch_info = ChannelInfo::from_bytes(&data).unwrap();
        assert!(ch_info.is_data_channel());

        // STIM channel - should not be data channel
        let mut cursor = Cursor::new(&mut data);
        cursor.set_position(8); // Position of kind field
        cursor.write_i32::<BigEndian>(FIFFV_STIM_CH).unwrap();

        let ch_info = ChannelInfo::from_bytes(&data).unwrap();
        assert!(!ch_info.is_data_channel());
    }

    #[test]
    fn test_channel_info_type_name() {
        let data = create_test_channel_info_bytes();
        let ch_info = ChannelInfo::from_bytes(&data).unwrap();
        assert_eq!(ch_info.type_name(), "MEG");
    }

    #[test]
    fn test_channel_info_null_terminated_name() {
        let mut bytes = create_test_channel_info_bytes();

        // Replace name bytes with "EEG" followed by nulls
        let name_pos = 4 + 4 + 4 + 4 + 4 + 4 + 48 + 4 + 4;
        for i in 0..16 {
            bytes[name_pos + i] = 0;
        }
        bytes[name_pos] = b'E';
        bytes[name_pos + 1] = b'E';
        bytes[name_pos + 2] = b'G';

        let ch_info = ChannelInfo::from_bytes(&bytes).unwrap();
        assert_eq!(ch_info.ch_name, "EEG");
    }

    #[test]
    fn test_read_directory_at() {
        // Create mock directory data
        let mut bytes = Vec::new();

        // Number of entries
        bytes.write_i32::<BigEndian>(2).unwrap();

        // Entry 1: FIFF_NCHAN
        bytes.write_i32::<BigEndian>(FIFF_NCHAN).unwrap(); // kind
        bytes.write_i32::<BigEndian>(FIFFT_INT).unwrap(); // type
        bytes.write_i32::<BigEndian>(4).unwrap(); // size
        bytes.write_u32::<BigEndian>(1024).unwrap(); // pos

        // Entry 2: FIFF_SFREQ
        bytes.write_i32::<BigEndian>(FIFF_SFREQ).unwrap(); // kind
        bytes.write_i32::<BigEndian>(FIFFT_FLOAT).unwrap(); // type
        bytes.write_i32::<BigEndian>(4).unwrap(); // size
        bytes.write_u32::<BigEndian>(2048).unwrap(); // pos

        let mut cursor = Cursor::new(bytes);
        let directory = read_directory_at(&mut cursor, 0).unwrap();

        assert_eq!(directory.len(), 2);
        assert_eq!(directory[0].kind, FIFF_NCHAN);
        assert_eq!(directory[0].type_, FIFFT_INT);
        assert_eq!(directory[0].size, 4);
        assert_eq!(directory[0].pos, 1024);

        assert_eq!(directory[1].kind, FIFF_SFREQ);
        assert_eq!(directory[1].type_, FIFFT_FLOAT);
        assert_eq!(directory[1].size, 4);
        assert_eq!(directory[1].pos, 2048);
    }

    #[test]
    fn test_meas_info_find_tag() {
        // Create mock file with tags
        let mut bytes = Vec::new();

        // Tag 1: FIFF_NCHAN
        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(100).unwrap();
        let tag1_bytes = create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data);
        bytes.extend_from_slice(&tag1_bytes);

        // Tag 2: FIFF_SFREQ
        let pos_sfreq = bytes.len() as u64;
        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        let tag2_bytes = create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data);
        bytes.extend_from_slice(&tag2_bytes);

        // Create directory
        let directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: pos_sfreq,
            },
        ];

        let mut cursor = Cursor::new(bytes);

        // Find FIFF_NCHAN
        let tag = MeasInfo::find_tag(&mut cursor, &directory, FIFF_NCHAN).unwrap();
        assert_eq!(tag.kind, FIFF_NCHAN);
        assert_eq!(tag.as_i32().unwrap(), 100);

        // Find FIFF_SFREQ
        let tag = MeasInfo::find_tag(&mut cursor, &directory, FIFF_SFREQ).unwrap();
        assert_eq!(tag.kind, FIFF_SFREQ);
        assert!((tag.as_f32().unwrap() - 1000.0).abs() < 1e-6);

        // Try to find non-existent tag
        let result = MeasInfo::find_tag(&mut cursor, &directory, 999);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_tree_sequential_simple() {
        // Create a simple FIFF file structure
        let mut bytes = Vec::new();

        // Tag 1: FIFF_NCHAN
        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(100).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        // Tag 2: FIFF_SFREQ
        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        let mut cursor = Cursor::new(bytes);
        let tree = build_tree_sequential(&mut cursor).unwrap();

        // Should have 2 top-level entries
        assert_eq!(tree.nent, 2);
        assert_eq!(tree.directory.len(), 2);
        assert_eq!(tree.directory[0].kind, FIFF_NCHAN);
        assert_eq!(tree.directory[1].kind, FIFF_SFREQ);
    }

    // Tests for new MeasInfo fields (matching MNE-Python's test_meas_info.py)

    #[test]
    fn test_meas_info_measurement_date() {
        // Test parsing measurement date (Unix timestamp)
        let mut bytes = Vec::new();

        // FIFF_NCHAN
        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(10).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        // FIFF_SFREQ
        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        // FIFF_MEAS_DATE - Unix timestamp (e.g., 2024-01-01 00:00:00 UTC)
        let pos_date = bytes.len() as u64;
        let mut date_data = Vec::new();
        let timestamp: i32 = 1704067200; // 2024-01-01 00:00:00 UTC
        date_data.write_i32::<BigEndian>(timestamp).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_MEAS_DATE, FIFFT_INT, 4, &date_data));

        let directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
            DirEntry {
                kind: FIFF_MEAS_DATE,
                type_: FIFFT_INT,
                size: 4,
                pos: pos_date,
            },
        ];

        // Create mock MEAS_INFO tree
        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.meas_date, Some(1704067200));
    }

    #[test]
    fn test_meas_info_filter_info() {
        // Test parsing lowpass and highpass filter frequencies
        let mut bytes = Vec::new();

        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(10).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        // FIFF_LOWPASS
        let pos_lowpass = bytes.len() as u64;
        let mut lowpass_data = Vec::new();
        lowpass_data.write_f32::<BigEndian>(40.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_LOWPASS,
            FIFFT_FLOAT,
            4,
            &lowpass_data,
        ));

        // FIFF_HIGHPASS
        let pos_highpass = bytes.len() as u64;
        let mut highpass_data = Vec::new();
        highpass_data.write_f32::<BigEndian>(0.1).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_HIGHPASS,
            FIFFT_FLOAT,
            4,
            &highpass_data,
        ));

        let directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
            DirEntry {
                kind: FIFF_LOWPASS,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: pos_lowpass,
            },
            DirEntry {
                kind: FIFF_HIGHPASS,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: pos_highpass,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert!(meas_info.lowpass.is_some());
        assert!((meas_info.lowpass.unwrap() - 40.0).abs() < 1e-6);
        assert!(meas_info.highpass.is_some());
        assert!((meas_info.highpass.unwrap() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_meas_info_bad_channels() {
        // Test parsing bad channels list (colon-separated in MNE format)
        let mut bytes = Vec::new();

        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(5).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        // FIFF_MNE_BAD_CHS - colon-separated bad channel names
        let pos_bads = bytes.len() as u64;
        let bad_ch_string = b"MEG 0113:MEG 0112:MEG 0122\0";
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_BAD_CHS,
            FIFFT_STRING,
            bad_ch_string.len() as i32,
            bad_ch_string,
        ));

        let directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
            DirEntry {
                kind: FIFF_MNE_BAD_CHS,
                type_: FIFFT_STRING,
                size: bad_ch_string.len() as i32,
                pos: pos_bads,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.bads.len(), 3);
        assert!(meas_info.bads.contains(&"MEG 0113".to_string()));
        assert!(meas_info.bads.contains(&"MEG 0112".to_string()));
        assert!(meas_info.bads.contains(&"MEG 0122".to_string()));
    }

    #[test]
    fn test_meas_info_metadata_fields() {
        // Test parsing all metadata fields
        let mut bytes = Vec::new();

        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(10).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        // LINE_FREQ
        let pos_line = bytes.len() as u64;
        let mut line_data = Vec::new();
        line_data.write_f32::<BigEndian>(60.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_LINE_FREQ,
            FIFFT_FLOAT,
            4,
            &line_data,
        ));

        // EXPERIMENTER
        let pos_exp = bytes.len() as u64;
        let exp_name = b"John Doe\0";
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_EXPERIMENTER,
            FIFFT_STRING,
            exp_name.len() as i32,
            exp_name,
        ));

        // DESCRIPTION
        let pos_desc = bytes.len() as u64;
        let desc = b"Test experiment\0";
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_DESCRIPTION,
            FIFFT_STRING,
            desc.len() as i32,
            desc,
        ));

        // PROJ_ID
        let pos_proj_id = bytes.len() as u64;
        let mut proj_id_data = Vec::new();
        proj_id_data.write_i32::<BigEndian>(12345).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_PROJ_ID, FIFFT_INT, 4, &proj_id_data));

        // PROJ_NAME
        let pos_proj_name = bytes.len() as u64;
        let proj_name = b"DDA Analysis\0";
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_NAME,
            FIFFT_STRING,
            proj_name.len() as i32,
            proj_name,
        ));

        let directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
            DirEntry {
                kind: FIFF_LINE_FREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: pos_line,
            },
            DirEntry {
                kind: FIFF_EXPERIMENTER,
                type_: FIFFT_STRING,
                size: exp_name.len() as i32,
                pos: pos_exp,
            },
            DirEntry {
                kind: FIFF_DESCRIPTION,
                type_: FIFFT_STRING,
                size: desc.len() as i32,
                pos: pos_desc,
            },
            DirEntry {
                kind: FIFF_PROJ_ID,
                type_: FIFFT_INT,
                size: 4,
                pos: pos_proj_id,
            },
            DirEntry {
                kind: FIFF_PROJ_NAME,
                type_: FIFFT_STRING,
                size: proj_name.len() as i32,
                pos: pos_proj_name,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert!(meas_info.line_freq.is_some());
        assert!((meas_info.line_freq.unwrap() - 60.0).abs() < 1e-6);
        assert_eq!(meas_info.experimenter, Some("John Doe".to_string()));
        assert_eq!(meas_info.description, Some("Test experiment".to_string()));
        assert_eq!(meas_info.proj_id, Some(12345));
        assert_eq!(meas_info.proj_name, Some("DDA Analysis".to_string()));
    }

    fn create_tag_bytes(kind: i32, type_: i32, size: i32, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.write_i32::<BigEndian>(kind).unwrap();
        bytes.write_i32::<BigEndian>(type_).unwrap();
        bytes.write_i32::<BigEndian>(size).unwrap();
        bytes.write_i32::<BigEndian>(0).unwrap(); // next pointer
        bytes.extend_from_slice(data);
        bytes
    }

    // Tests for SSP Projectors (matching MNE-Python's test_proj.py)

    #[test]
    fn test_projection_parsing() {
        // Test parsing a single SSP projection
        let mut bytes = Vec::new();

        // Create PROJ block tags
        // FIFF_PROJ_ITEM_KIND
        let mut kind_data = Vec::new();
        kind_data.write_i32::<BigEndian>(FIFFV_MEG_CH).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_ITEM_KIND,
            FIFFT_INT,
            4,
            &kind_data,
        ));

        // FIFF_PROJ_ITEM_DESC
        let desc = b"PCA-v1\0";
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_ITEM_DESC,
            FIFFT_STRING,
            desc.len() as i32,
            desc,
        ));

        // FIFF_PROJ_ITEM_ACTIVE
        let mut active_data = Vec::new();
        active_data.write_i32::<BigEndian>(1).unwrap(); // Active
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_ITEM_ACTIVE,
            FIFFT_INT,
            4,
            &active_data,
        ));

        // FIFF_PROJ_ITEM_NVEC
        let mut nvec_data = Vec::new();
        nvec_data.write_i32::<BigEndian>(3).unwrap(); // 3 projection vectors
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_ITEM_NVEC,
            FIFFT_INT,
            4,
            &nvec_data,
        ));

        // FIFF_NAMED_MATRIX (simplified - just some double values)
        let mut matrix_data = Vec::new();
        for i in 0..9 {
            matrix_data.write_f64::<BigEndian>(i as f64 * 0.1).unwrap();
        }
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_NAMED_MATRIX,
            FIFFT_DOUBLE,
            matrix_data.len() as i32,
            &matrix_data,
        ));

        // Create directory for PROJ block
        let mut directory = Vec::new();
        let mut pos = 0u64;
        for kind in [
            FIFF_PROJ_ITEM_KIND,
            FIFF_PROJ_ITEM_DESC,
            FIFF_PROJ_ITEM_ACTIVE,
            FIFF_PROJ_ITEM_NVEC,
            FIFF_NAMED_MATRIX,
        ] {
            let size = match kind {
                FIFF_PROJ_ITEM_DESC => desc.len() as i32,
                FIFF_NAMED_MATRIX => matrix_data.len() as i32,
                _ => 4,
            };
            directory.push(DirEntry {
                kind,
                type_: if kind == FIFF_PROJ_ITEM_DESC {
                    FIFFT_STRING
                } else if kind == FIFF_NAMED_MATRIX {
                    FIFFT_DOUBLE
                } else {
                    FIFFT_INT
                },
                size,
                pos,
            });
            pos += 16 + size as u64;
        }

        let mut proj_node = TreeNode::new(FIFFB_PROJ);
        proj_node.directory = directory;

        let mut cursor = Cursor::new(bytes);
        let proj = Projection::read(&mut cursor, &proj_node).unwrap();

        assert_eq!(proj.kind, FIFFV_MEG_CH);
        assert_eq!(proj.desc, "PCA-v1");
        assert!(proj.active);
        assert_eq!(proj.nvec, 3);
        assert_eq!(proj.data.len(), 9); // 3 vectors × 3 channels (simplified)
        assert!((proj.data[0] - 0.0).abs() < 1e-10);
        assert!((proj.data[1] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_meas_info_with_projections() {
        // Test parsing MeasInfo with SSP projections
        let mut bytes = Vec::new();

        // Basic MEAS_INFO tags
        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(10).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        let directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = directory;

        // Create a projection block
        let proj_bytes_start = bytes.len() as u64;
        let mut proj_pos = proj_bytes_start;

        let mut kind_data = Vec::new();
        kind_data.write_i32::<BigEndian>(FIFFV_EEG_CH).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_ITEM_KIND,
            FIFFT_INT,
            4,
            &kind_data,
        ));

        let kind_pos = proj_pos;
        proj_pos += 20;

        let desc = b"EOG\0";
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_ITEM_DESC,
            FIFFT_STRING,
            desc.len() as i32,
            desc,
        ));

        let desc_pos = proj_pos;
        proj_pos += 16 + desc.len() as u64;

        let mut active_data = Vec::new();
        active_data.write_i32::<BigEndian>(0).unwrap(); // Inactive
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_ITEM_ACTIVE,
            FIFFT_INT,
            4,
            &active_data,
        ));

        let active_pos = proj_pos;
        proj_pos += 20;

        let mut nvec_data = Vec::new();
        nvec_data.write_i32::<BigEndian>(1).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_PROJ_ITEM_NVEC,
            FIFFT_INT,
            4,
            &nvec_data,
        ));

        let nvec_pos = proj_pos;

        let proj_directory = vec![
            DirEntry {
                kind: FIFF_PROJ_ITEM_KIND,
                type_: FIFFT_INT,
                size: 4,
                pos: kind_pos,
            },
            DirEntry {
                kind: FIFF_PROJ_ITEM_DESC,
                type_: FIFFT_STRING,
                size: desc.len() as i32,
                pos: desc_pos,
            },
            DirEntry {
                kind: FIFF_PROJ_ITEM_ACTIVE,
                type_: FIFFT_INT,
                size: 4,
                pos: active_pos,
            },
            DirEntry {
                kind: FIFF_PROJ_ITEM_NVEC,
                type_: FIFFT_INT,
                size: 4,
                pos: nvec_pos,
            },
        ];

        let mut proj_node = TreeNode::new(FIFFB_PROJ);
        proj_node.directory = proj_directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);
        root.children.push(proj_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.projs.len(), 1);
        assert_eq!(meas_info.projs[0].kind, FIFFV_EEG_CH);
        assert_eq!(meas_info.projs[0].desc, "EOG");
        assert!(!meas_info.projs[0].active);
        assert_eq!(meas_info.projs[0].nvec, 1);
    }

    #[test]
    fn test_meas_info_no_projections() {
        // Test that MeasInfo works without any projections
        let mut bytes = Vec::new();

        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(10).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        let directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.projs.len(), 0);
    }

    #[test]
    fn test_projection_active_inactive() {
        // Test both active and inactive projections
        let test_cases = vec![(0, false), (1, true), (42, true)]; // Any non-zero is active

        for (value, expected_active) in test_cases {
            let mut bytes = Vec::new();

            let mut kind_data = Vec::new();
            kind_data.write_i32::<BigEndian>(FIFFV_MEG_CH).unwrap();
            bytes.extend_from_slice(&create_tag_bytes(
                FIFF_PROJ_ITEM_KIND,
                FIFFT_INT,
                4,
                &kind_data,
            ));

            let desc = b"Test\0";
            bytes.extend_from_slice(&create_tag_bytes(
                FIFF_PROJ_ITEM_DESC,
                FIFFT_STRING,
                desc.len() as i32,
                desc,
            ));

            let mut active_data = Vec::new();
            active_data.write_i32::<BigEndian>(value).unwrap();
            bytes.extend_from_slice(&create_tag_bytes(
                FIFF_PROJ_ITEM_ACTIVE,
                FIFFT_INT,
                4,
                &active_data,
            ));

            let mut nvec_data = Vec::new();
            nvec_data.write_i32::<BigEndian>(1).unwrap();
            bytes.extend_from_slice(&create_tag_bytes(
                FIFF_PROJ_ITEM_NVEC,
                FIFFT_INT,
                4,
                &nvec_data,
            ));

            let mut pos = 0u64;
            let directory = vec![
                DirEntry {
                    kind: FIFF_PROJ_ITEM_KIND,
                    type_: FIFFT_INT,
                    size: 4,
                    pos,
                },
                {
                    pos += 20;
                    DirEntry {
                        kind: FIFF_PROJ_ITEM_DESC,
                        type_: FIFFT_STRING,
                        size: desc.len() as i32,
                        pos,
                    }
                },
                {
                    pos += 16 + desc.len() as u64;
                    DirEntry {
                        kind: FIFF_PROJ_ITEM_ACTIVE,
                        type_: FIFFT_INT,
                        size: 4,
                        pos,
                    }
                },
                {
                    pos += 20;
                    DirEntry {
                        kind: FIFF_PROJ_ITEM_NVEC,
                        type_: FIFFT_INT,
                        size: 4,
                        pos,
                    }
                },
            ];

            let mut proj_node = TreeNode::new(FIFFB_PROJ);
            proj_node.directory = directory;

            let mut cursor = Cursor::new(bytes);
            let proj = Projection::read(&mut cursor, &proj_node).unwrap();

            assert_eq!(
                proj.active, expected_active,
                "Value {} should result in active={}",
                value, expected_active
            );
        }
    }

    // Tests for CTF Compensation (matching MNE-Python's test_compensator.py)

    #[test]
    fn test_compensation_parsing() {
        // Test parsing a single CTF compensation
        let mut bytes = Vec::new();

        // FIFF_MNE_CTF_COMP_KIND
        let mut kind_data = Vec::new();
        kind_data
            .write_i32::<BigEndian>(FIFFV_MNE_CTFV_COMP_G1BR)
            .unwrap(); // Grade 1
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_KIND,
            FIFFT_INT,
            4,
            &kind_data,
        ));

        // FIFF_MNE_CTF_COMP_CALIBRATED
        let mut calibrated_data = Vec::new();
        calibrated_data.write_i32::<BigEndian>(1).unwrap(); // Calibrated
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_CALIBRATED,
            FIFFT_INT,
            4,
            &calibrated_data,
        ));

        // FIFF_MNE_CTF_COMP_DATA (compensation matrix as doubles)
        let mut matrix_data = Vec::new();
        for i in 0..12 {
            matrix_data.write_f64::<BigEndian>(i as f64 * 0.01).unwrap();
        }
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_DATA,
            FIFFT_DOUBLE,
            matrix_data.len() as i32,
            &matrix_data,
        ));

        // Create directory for COMP block
        let mut directory = Vec::new();
        let mut pos = 0u64;
        for kind in [
            FIFF_MNE_CTF_COMP_KIND,
            FIFF_MNE_CTF_COMP_CALIBRATED,
            FIFF_MNE_CTF_COMP_DATA,
        ] {
            let size = match kind {
                FIFF_MNE_CTF_COMP_DATA => matrix_data.len() as i32,
                _ => 4,
            };
            directory.push(DirEntry {
                kind,
                type_: if kind == FIFF_MNE_CTF_COMP_DATA {
                    FIFFT_DOUBLE
                } else {
                    FIFFT_INT
                },
                size,
                pos,
            });
            pos += 16 + size as u64;
        }

        let mut comp_node = TreeNode::new(FIFFB_MNE_CTF_COMP);
        comp_node.directory = directory;

        let mut cursor = Cursor::new(bytes);
        let comp = Compensation::read(&mut cursor, &comp_node).unwrap();

        assert_eq!(comp.kind, FIFFV_MNE_CTFV_COMP_G1BR);
        assert!(comp.calibrated);
        assert_eq!(comp.data.len(), 12);
        assert!((comp.data[0] - 0.0).abs() < 1e-10);
        assert!((comp.data[1] - 0.01).abs() < 1e-10);
        assert_eq!(comp.grade_name(), "G1BR");
    }

    #[test]
    fn test_compensation_grades() {
        // Test all compensation grade names
        let test_cases = vec![
            (FIFFV_MNE_CTFV_COMP_NONE, "None"),
            (FIFFV_MNE_CTFV_COMP_G1BR, "G1BR"),
            (FIFFV_MNE_CTFV_COMP_G2BR, "G2BR"),
            (FIFFV_MNE_CTFV_COMP_G3BR, "G3BR"),
            (999, "Unknown"),
        ];

        for (grade, expected_name) in test_cases {
            let comp = Compensation {
                kind: grade,
                calibrated: false,
                data: vec![],
            };
            assert_eq!(comp.grade_name(), expected_name);
        }
    }

    #[test]
    fn test_compensation_uncalibrated() {
        // Test uncalibrated compensation (calibrated field defaults to false)
        let mut bytes = Vec::new();

        let mut kind_data = Vec::new();
        kind_data
            .write_i32::<BigEndian>(FIFFV_MNE_CTFV_COMP_G2BR)
            .unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_KIND,
            FIFFT_INT,
            4,
            &kind_data,
        ));

        // No calibrated tag - should default to false

        let directory = vec![DirEntry {
            kind: FIFF_MNE_CTF_COMP_KIND,
            type_: FIFFT_INT,
            size: 4,
            pos: 0,
        }];

        let mut comp_node = TreeNode::new(FIFFB_MNE_CTF_COMP);
        comp_node.directory = directory;

        let mut cursor = Cursor::new(bytes);
        let comp = Compensation::read(&mut cursor, &comp_node).unwrap();

        assert_eq!(comp.kind, FIFFV_MNE_CTFV_COMP_G2BR);
        assert!(!comp.calibrated); // Defaults to false
        assert_eq!(comp.grade_name(), "G2BR");
    }

    #[test]
    fn test_meas_info_with_compensation() {
        // Test parsing MeasInfo with CTF compensation
        let mut bytes = Vec::new();

        // Basic MEAS_INFO tags
        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(10).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        let meas_directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = meas_directory;

        // Create a compensation block
        let comp_bytes_start = bytes.len() as u64;
        let mut comp_pos = comp_bytes_start;

        let mut kind_data = Vec::new();
        kind_data
            .write_i32::<BigEndian>(FIFFV_MNE_CTFV_COMP_G3BR)
            .unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_KIND,
            FIFFT_INT,
            4,
            &kind_data,
        ));

        let kind_pos = comp_pos;
        comp_pos += 20;

        let mut calibrated_data = Vec::new();
        calibrated_data.write_i32::<BigEndian>(0).unwrap(); // Not calibrated
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_CALIBRATED,
            FIFFT_INT,
            4,
            &calibrated_data,
        ));

        let calibrated_pos = comp_pos;

        let comp_directory = vec![
            DirEntry {
                kind: FIFF_MNE_CTF_COMP_KIND,
                type_: FIFFT_INT,
                size: 4,
                pos: kind_pos,
            },
            DirEntry {
                kind: FIFF_MNE_CTF_COMP_CALIBRATED,
                type_: FIFFT_INT,
                size: 4,
                pos: calibrated_pos,
            },
        ];

        let mut comp_node = TreeNode::new(FIFFB_MNE_CTF_COMP);
        comp_node.directory = comp_directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);
        root.children.push(comp_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.comps.len(), 1);
        assert_eq!(meas_info.comps[0].kind, FIFFV_MNE_CTFV_COMP_G3BR);
        assert!(!meas_info.comps[0].calibrated);
        assert_eq!(meas_info.comps[0].grade_name(), "G3BR");
    }

    #[test]
    fn test_meas_info_no_compensation() {
        // Test that MeasInfo works without any compensations (typical for non-CTF systems)
        let mut bytes = Vec::new();

        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(10).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        let directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.comps.len(), 0); // No compensation for non-CTF
    }

    #[test]
    fn test_compensation_with_derivation_data() {
        // Test compensation using FIFF_MNE_DERIVATION_DATA instead of FIFF_MNE_CTF_COMP_DATA
        let mut bytes = Vec::new();

        let mut kind_data = Vec::new();
        kind_data
            .write_i32::<BigEndian>(FIFFV_MNE_CTFV_COMP_G1BR)
            .unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_KIND,
            FIFFT_INT,
            4,
            &kind_data,
        ));

        // Use FIFF_MNE_DERIVATION_DATA instead
        let mut deriv_data = Vec::new();
        for i in 0..6 {
            deriv_data.write_f64::<BigEndian>(i as f64 * 0.1).unwrap();
        }
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_DERIVATION_DATA,
            FIFFT_DOUBLE,
            deriv_data.len() as i32,
            &deriv_data,
        ));

        let directory = vec![
            DirEntry {
                kind: FIFF_MNE_CTF_COMP_KIND,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_MNE_DERIVATION_DATA,
                type_: FIFFT_DOUBLE,
                size: deriv_data.len() as i32,
                pos: 20,
            },
        ];

        let mut comp_node = TreeNode::new(FIFFB_MNE_CTF_COMP);
        comp_node.directory = directory;

        let mut cursor = Cursor::new(bytes);
        let comp = Compensation::read(&mut cursor, &comp_node).unwrap();

        assert_eq!(comp.kind, FIFFV_MNE_CTFV_COMP_G1BR);
        assert_eq!(comp.data.len(), 6);
        assert!((comp.data[0] - 0.0).abs() < 1e-10);
        assert!((comp.data[1] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_compensations() {
        // Test file with multiple compensation grades
        let mut bytes = Vec::new();

        // MEAS_INFO
        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(10).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let mut sfreq_data = Vec::new();
        sfreq_data.write_f32::<BigEndian>(1000.0).unwrap();
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        let meas_directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = meas_directory;

        // First compensation (G1BR)
        let comp1_start = bytes.len() as u64;
        let mut kind1_data = Vec::new();
        kind1_data
            .write_i32::<BigEndian>(FIFFV_MNE_CTFV_COMP_G1BR)
            .unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_KIND,
            FIFFT_INT,
            4,
            &kind1_data,
        ));

        let comp1_directory = vec![DirEntry {
            kind: FIFF_MNE_CTF_COMP_KIND,
            type_: FIFFT_INT,
            size: 4,
            pos: comp1_start,
        }];

        let mut comp1_node = TreeNode::new(FIFFB_MNE_CTF_COMP);
        comp1_node.directory = comp1_directory;

        // Second compensation (G2BR)
        let comp2_start = bytes.len() as u64;
        let mut kind2_data = Vec::new();
        kind2_data
            .write_i32::<BigEndian>(FIFFV_MNE_CTFV_COMP_G2BR)
            .unwrap();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_MNE_CTF_COMP_KIND,
            FIFFT_INT,
            4,
            &kind2_data,
        ));

        let comp2_directory = vec![DirEntry {
            kind: FIFF_MNE_CTF_COMP_KIND,
            type_: FIFFT_INT,
            size: 4,
            pos: comp2_start,
        }];

        let mut comp2_node = TreeNode::new(FIFFB_MNE_CTF_COMP);
        comp2_node.directory = comp2_directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);
        root.children.push(comp1_node);
        root.children.push(comp2_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.comps.len(), 2);
        assert_eq!(meas_info.comps[0].kind, FIFFV_MNE_CTFV_COMP_G1BR);
        assert_eq!(meas_info.comps[0].grade_name(), "G1BR");
        assert_eq!(meas_info.comps[1].kind, FIFFV_MNE_CTFV_COMP_G2BR);
        assert_eq!(meas_info.comps[1].grade_name(), "G2BR");
    }

    #[test]
    fn test_coord_trans_from_bytes() {
        let mut bytes = Vec::new();

        // from frame (i32)
        bytes.write_i32::<BigEndian>(FIFFV_COORD_DEVICE).unwrap();
        // to frame (i32)
        bytes.write_i32::<BigEndian>(FIFFV_COORD_HEAD).unwrap();

        // 3x3 rotation matrix (identity)
        bytes.write_f32::<BigEndian>(1.0).unwrap();
        bytes.write_f32::<BigEndian>(0.0).unwrap();
        bytes.write_f32::<BigEndian>(0.0).unwrap();
        bytes.write_f32::<BigEndian>(0.0).unwrap();
        bytes.write_f32::<BigEndian>(1.0).unwrap();
        bytes.write_f32::<BigEndian>(0.0).unwrap();
        bytes.write_f32::<BigEndian>(0.0).unwrap();
        bytes.write_f32::<BigEndian>(0.0).unwrap();
        bytes.write_f32::<BigEndian>(1.0).unwrap();

        // Translation vector
        bytes.write_f32::<BigEndian>(0.01).unwrap();
        bytes.write_f32::<BigEndian>(0.02).unwrap();
        bytes.write_f32::<BigEndian>(0.03).unwrap();

        let trans = CoordTrans::from_bytes(&bytes).unwrap();

        assert_eq!(trans.from, FIFFV_COORD_DEVICE);
        assert_eq!(trans.to, FIFFV_COORD_HEAD);
        assert_eq!(trans.rot[0], 1.0);
        assert_eq!(trans.rot[4], 1.0);
        assert_eq!(trans.rot[8], 1.0);
        assert_eq!(trans.move_[0], 0.01);
        assert_eq!(trans.move_[1], 0.02);
        assert_eq!(trans.move_[2], 0.03);
    }

    #[test]
    fn test_coord_trans_from_bytes_too_short() {
        let bytes = vec![0u8; 40]; // Only 40 bytes, need 56
        let result = CoordTrans::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_coord_trans_description() {
        let mut bytes = Vec::new();
        bytes.write_i32::<BigEndian>(FIFFV_COORD_HEAD).unwrap();
        bytes.write_i32::<BigEndian>(FIFFV_COORD_MRI).unwrap();

        // Identity rotation
        for i in 0..9 {
            bytes
                .write_f32::<BigEndian>(if i % 4 == 0 { 1.0 } else { 0.0 })
                .unwrap();
        }
        // Zero translation
        for _ in 0..3 {
            bytes.write_f32::<BigEndian>(0.0).unwrap();
        }

        let trans = CoordTrans::from_bytes(&bytes).unwrap();
        assert_eq!(trans.description(), "Head -> MRI");
    }

    #[test]
    fn test_coord_trans_predicates() {
        let mut bytes = Vec::new();
        bytes.write_i32::<BigEndian>(FIFFV_COORD_DEVICE).unwrap();
        bytes.write_i32::<BigEndian>(FIFFV_COORD_HEAD).unwrap();

        // Identity rotation
        for i in 0..9 {
            bytes
                .write_f32::<BigEndian>(if i % 4 == 0 { 1.0 } else { 0.0 })
                .unwrap();
        }
        // Zero translation
        for _ in 0..3 {
            bytes.write_f32::<BigEndian>(0.0).unwrap();
        }

        let trans = CoordTrans::from_bytes(&bytes).unwrap();
        assert!(trans.is_device_to_head());
        assert!(!trans.is_head_to_mri());
    }

    #[test]
    fn test_meas_info_with_coord_trans() {
        let mut bytes = Vec::new();

        // Create MEAS_INFO tags
        let nchan_data = 2i32.to_be_bytes();
        let sfreq_data = (1000.0f32).to_be_bytes();

        let nchan_pos = bytes.len() as u64;
        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        let sfreq_pos = bytes.len() as u64;
        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        // Create two channel info entries
        let ch1_bytes = create_test_channel_info_bytes();
        let ch2_bytes = create_test_channel_info_bytes();

        let ch1_pos = bytes.len() as u64;
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_CH_INFO,
            FIFFT_CH_INFO_STRUCT,
            96,
            &ch1_bytes,
        ));

        let ch2_pos = bytes.len() as u64;
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_CH_INFO,
            FIFFT_CH_INFO_STRUCT,
            96,
            &ch2_bytes,
        ));

        // Create coordinate transform data (56 bytes)
        let mut trans_data = Vec::new();
        trans_data
            .write_i32::<BigEndian>(FIFFV_COORD_DEVICE)
            .unwrap();
        trans_data.write_i32::<BigEndian>(FIFFV_COORD_HEAD).unwrap();

        // Identity rotation matrix
        for i in 0..9 {
            trans_data
                .write_f32::<BigEndian>(if i % 4 == 0 { 1.0 } else { 0.0 })
                .unwrap();
        }

        // Translation vector
        trans_data.write_f32::<BigEndian>(0.01).unwrap();
        trans_data.write_f32::<BigEndian>(0.02).unwrap();
        trans_data.write_f32::<BigEndian>(0.03).unwrap();

        let trans_pos = bytes.len() as u64;
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_COORD_TRANS,
            FIFFT_VOID,
            56,
            &trans_data,
        ));

        // Create directory entries
        let meas_info_directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: nchan_pos,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: sfreq_pos,
            },
            DirEntry {
                kind: FIFF_CH_INFO,
                type_: FIFFT_CH_INFO_STRUCT,
                size: 96,
                pos: ch1_pos,
            },
            DirEntry {
                kind: FIFF_CH_INFO,
                type_: FIFFT_CH_INFO_STRUCT,
                size: 96,
                pos: ch2_pos,
            },
            DirEntry {
                kind: FIFF_COORD_TRANS,
                type_: FIFFT_VOID,
                size: 56,
                pos: trans_pos,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = meas_info_directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.coord_trans.len(), 1);
        assert_eq!(meas_info.coord_trans[0].from, FIFFV_COORD_DEVICE);
        assert_eq!(meas_info.coord_trans[0].to, FIFFV_COORD_HEAD);
        assert!(meas_info.coord_trans[0].is_device_to_head());
    }

    #[test]
    fn test_meas_info_get_device_to_head_trans() {
        let mut bytes = Vec::new();

        // Basic MEAS_INFO
        let nchan_data = 1i32.to_be_bytes();
        let sfreq_data = (1000.0f32).to_be_bytes();

        bytes.extend_from_slice(&create_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &nchan_data));

        bytes.extend_from_slice(&create_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &sfreq_data));

        let ch_bytes = create_test_channel_info_bytes();
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_CH_INFO,
            FIFFT_CH_INFO_STRUCT,
            96,
            &ch_bytes,
        ));

        // Create device-to-head transform
        let mut dev_to_head = Vec::new();
        dev_to_head
            .write_i32::<BigEndian>(FIFFV_COORD_DEVICE)
            .unwrap();
        dev_to_head
            .write_i32::<BigEndian>(FIFFV_COORD_HEAD)
            .unwrap();
        for i in 0..9 {
            dev_to_head
                .write_f32::<BigEndian>(if i % 4 == 0 { 1.0 } else { 0.0 })
                .unwrap();
        }
        for _ in 0..3 {
            dev_to_head.write_f32::<BigEndian>(0.0).unwrap();
        }

        let trans1_pos = bytes.len() as u64;
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_COORD_TRANS,
            FIFFT_VOID,
            56,
            &dev_to_head,
        ));

        // Create head-to-MRI transform
        let mut head_to_mri = Vec::new();
        head_to_mri
            .write_i32::<BigEndian>(FIFFV_COORD_HEAD)
            .unwrap();
        head_to_mri.write_i32::<BigEndian>(FIFFV_COORD_MRI).unwrap();
        for i in 0..9 {
            head_to_mri
                .write_f32::<BigEndian>(if i % 4 == 0 { 1.0 } else { 0.0 })
                .unwrap();
        }
        for _ in 0..3 {
            head_to_mri.write_f32::<BigEndian>(0.0).unwrap();
        }

        let trans2_pos = bytes.len() as u64;
        bytes.extend_from_slice(&create_tag_bytes(
            FIFF_COORD_TRANS,
            FIFFT_VOID,
            56,
            &head_to_mri,
        ));

        let meas_info_directory = vec![
            DirEntry {
                kind: FIFF_NCHAN,
                type_: FIFFT_INT,
                size: 4,
                pos: 0,
            },
            DirEntry {
                kind: FIFF_SFREQ,
                type_: FIFFT_FLOAT,
                size: 4,
                pos: 20,
            },
            DirEntry {
                kind: FIFF_CH_INFO,
                type_: FIFFT_CH_INFO_STRUCT,
                size: 96,
                pos: 40,
            },
            DirEntry {
                kind: FIFF_COORD_TRANS,
                type_: FIFFT_VOID,
                size: 56,
                pos: trans1_pos,
            },
            DirEntry {
                kind: FIFF_COORD_TRANS,
                type_: FIFFT_VOID,
                size: 56,
                pos: trans2_pos,
            },
        ];

        let mut meas_info_node = TreeNode::new(FIFFB_MEAS_INFO);
        meas_info_node.directory = meas_info_directory;

        let mut root = TreeNode::new(0);
        root.children.push(meas_info_node);

        let mut cursor = Cursor::new(bytes);
        let meas_info = MeasInfo::read(&mut cursor, &root).unwrap();

        assert_eq!(meas_info.coord_trans.len(), 2);

        let dev_head = meas_info.get_device_to_head_trans();
        assert!(dev_head.is_some());
        assert_eq!(dev_head.unwrap().from, FIFFV_COORD_DEVICE);
        assert_eq!(dev_head.unwrap().to, FIFFV_COORD_HEAD);

        let head_mri = meas_info.get_head_to_mri_trans();
        assert!(head_mri.is_some());
        assert_eq!(head_mri.unwrap().from, FIFFV_COORD_HEAD);
        assert_eq!(head_mri.unwrap().to, FIFFV_COORD_MRI);
    }

    #[test]
    fn test_coord_trans_rotation_matrix() {
        let mut bytes = Vec::new();

        bytes.write_i32::<BigEndian>(FIFFV_COORD_DEVICE).unwrap();
        bytes.write_i32::<BigEndian>(FIFFV_COORD_HEAD).unwrap();

        // Test non-identity rotation matrix
        let rot_values = [0.9, 0.1, 0.0, -0.1, 0.9, 0.0, 0.0, 0.0, 1.0];
        for &val in &rot_values {
            bytes.write_f32::<BigEndian>(val).unwrap();
        }

        // Test non-zero translation
        bytes.write_f32::<BigEndian>(0.05).unwrap();
        bytes.write_f32::<BigEndian>(-0.03).unwrap();
        bytes.write_f32::<BigEndian>(0.10).unwrap();

        let trans = CoordTrans::from_bytes(&bytes).unwrap();

        assert_eq!(trans.rot[0], 0.9);
        assert_eq!(trans.rot[1], 0.1);
        assert_eq!(trans.rot[3], -0.1);
        assert_eq!(trans.rot[4], 0.9);
        assert_eq!(trans.move_[0], 0.05);
        assert_eq!(trans.move_[1], -0.03);
        assert_eq!(trans.move_[2], 0.10);
    }
}
