use super::constants::*;
use byteorder::{BigEndian, ReadBytesExt};
/// FIFF Tag reading
/// Based on MNE-Python's _fiff/tag.py
use std::io::{Read, Seek, SeekFrom};

/// FIFF directory entry
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub kind: i32,
    pub type_: i32,
    pub size: i32,
    pub pos: u64,
}

/// FIFF tag
#[derive(Debug)]
pub struct Tag {
    pub kind: i32,
    pub type_: i32,
    pub size: i32,
    pub data: Vec<u8>,
}

impl DirEntry {
    /// Read a directory entry from the file
    pub fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        Ok(DirEntry {
            kind: reader.read_i32::<BigEndian>()?,
            type_: reader.read_i32::<BigEndian>()?,
            size: reader.read_i32::<BigEndian>()?,
            pos: reader.read_u32::<BigEndian>()? as u64,
        })
    }
}

impl Tag {
    /// Read a tag from the file at current position
    pub fn read<R: Read + Seek>(reader: &mut R) -> std::io::Result<Self> {
        let kind = reader.read_i32::<BigEndian>()?;
        let type_ = reader.read_i32::<BigEndian>()?;
        let size = reader.read_i32::<BigEndian>()?;
        let _next = reader.read_i32::<BigEndian>()?; // next tag pointer (unused)

        // Read data
        let mut data = vec![0u8; size as usize];
        reader.read_exact(&mut data)?;

        Ok(Tag {
            kind,
            type_,
            size,
            data,
        })
    }

    /// Read a tag at a specific position
    pub fn read_at<R: Read + Seek>(reader: &mut R, pos: u64) -> std::io::Result<Self> {
        reader.seek(SeekFrom::Start(pos))?;
        Self::read(reader)
    }

    /// Get tag data as i32
    pub fn as_i32(&self) -> std::io::Result<i32> {
        if self.data.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Tag data too small for i32",
            ));
        }
        let mut cursor = std::io::Cursor::new(&self.data);
        cursor.read_i32::<BigEndian>()
    }

    /// Get tag data as f32
    pub fn as_f32(&self) -> std::io::Result<f32> {
        if self.data.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Tag data too small for f32",
            ));
        }
        let mut cursor = std::io::Cursor::new(&self.data);
        cursor.read_f32::<BigEndian>()
    }

    /// Get tag data as f64
    pub fn as_f64(&self) -> std::io::Result<f64> {
        if self.data.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Tag data too small for f64",
            ));
        }
        let mut cursor = std::io::Cursor::new(&self.data);
        cursor.read_f64::<BigEndian>()
    }

    /// Get tag data as string (UTF-8)
    pub fn as_string(&self) -> std::io::Result<String> {
        Ok(String::from_utf8_lossy(&self.data)
            .trim_end_matches('\0')
            .to_string())
    }

    /// Parse data buffer as samples
    pub fn as_samples(&self, nchan: usize) -> std::io::Result<Vec<Vec<f64>>> {
        let type_sz = type_size(self.type_).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown FIFF type: {}", self.type_),
            )
        })?;

        let nsamp = self.size as usize / (type_sz * nchan);
        let mut cursor = std::io::Cursor::new(&self.data);
        let mut samples = vec![vec![0.0f64; nsamp]; nchan];

        match self.type_ {
            FIFFT_SHORT | FIFFT_DAU_PACK16 => {
                for samp_idx in 0..nsamp {
                    for ch_idx in 0..nchan {
                        let val = cursor.read_i16::<BigEndian>()? as f64;
                        samples[ch_idx][samp_idx] = val;
                    }
                }
            }
            FIFFT_INT => {
                for samp_idx in 0..nsamp {
                    for ch_idx in 0..nchan {
                        let val = cursor.read_i32::<BigEndian>()? as f64;
                        samples[ch_idx][samp_idx] = val;
                    }
                }
            }
            FIFFT_FLOAT => {
                for samp_idx in 0..nsamp {
                    for ch_idx in 0..nchan {
                        let val = cursor.read_f32::<BigEndian>()? as f64;
                        samples[ch_idx][samp_idx] = val;
                    }
                }
            }
            FIFFT_DOUBLE => {
                for samp_idx in 0..nsamp {
                    for ch_idx in 0..nchan {
                        let val = cursor.read_f64::<BigEndian>()?;
                        samples[ch_idx][samp_idx] = val;
                    }
                }
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unsupported data type for samples: {}", self.type_),
                ));
            }
        }

        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Cursor;

    fn create_test_tag_bytes(kind: i32, type_: i32, size: i32, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.write_i32::<BigEndian>(kind).unwrap();
        bytes.write_i32::<BigEndian>(type_).unwrap();
        bytes.write_i32::<BigEndian>(size).unwrap();
        bytes.write_i32::<BigEndian>(0).unwrap(); // next pointer
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn test_tag_read_i32() {
        let mut data = Vec::new();
        data.write_i32::<BigEndian>(42).unwrap();

        let bytes = create_test_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        assert_eq!(tag.kind, FIFF_NCHAN);
        assert_eq!(tag.type_, FIFFT_INT);
        assert_eq!(tag.size, 4);
        assert_eq!(tag.as_i32().unwrap(), 42);
    }

    #[test]
    fn test_tag_read_f32() {
        let mut data = Vec::new();
        data.write_f32::<BigEndian>(3.14f32).unwrap();

        let bytes = create_test_tag_bytes(FIFF_SFREQ, FIFFT_FLOAT, 4, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        assert_eq!(tag.kind, FIFF_SFREQ);
        assert_eq!(tag.type_, FIFFT_FLOAT);
        assert!((tag.as_f32().unwrap() - 3.14f32).abs() < 1e-6);
    }

    #[test]
    fn test_tag_read_f64() {
        let mut data = Vec::new();
        data.write_f64::<BigEndian>(2.71828).unwrap();

        let bytes = create_test_tag_bytes(100, FIFFT_DOUBLE, 8, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        assert_eq!(tag.type_, FIFFT_DOUBLE);
        assert!((tag.as_f64().unwrap() - 2.71828).abs() < 1e-10);
    }

    #[test]
    fn test_tag_read_string() {
        let data = b"Hello FIFF\0\0\0\0\0\0";

        let bytes = create_test_tag_bytes(100, FIFFT_STRING, data.len() as i32, data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        assert_eq!(tag.as_string().unwrap(), "Hello FIFF");
    }

    #[test]
    fn test_tag_as_samples_float() {
        // Create 2 channels, 3 samples each
        let mut data = Vec::new();
        // Sample 0: ch0=1.0, ch1=2.0
        data.write_f32::<BigEndian>(1.0).unwrap();
        data.write_f32::<BigEndian>(2.0).unwrap();
        // Sample 1: ch0=3.0, ch1=4.0
        data.write_f32::<BigEndian>(3.0).unwrap();
        data.write_f32::<BigEndian>(4.0).unwrap();
        // Sample 2: ch0=5.0, ch1=6.0
        data.write_f32::<BigEndian>(5.0).unwrap();
        data.write_f32::<BigEndian>(6.0).unwrap();

        let bytes = create_test_tag_bytes(FIFF_DATA_BUFFER, FIFFT_FLOAT, 24, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        let samples = tag.as_samples(2).unwrap();

        assert_eq!(samples.len(), 2); // 2 channels
        assert_eq!(samples[0].len(), 3); // 3 samples
        assert_eq!(samples[1].len(), 3);

        assert!((samples[0][0] - 1.0).abs() < 1e-6);
        assert!((samples[0][1] - 3.0).abs() < 1e-6);
        assert!((samples[0][2] - 5.0).abs() < 1e-6);
        assert!((samples[1][0] - 2.0).abs() < 1e-6);
        assert!((samples[1][1] - 4.0).abs() < 1e-6);
        assert!((samples[1][2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_tag_as_samples_int() {
        let mut data = Vec::new();
        data.write_i32::<BigEndian>(100).unwrap();
        data.write_i32::<BigEndian>(200).unwrap();

        let bytes = create_test_tag_bytes(FIFF_DATA_BUFFER, FIFFT_INT, 8, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        let samples = tag.as_samples(1).unwrap();

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].len(), 2);
        assert_eq!(samples[0][0], 100.0);
        assert_eq!(samples[0][1], 200.0);
    }

    #[test]
    fn test_tag_as_samples_short() {
        let mut data = Vec::new();
        data.write_i16::<BigEndian>(10).unwrap();
        data.write_i16::<BigEndian>(20).unwrap();

        let bytes = create_test_tag_bytes(FIFF_DATA_BUFFER, FIFFT_SHORT, 4, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        let samples = tag.as_samples(1).unwrap();

        assert_eq!(samples[0][0], 10.0);
        assert_eq!(samples[0][1], 20.0);
    }

    #[test]
    fn test_tag_read_at_position() {
        let mut data = Vec::new();
        data.write_i32::<BigEndian>(999).unwrap();

        let bytes = create_test_tag_bytes(FIFF_NCHAN, FIFFT_INT, 4, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read_at(&mut cursor, 0).unwrap();
        assert_eq!(tag.as_i32().unwrap(), 999);
    }

    #[test]
    fn test_dir_entry_read() {
        let mut bytes = Vec::new();
        bytes.write_i32::<BigEndian>(FIFF_NCHAN).unwrap(); // kind
        bytes.write_i32::<BigEndian>(FIFFT_INT).unwrap(); // type
        bytes.write_i32::<BigEndian>(4).unwrap(); // size
        bytes.write_u32::<BigEndian>(1024).unwrap(); // pos

        let mut cursor = Cursor::new(bytes);
        let entry = DirEntry::read(&mut cursor).unwrap();

        assert_eq!(entry.kind, FIFF_NCHAN);
        assert_eq!(entry.type_, FIFFT_INT);
        assert_eq!(entry.size, 4);
        assert_eq!(entry.pos, 1024);
    }

    #[test]
    fn test_tag_invalid_as_i32() {
        let data = vec![0, 1]; // Only 2 bytes

        let bytes = create_test_tag_bytes(100, FIFFT_INT, 2, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        assert!(tag.as_i32().is_err());
    }

    #[test]
    fn test_tag_invalid_as_f64() {
        let data = vec![0, 1, 2, 3]; // Only 4 bytes

        let bytes = create_test_tag_bytes(100, FIFFT_DOUBLE, 4, &data);
        let mut cursor = Cursor::new(bytes);

        let tag = Tag::read(&mut cursor).unwrap();
        assert!(tag.as_f64().is_err());
    }
}
