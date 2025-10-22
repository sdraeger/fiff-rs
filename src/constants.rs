/// FIFF Constants
/// Based on MNE-Python's _fiff/constants.py

// File structure tags
pub const FIFF_FILE_ID: i32 = 100;
pub const FIFF_DIR_POINTER: i32 = 101;
pub const FIFF_BLOCK_START: i32 = 104;
pub const FIFF_BLOCK_END: i32 = 105;

// Block types (used in BLOCK_START tag data)
pub const FIFFB_MEAS: i32 = 100;
pub const FIFFB_MEAS_INFO: i32 = 101;
pub const FIFFB_RAW_DATA: i32 = 102;
pub const FIFFB_CONTINUOUS_DATA: i32 = 112; // Note: 112, not 103
pub const FIFFB_IAS_RAW_DATA: i32 = 113;

// Tag types - Data types
pub const FIFFT_VOID: i32 = 0;
pub const FIFFT_BYTE: i32 = 1;
pub const FIFFT_SHORT: i32 = 2;
pub const FIFFT_INT: i32 = 3;
pub const FIFFT_FLOAT: i32 = 4;
pub const FIFFT_DOUBLE: i32 = 5;
pub const FIFFT_STRING: i32 = 10;
pub const FIFFT_DAU_PACK16: i32 = 16;
pub const FIFFT_COMPLEX_FLOAT: i32 = 20;
pub const FIFFT_COMPLEX_DOUBLE: i32 = 21;
pub const FIFFT_CH_INFO_STRUCT: i32 = 30; // Channel info structure (96 bytes)
pub const FIFFT_ID_STRUCT: i32 = 31; // File ID structure

// Tag kinds - Measurement info
pub const FIFF_NCHAN: i32 = 200; // Number of channels
pub const FIFF_SFREQ: i32 = 201; // Sampling frequency
pub const FIFF_LOWPASS: i32 = 202; // Lowpass filter frequency
pub const FIFF_CH_INFO: i32 = 203; // Channel information
pub const FIFF_MEAS_DATE: i32 = 204; // Measurement date
pub const FIFF_LINE_FREQ: i32 = 205; // Line frequency (50/60 Hz)
pub const FIFF_HIGHPASS: i32 = 206; // Highpass filter frequency
pub const FIFF_FIRST_SAMPLE: i32 = 208; // First sample index
pub const FIFF_EXPERIMENTER: i32 = 211; // Experimenter name
pub const FIFF_DESCRIPTION: i32 = 212; // Measurement description
pub const FIFF_PROJ_ID: i32 = 213; // Project ID
pub const FIFF_PROJ_NAME: i32 = 214; // Project name
pub const FIFF_DATA_BUFFER: i32 = 300; // Data buffer
pub const FIFF_DATA_SKIP: i32 = 301; // Data skip

// MNE-specific tags
pub const FIFF_MNE_BAD_CHS: i32 = 3502; // Bad channel names (colon-separated string)

// SSP/Projection tags
pub const FIFFB_PROJ: i32 = 3411; // Projection block
pub const FIFF_PROJ_ITEM_KIND: i32 = 3412; // Projection kind (MEG/EEG)
pub const FIFF_PROJ_ITEM_TIME: i32 = 3413; // Time range
pub const FIFF_PROJ_ITEM_NVEC: i32 = 3414; // Number of vectors
pub const FIFF_PROJ_ITEM_VECTORS: i32 = 3415; // Projection vectors
pub const FIFF_PROJ_ITEM_DESC: i32 = 3416; // Description
pub const FIFF_PROJ_ITEM_ACTIVE: i32 = 3417; // Active flag
pub const FIFF_NAMED_MATRIX: i32 = 3420; // Named matrix structure

// Projection kinds
pub const FIFFV_PROJ_ITEM_NONE: i32 = 0;
pub const FIFFV_MNE_PROJ_ITEM_EEG_AVREF: i32 = 10; // EEG average reference

// CTF Compensation tags
pub const FIFFB_MNE_CTF_COMP: i32 = 3501; // CTF compensation block
pub const FIFF_MNE_CTF_COMP_KIND: i32 = 3503; // Compensation kind
pub const FIFF_MNE_CTF_COMP_DATA: i32 = 3504; // Compensation data (named matrix)
pub const FIFF_MNE_CTF_COMP_CALIBRATED: i32 = 3505; // Whether data is already compensated
pub const FIFF_MNE_DERIVATION_DATA: i32 = 3506; // Derivation data

// CTF compensation grades
pub const FIFFV_MNE_CTFV_COMP_NONE: i32 = 0; // No compensation
pub const FIFFV_MNE_CTFV_COMP_G1BR: i32 = 1; // Gradient compensation grade 1
pub const FIFFV_MNE_CTFV_COMP_G2BR: i32 = 2; // Gradient compensation grade 2
pub const FIFFV_MNE_CTFV_COMP_G3BR: i32 = 3; // Gradient compensation grade 3

// Coordinate transformation tags
pub const FIFF_COORD_TRANS: i32 = 222; // Coordinate transformation
pub const FIFF_MNE_COORD_FRAME: i32 = 3507; // Coordinate frame

// Coordinate frames (FIFFV_COORD_*)
pub const FIFFV_COORD_UNKNOWN: i32 = 0; // Unknown coordinate frame
pub const FIFFV_COORD_DEVICE: i32 = 1; // Device coordinates (MEG sensor array)
pub const FIFFV_COORD_ISOTRAK: i32 = 2; // Isotrak digitizer coordinates
pub const FIFFV_COORD_HPI: i32 = 3; // HPI coil coordinates
pub const FIFFV_COORD_HEAD: i32 = 4; // Head coordinates (fiducial-based)
pub const FIFFV_COORD_MRI: i32 = 5; // MRI coordinates
pub const FIFFV_COORD_MRI_SLICE: i32 = 6; // MRI slice coordinates
pub const FIFFV_COORD_MRI_DISPLAY: i32 = 7; // MRI display coordinates
pub const FIFFV_COORD_DICOM_DEVICE: i32 = 8; // DICOM device coordinates
pub const FIFFV_COORD_IMAGING_DEVICE: i32 = 9; // Generic imaging device coordinates

// Channel info
pub const FIFF_CH_SCAN_NO: i32 = 400;
pub const FIFF_CH_LOGICAL_NO: i32 = 401;
pub const FIFF_CH_KIND: i32 = 402;
pub const FIFF_CH_RANGE: i32 = 403;
pub const FIFF_CH_CAL: i32 = 404;
pub const FIFF_CH_POS: i32 = 405;
pub const FIFF_CH_UNIT: i32 = 406;
pub const FIFF_CH_UNIT_MUL: i32 = 407;
pub const FIFF_CH_DACQ_NAME: i32 = 408;

// Channel types (FIFFV_*_CH)
pub const FIFFV_MEG_CH: i32 = 1; // MEG channel (magnetometer or gradiometer)
pub const FIFFV_REF_MEG_CH: i32 = 301; // MEG reference channel
pub const FIFFV_EEG_CH: i32 = 2; // EEG channel
pub const FIFFV_MCG_CH: i32 = 201; // MCG channel
pub const FIFFV_STIM_CH: i32 = 3; // Stimulus channel
pub const FIFFV_EOG_CH: i32 = 202; // EOG channel
pub const FIFFV_EMG_CH: i32 = 302; // EMG channel
pub const FIFFV_ECG_CH: i32 = 402; // ECG channel
pub const FIFFV_MISC_CH: i32 = 502; // Miscellaneous channel
pub const FIFFV_RESP_CH: i32 = 602; // Respiration channel

/// Get size in bytes for a given FIFF data type
pub fn type_size(fiff_type: i32) -> Option<usize> {
    match fiff_type {
        FIFFT_BYTE => Some(1),
        FIFFT_SHORT | FIFFT_DAU_PACK16 => Some(2),
        FIFFT_INT | FIFFT_FLOAT => Some(4),
        FIFFT_DOUBLE => Some(8),
        FIFFT_COMPLEX_FLOAT => Some(8),
        FIFFT_COMPLEX_DOUBLE => Some(16),
        _ => None,
    }
}

/// Get Rust type name for FIFF data type
pub fn type_name(fiff_type: i32) -> &'static str {
    match fiff_type {
        FIFFT_SHORT | FIFFT_DAU_PACK16 => "short",
        FIFFT_INT => "int",
        FIFFT_FLOAT | FIFFT_COMPLEX_FLOAT => "single",
        FIFFT_DOUBLE | FIFFT_COMPLEX_DOUBLE => "double",
        _ => "unknown",
    }
}

/// Check if a channel type is a data channel (not stimulus, etc.)
pub fn is_data_channel(kind: i32) -> bool {
    matches!(
        kind,
        FIFFV_MEG_CH
            | FIFFV_REF_MEG_CH
            | FIFFV_EEG_CH
            | FIFFV_MCG_CH
            | FIFFV_EOG_CH
            | FIFFV_EMG_CH
            | FIFFV_ECG_CH
            | FIFFV_MISC_CH
            | FIFFV_RESP_CH
    )
}

/// Get human-readable channel type name
pub fn channel_type_name(kind: i32) -> &'static str {
    match kind {
        FIFFV_MEG_CH => "MEG",
        FIFFV_REF_MEG_CH => "REF_MEG",
        FIFFV_EEG_CH => "EEG",
        FIFFV_MCG_CH => "MCG",
        FIFFV_STIM_CH => "STIM",
        FIFFV_EOG_CH => "EOG",
        FIFFV_EMG_CH => "EMG",
        FIFFV_ECG_CH => "ECG",
        FIFFV_MISC_CH => "MISC",
        FIFFV_RESP_CH => "RESP",
        _ => "UNKNOWN",
    }
}

/// Get human-readable coordinate frame name
pub fn coord_frame_name(frame: i32) -> &'static str {
    match frame {
        FIFFV_COORD_UNKNOWN => "Unknown",
        FIFFV_COORD_DEVICE => "Device",
        FIFFV_COORD_ISOTRAK => "Isotrak",
        FIFFV_COORD_HPI => "HPI",
        FIFFV_COORD_HEAD => "Head",
        FIFFV_COORD_MRI => "MRI",
        FIFFV_COORD_MRI_SLICE => "MRI Slice",
        FIFFV_COORD_MRI_DISPLAY => "MRI Display",
        FIFFV_COORD_DICOM_DEVICE => "DICOM Device",
        FIFFV_COORD_IMAGING_DEVICE => "Imaging Device",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_size() {
        assert_eq!(type_size(FIFFT_BYTE), Some(1));
        assert_eq!(type_size(FIFFT_SHORT), Some(2));
        assert_eq!(type_size(FIFFT_DAU_PACK16), Some(2));
        assert_eq!(type_size(FIFFT_INT), Some(4));
        assert_eq!(type_size(FIFFT_FLOAT), Some(4));
        assert_eq!(type_size(FIFFT_DOUBLE), Some(8));
        assert_eq!(type_size(FIFFT_COMPLEX_FLOAT), Some(8));
        assert_eq!(type_size(FIFFT_COMPLEX_DOUBLE), Some(16));
        assert_eq!(type_size(FIFFT_VOID), None);
        assert_eq!(type_size(FIFFT_STRING), None);
        assert_eq!(type_size(999), None); // Unknown type
    }

    #[test]
    fn test_type_name() {
        assert_eq!(type_name(FIFFT_SHORT), "short");
        assert_eq!(type_name(FIFFT_DAU_PACK16), "short");
        assert_eq!(type_name(FIFFT_INT), "int");
        assert_eq!(type_name(FIFFT_FLOAT), "single");
        assert_eq!(type_name(FIFFT_COMPLEX_FLOAT), "single");
        assert_eq!(type_name(FIFFT_DOUBLE), "double");
        assert_eq!(type_name(FIFFT_COMPLEX_DOUBLE), "double");
        assert_eq!(type_name(FIFFT_BYTE), "unknown");
        assert_eq!(type_name(999), "unknown");
    }

    #[test]
    fn test_is_data_channel() {
        // Data channels
        assert!(is_data_channel(FIFFV_MEG_CH));
        assert!(is_data_channel(FIFFV_REF_MEG_CH));
        assert!(is_data_channel(FIFFV_EEG_CH));
        assert!(is_data_channel(FIFFV_MCG_CH));
        assert!(is_data_channel(FIFFV_EOG_CH));
        assert!(is_data_channel(FIFFV_EMG_CH));
        assert!(is_data_channel(FIFFV_ECG_CH));
        assert!(is_data_channel(FIFFV_MISC_CH));
        assert!(is_data_channel(FIFFV_RESP_CH));

        // Non-data channels
        assert!(!is_data_channel(FIFFV_STIM_CH));
        assert!(!is_data_channel(999)); // Unknown type
    }

    #[test]
    fn test_channel_type_name() {
        assert_eq!(channel_type_name(FIFFV_MEG_CH), "MEG");
        assert_eq!(channel_type_name(FIFFV_REF_MEG_CH), "REF_MEG");
        assert_eq!(channel_type_name(FIFFV_EEG_CH), "EEG");
        assert_eq!(channel_type_name(FIFFV_MCG_CH), "MCG");
        assert_eq!(channel_type_name(FIFFV_STIM_CH), "STIM");
        assert_eq!(channel_type_name(FIFFV_EOG_CH), "EOG");
        assert_eq!(channel_type_name(FIFFV_EMG_CH), "EMG");
        assert_eq!(channel_type_name(FIFFV_ECG_CH), "ECG");
        assert_eq!(channel_type_name(FIFFV_MISC_CH), "MISC");
        assert_eq!(channel_type_name(FIFFV_RESP_CH), "RESP");
        assert_eq!(channel_type_name(999), "UNKNOWN");
    }

    #[test]
    fn test_coord_frame_name() {
        assert_eq!(coord_frame_name(FIFFV_COORD_UNKNOWN), "Unknown");
        assert_eq!(coord_frame_name(FIFFV_COORD_DEVICE), "Device");
        assert_eq!(coord_frame_name(FIFFV_COORD_ISOTRAK), "Isotrak");
        assert_eq!(coord_frame_name(FIFFV_COORD_HPI), "HPI");
        assert_eq!(coord_frame_name(FIFFV_COORD_HEAD), "Head");
        assert_eq!(coord_frame_name(FIFFV_COORD_MRI), "MRI");
        assert_eq!(coord_frame_name(FIFFV_COORD_MRI_SLICE), "MRI Slice");
        assert_eq!(coord_frame_name(FIFFV_COORD_MRI_DISPLAY), "MRI Display");
        assert_eq!(coord_frame_name(FIFFV_COORD_DICOM_DEVICE), "DICOM Device");
        assert_eq!(
            coord_frame_name(FIFFV_COORD_IMAGING_DEVICE),
            "Imaging Device"
        );
        assert_eq!(coord_frame_name(999), "Unknown");
    }

    #[test]
    fn test_constant_values() {
        // Verify critical constants match MNE-Python values
        assert_eq!(FIFF_FILE_ID, 100);
        assert_eq!(FIFF_DIR_POINTER, 101);
        assert_eq!(FIFF_BLOCK_START, 104);
        assert_eq!(FIFF_BLOCK_END, 105);

        assert_eq!(FIFFB_MEAS_INFO, 101);
        assert_eq!(FIFFB_RAW_DATA, 102);
        assert_eq!(FIFFB_CONTINUOUS_DATA, 112);

        assert_eq!(FIFF_NCHAN, 200);
        assert_eq!(FIFF_SFREQ, 201);
        assert_eq!(FIFF_CH_INFO, 203);
        assert_eq!(FIFF_FIRST_SAMPLE, 208);
        assert_eq!(FIFF_DATA_BUFFER, 300);

        assert_eq!(FIFFV_MEG_CH, 1);
        assert_eq!(FIFFV_EEG_CH, 2);
        assert_eq!(FIFFV_STIM_CH, 3);

        // Coordinate transformation constants
        assert_eq!(FIFF_COORD_TRANS, 222);
        assert_eq!(FIFFV_COORD_DEVICE, 1);
        assert_eq!(FIFFV_COORD_HEAD, 4);
        assert_eq!(FIFFV_COORD_MRI, 5);
    }
}
