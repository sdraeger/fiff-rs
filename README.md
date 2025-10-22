# FIFF - Functional Imaging File Format Parser

[![Crates.io](https://img.shields.io/crates/v/fiff.svg)](https://crates.io/crates/fiff)
[![Documentation](https://docs.rs/fiff/badge.svg)](https://docs.rs/fiff)
[![License](https://img.shields.io/crates/l/fiff.svg)](https://github.com/yourusername/ddalab)

⚠️ **MINIMAL IMPLEMENTATION** - See [limitations](#limitations) below.

Pure Rust implementation of the FIFF (Functional Imaging File Format) parser for MEG/EEG data.

## Features

- ✅ **Pure Rust** - No C dependencies, fully memory-safe
- ✅ **Zero-copy parsing** where possible for performance
- ✅ **Channel calibration** - Automatic conversion to physical units
- ✅ **Sequential reading** - Handles files with and without directory pointers
- ✅ **Complete metadata** - Channel names, types, positions, sampling rates
- ✅ **Bad channel detection** - Automatically filter out defective sensors
- ✅ **Filter information** - Lowpass/highpass filter settings preserved
- ✅ **Experiment metadata** - Recording date, experimenter, project info
- ✅ **SSP Projectors** - Signal space projection metadata for artifact removal
- ✅ **CTF Compensation** - CTF MEG compensation grades and matrices
- ✅ **Coordinate Transformations** - Device-to-head, head-to-MRI transforms
- ✅ **Well-documented** - Based on MNE-Python's battle-tested implementation

## What is FIFF?

FIFF is the native file format for Neuromag/Elekta MEG systems and is widely used in magnetoencephalography (MEG) and electroencephalography (EEG) research. It stores:
- Raw sensor data (MEG, EEG, EOG, ECG, etc.)
- Channel metadata (names, types, calibration coefficients, sensor positions)
- Measurement metadata (sampling rate, recording time, coordinate frames)
- Processing history and annotations

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
fiff = "0.1"
```

### Reading a FIFF file

```rust
use fiff::{open_fiff, MeasInfo};
use std::path::Path;

fn main() -> std::io::Result<()> {
    // Open the FIFF file
    let path = Path::new("data.fif");
    let (mut reader, tree) = open_fiff(path)?;

    // Extract measurement metadata
    let meas_info = MeasInfo::read(&mut reader, &tree)?;

    // Print channel information
    println!("Channels: {}", meas_info.nchan);
    println!("Sampling rate: {} Hz", meas_info.sfreq);

    for ch in &meas_info.channels {
        println!("{}: {} (cal={}, range={})",
            ch.ch_name, ch.type_name(), ch.cal, ch.range);
    }

    Ok(())
}
```

### Filtering channels by type

```rust
use fiff::{FIFFV_MEG_CH, FIFFV_STIM_CH};

// Get only MEG channels
let meg_channels: Vec<_> = meas_info.channels
    .iter()
    .filter(|ch| ch.kind == FIFFV_MEG_CH)
    .collect();

// Exclude stimulus channels
let data_channels: Vec<_> = meas_info.channels
    .iter()
    .filter(|ch| ch.is_data_channel())
    .collect();
```

### Working with bad channels

```rust
// Get indices of good (non-bad) channels
let good_indices = meas_info.get_good_channels();

// Check if a specific channel is bad
if meas_info.is_bad_channel("MEG 0113") {
    println!("Channel MEG 0113 is marked as bad");
}

// Get good channels for analysis
let good_channels: Vec<_> = meas_info.get_good_channels()
    .iter()
    .map(|&idx| &meas_info.channels[idx])
    .collect();
```

### Accessing metadata

```rust
// Filter information
if let Some(lowpass) = meas_info.lowpass {
    println!("Lowpass filter: {} Hz", lowpass);
}

// Measurement date (Unix timestamp)
if let Some(date) = meas_info.meas_date {
    println!("Recorded at: {}", date);
}

// Experiment information
if let Some(exp) = &meas_info.experimenter {
    println!("Experimenter: {}", exp);
}
```

### Working with SSP projections and CTF compensation

```rust
// Check for SSP projections (artifact removal)
println!("SSP Projections: {}", meas_info.projs.len());
for proj in &meas_info.projs {
    println!("- {}: {} vectors ({})",
        proj.desc,
        proj.nvec,
        if proj.active { "active" } else { "inactive" }
    );
}

// Check CTF compensation (for CTF MEG systems)
println!("CTF Compensation: {}", meas_info.comps.len());
for comp in &meas_info.comps {
    println!("- Grade {}: {} ({})",
        comp.grade_name(),
        comp.kind,
        if comp.calibrated { "calibrated" } else { "uncalibrated" }
    );
}
```

### Working with coordinate transformations

```rust
// Get device-to-head transformation
if let Some(trans) = meas_info.get_device_to_head_trans() {
    println!("Device-to-head transform: {}", trans.description());
    println!("  Rotation matrix: {:?}", trans.rot);
    println!("  Translation: {:?}", trans.move_);
}

// Get head-to-MRI transformation
if let Some(trans) = meas_info.get_head_to_mri_trans() {
    println!("Head-to-MRI transform: {}", trans.description());
}

// Check all available transformations
println!("Coordinate transforms: {}", meas_info.coord_trans.len());
for trans in &meas_info.coord_trans {
    println!("- {}", trans.description());
}
```

## Architecture

The crate is organized into focused modules:

- **`constants`** - FIFF format constants (block types, tag kinds, channel types)
- **`tag`** - Binary tag reading and parsing
- **`tree`** - Directory tree structure navigation
- **Main module** - High-level API (`open_fiff`, `MeasInfo`, `ChannelInfo`)

## Supported Features

- [x] File opening and validation
- [x] Tag-based file structure parsing
- [x] Sequential tag reading (no directory required)
- [x] Directory-based random access
- [x] Measurement info extraction
- [x] Channel info parsing (96-byte structures)
- [x] Channel calibration coefficients
- [x] Channel type detection
- [x] Bad channel detection and filtering
- [x] Filter information (lowpass/highpass)
- [x] Measurement date and timing
- [x] Experiment metadata (experimenter, project, description)
- [x] Raw data buffer reading
- [x] Coordinate transformations (device-to-head, head-to-MRI)
- [x] SSP projectors (metadata parsing)
- [x] Compensation matrices (metadata parsing)
- [ ] Event parsing
- [ ] Annotations

## Compatibility

Based on the [MNE-Python](https://mne.tools/) FIFF implementation, this crate is compatible with:
- Neuromag/Elekta MEG systems
- Vectorview systems
- Modern MEGIN systems
- Files created by MNE-Python
- Files from OpenNeuro NEMAR datasets

## Performance

The parser is designed for efficiency:
- Big-endian binary parsing with `byteorder`
- Minimal allocations during parsing
- Lazy loading of data buffers
- Streaming-friendly API

## Testing

The crate includes comprehensive tests based on MNE-Python's test suite:

```bash
cargo test
```

Note: Integration tests require actual FIFF files (not included).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Limitations

⚠️ **This is a minimal implementation suitable for basic raw data reading.**

### What This Crate DOES
- ✅ Read raw MEG/EEG sensor data from Neuromag/Elekta systems
- ✅ Parse channel metadata (names, types, calibration)
- ✅ Apply calibration coefficients (cal × range)
- ✅ Extract sampling rate and basic info
- ✅ Filter channels by type

### What This Crate DOES NOT Do
- ⚠️ **Matrix application** - SSP/CTF matrices stored but not applied (no ndarray dependency)
- ❌ **Digitization data** - No HPI/fiducial/head shape points
- ❌ **Events/annotations** - No stimulus timing
- ❌ **File writing** - Read-only

### Comparison to MNE-Python
We implement **~25%** of MNE-Python's `_fiff` module:
- **5 of 17 files** implemented (constants, tag, tree, meas_info, coord_trans)
- **86 of 350+ constants** defined
- **15 of 25+ metadata tags** parsed

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed comparison.

### When to Use This Crate
✅ **Good for:**
- Quick data extraction from Neuromag FIF files
- Integrating MEG data into Rust pipelines
- Educational/learning purposes
- Prototyping

❌ **Not suitable for:**
- CTF MEG data (without implementing compensation)
- Advanced MEG analysis requiring full metadata
- Source localization
- Publication-quality analysis (use MNE-Python)

### Recommendation
For scientific MEG/EEG analysis, use [MNE-Python](https://mne.tools/). This crate is for Rust integration and basic data access.

## Acknowledgments

This implementation is based on [MNE-Python](https://mne.tools/)'s FIFF parser. We thank the MNE community for their excellent work and documentation of the FIFF format.

## See Also

- [MNE-Python](https://mne.tools/) - The Python library for MEG/EEG analysis
- [edfio](https://crates.io/crates/edfio) - EDF/EDF+ file parser in Rust
- [neurokit](https://crates.io/crates/neurokit) - Neuroscience data processing toolkit
