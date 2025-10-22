/// Simple example of reading a FIFF file and extracting metadata
///
/// Usage: cargo run --example read_fiff -- path/to/file.fif
use fiff::{open_fiff, MeasInfo, FIFFV_EEG_CH, FIFFV_MEG_CH, FIFFV_STIM_CH};
use std::env;
use std::path::Path;

fn main() -> std::io::Result<()> {
    // Get file path from command line
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <fiff_file>", args[0]);
        eprintln!("Example: {} data/sample_audvis_raw.fif", args[0]);
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);

    println!("Opening FIFF file: {}", path.display());
    println!("{}", "=".repeat(60));

    // Open the FIFF file
    let (mut reader, tree) = open_fiff(path)?;

    // Extract measurement metadata
    let meas_info = MeasInfo::read(&mut reader, &tree)?;

    // Print basic info
    println!("Basic Information:");
    println!("  Number of channels: {}", meas_info.nchan);
    println!("  Sampling frequency: {:.2} Hz", meas_info.sfreq);
    println!();

    // Count channels by type
    let mut meg_count = 0;
    let mut eeg_count = 0;
    let mut stim_count = 0;
    let mut other_count = 0;

    for ch in &meas_info.channels {
        match ch.kind {
            FIFFV_MEG_CH => meg_count += 1,
            FIFFV_EEG_CH => eeg_count += 1,
            FIFFV_STIM_CH => stim_count += 1,
            _ => other_count += 1,
        }
    }

    println!("Channel Types:");
    if meg_count > 0 {
        println!("  MEG channels: {}", meg_count);
    }
    if eeg_count > 0 {
        println!("  EEG channels: {}", eeg_count);
    }
    if stim_count > 0 {
        println!("  Stimulus channels: {}", stim_count);
    }
    if other_count > 0 {
        println!("  Other channels: {}", other_count);
    }
    println!();

    // Show first 10 channels
    println!("First 10 Channels:");
    for (i, ch) in meas_info.channels.iter().take(10).enumerate() {
        println!(
            "  {}: {} ({}) - cal={:.2e}, range={:.2e}",
            i,
            ch.ch_name,
            ch.type_name(),
            ch.cal,
            ch.range
        );
    }

    if meas_info.channels.len() > 10 {
        println!("  ... and {} more channels", meas_info.channels.len() - 10);
    }

    Ok(())
}
