use anyhow::Result;
use clap::Parser;
use cobra::Cobra;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SupportedStreamConfig;
use std::io;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

static LISTENING: AtomicBool = AtomicBool::new(false);

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author = "Leonard Souza <hello@leonardsouza.com>")]
#[command(version = "1.0")]
#[command(about = "Convert mic input to a different language.")]
#[command(
    long_about = "Convert mic input to a different language via whisper net and [INSERT MODEL]."
)]
struct Args {
    /// Picovoice access key for Cobra Voice Audio Detection
    #[arg(short, long)]
    access_key: String,

    /// Show available devices
    #[arg(short, long, default_value_t = false)]
    show_devices: bool,

    /// Output path
    #[arg(short, long)]
    output_path: Option<String>,
}

fn convert_f32_to_i16(input: &[f32], output: &mut Vec<i16>) {
    output.clear(); // Clear existing contents
    output.extend(input.iter().map(|&sample| {
        // Scale and clamp as before
        (sample * 32767.0).clamp(-32768.0, 32767.0) as i16
    }));
}

fn print_voice_activity(voice_probability: f32) {
    let voice_percentage = voice_probability * 100.0;
    let bar_length = ((voice_percentage / 10.0) * 3.0).ceil() as usize;
    let empty_length = 30 - bar_length;
    print!(
        "\r[{:3.0}]|{}{}|",
        voice_percentage,
        "█".repeat(bar_length),
        " ".repeat(empty_length)
    );
    io::stdout().flush().expect("Unable to write to stdout");
}

fn run(access_key: String, output_path: Option<String>) -> Result<()> {
    let cobra = Cobra::new(access_key)
        .expect("Failed to create Cobra! Please obtain an access key from https://picovoice.ai/");

    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("No input device available");

    let config: SupportedStreamConfig = input_device.default_input_config()?;
    let sample_rate = config.sample_rate();
    // let channels = config.channels();

    println!("Sample rate: {:?}", config.sample_rate());

    let audio_data = Arc::new(Mutex::new(Vec::new()));
    let audio_data_clone = Arc::clone(&audio_data);
    let cloned_output_path = output_path.clone(); // Clone the output path

    let stream = input_device.build_input_stream(
        &config.config(),
        move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            // Process the audio data here
            // println!("Received {} samples", data.len());
            // println!("Timestamp {:?}", info.timestamp());
            let mut i16_samples: Vec<i16> = Vec::new();
            convert_f32_to_i16(data, &mut i16_samples);

            let voice_probability = cobra.process(&i16_samples).unwrap();
            print_voice_activity(voice_probability);

            if cloned_output_path.is_some() {
                let mut audio_data_locked = audio_data_clone.lock().unwrap();
                audio_data_locked.extend_from_slice(&i16_samples);
            }
        },
        |err| {
            // Handle stream error here
            eprintln!("An error occurred: {}", err);
        },
        Some(Duration::new(1, 0)),
    )?;

    LISTENING.store(true, Ordering::SeqCst);
    stream.play()?;

    ctrlc::set_handler(|| {
        LISTENING.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    while LISTENING.load(Ordering::SeqCst) {}

    if let Some(output_path) = output_path {
        println!("Saving recording to {}", output_path);
        let wavspec = hound::WavSpec {
            channels: 1,
            sample_rate: sample_rate.0,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(output_path, wavspec)
            .expect("Failed to open output audio file");
        let audio_data_locked = audio_data.lock().unwrap();
        for sample in audio_data_locked.iter() {
            writer.write_sample(*sample)?;
        }
    }
    println!("All done!");

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.show_devices == true {
        let host = cpal::default_host();
        let devices = host.devices()?;
        println!("Available devices:");
        for device in devices {
            println!("  {}", device.name()?);
        }
    }

    run(args.access_key, args.output_path)
}
