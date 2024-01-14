use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use clap::{Parser, ValueEnum};
use cobra::Cobra;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SupportedStreamConfig;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokenizers::Tokenizer;

use crate::decoder::{token_id, Decoder, Model, Task};
mod decoder;

static LISTENING: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum WhichModel {
    Tiny,
    #[value(name = "tiny.en")]
    TinyEn,
    Base,
    #[value(name = "base.en")]
    BaseEn,
    Small,
    #[value(name = "small.en")]
    SmallEn,
    Medium,
    #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    #[value(name = "distil-small.en")]
    DistilSmallEn,
    #[value(name = "distil-medium.en")]
    DistilMediumEn,
    #[value(name = "distil-large-v2")]
    DistilLargeV2,
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::DistilLargeV2 => true,
            Self::TinyEn
            | Self::BaseEn
            | Self::SmallEn
            | Self::MediumEn
            | Self::DistilMediumEn
            | Self::DistilSmallEn => false,
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::DistilSmallEn => ("distil-whisper/distil-small.en", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author = "Leonard Souza <hello@leonardsouza.com>")]
#[command(version = "1.0")]
#[command(about = "Convert mic input to a different language.")]
#[command(
    long_about = "Convert mic input to a different language via whisper net and [INSERT MODEL]."
)]
struct Args {
    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "tiny.en")]
    model: WhichModel,

    #[arg(long)]
    quantized: bool,

    /// Picovoice access key for Cobra Voice Audio Detection
    #[arg(short, long)]
    access_key: String,

    /// Show available devices
    #[arg(short, long, default_value_t = false)]
    show_devices: bool,

    /// Output path
    #[arg(short, long)]
    output_path: Option<String>,

    /// Language.
    #[arg(long)]
    language: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    revision: Option<String>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long)]
    task: Option<Task>,

    /// Timestamps mode, this is not fully implemented yet.
    #[arg(long)]
    timestamps: bool,

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,
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

fn run(
    access_key: String,
    output_path: Option<String>,
    decoder: Decoder,
    mel_config: Config,
    device: &Device,
) -> Result<()> {
    let mel_bytes = match mel_config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

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
    let device_clone = device.clone();

    let stream = input_device.build_input_stream(
        &config.config(),
        move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            // Process the audio data here
            // println!("Received {} samples", data.len());
            // println!("Timestamp {:?}", info.timestamp());
            let mut i16_samples: Vec<i16> = Vec::new();
            convert_f32_to_i16(data, &mut i16_samples);

            // let voice_probability = cobra.process(&i16_samples).unwrap();
            // print_voice_activity(voice_probability);

            let mel = audio::pcm_to_mel(&mel_config, &data, &mel_filters);
            let mel_len = mel.len();
            let mel = Tensor::from_vec(
                mel,
                (
                    1,
                    mel_config.num_mel_bins,
                    mel_len / mel_config.num_mel_bins,
                ),
                &device_clone,
            );
            print!("loaded mel: {:?}", mel.unwrap().dims());
            io::stdout().flush().expect("Unable to write to stdout");

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

fn create_decoder(args: Args, device: &Device) -> Result<(Decoder, Config)> {
    let (default_model, default_revision) = if args.quantized {
        ("lmz/candle-whisper", "main")
    } else {
        args.model.model_and_revision()
    };
    let default_model = default_model.to_string();
    let default_revision = default_revision.to_string();
    let (model_id, revision) = match (args.model_id, args.revision) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        // let dataset = api.dataset("Narsil/candle-examples".to_string());
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let (config, tokenizer, model) = if args.quantized {
            let ext = match args.model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => unimplemented!("no quantized support for {:?}", args.model),
            };
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let model = repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };
        (config, tokenizer, model)
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let c = config.clone();

    let model = if args.quantized {
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&weights_filename)?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config)?)
    } else {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        Model::Normal(m::model::Whisper::load(&vb, config)?)
    };

    let language_token = match (args.model.is_multilingual(), args.language) {
        (true, None) => None, //Some(multilingual::detect_language(&mut model, &tokenizer, &mel)?),
        (false, None) => None,
        (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
            Ok(token_id) => Some(token_id),
            Err(_) => anyhow::bail!("language {language} is not supported"),
        },
        (false, Some(_)) => {
            anyhow::bail!("a language cannot be set for non-multilingual models")
        }
    };

    Ok((
        Decoder::new(
            model,
            tokenizer,
            args.seed,
            &device,
            language_token,
            args.task,
            args.timestamps,
            args.verbose,
        )?,
        c,
    ))
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    if args.show_devices == true {
        let host = cpal::default_host();
        let devices = host.devices()?;
        println!("Available devices:");
        for device in devices {
            println!("  {}", device.name()?);
        }
    }

    let device = candle_examples::device(if args.cpu { false } else { true })?;
    let output_path = args.output_path.clone();
    let access_key = args.access_key.clone();
    let result = create_decoder(args, &device);
    let decoder = result.unwrap();

    run(access_key, output_path, decoder.0, decoder.1, &device)
}
