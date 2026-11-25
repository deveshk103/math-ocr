mod daemon;
mod models;
mod overlay;
mod pipeline;
mod platform;
mod preprocess;
mod stitch;
mod types;
mod worker_mfd;
mod worker_mfr;
mod worker_text;

use crate::models::{inspect_onnx_models, ModelPaths};
use crate::pipeline::run_benchmark;
use crate::preprocess::PreprocessConfig;
use crate::types::SnipMode;
use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMode {
    Mixed,
    Math,
    Text,
}

impl From<CliMode> for SnipMode {
    fn from(m: CliMode) -> Self {
        match m {
            CliMode::Mixed => SnipMode::Mixed,
            CliMode::Math => SnipMode::Math,
            CliMode::Text => SnipMode::Text,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "snip_ocr",
    version,
    about = "Low-RAM (<= 400 MB peak), offline Windows 11 snip-to-Markdown/LaTeX OCR tool"
)]
struct Cli {
    /// Internal worker stage for subprocess memory isolation (`mfd`, `text`, `mfr`)
    #[arg(long, hide = true)]
    worker: Option<String>,

    /// Override model directory (defaults to %LOCALAPPDATA%\\math-snip-ocr\\models)
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,

    /// Black point for watermark removal LUT (0..254)
    #[arg(long, global = true, default_value_t = 45)]
    black_pt: u8,

    /// White point for watermark removal LUT (1..255)
    #[arg(long, global = true, default_value_t = 175)]
    white_pt: u8,

    /// Trailing positional arguments used by internal `--worker` mode
    #[arg(hide = true)]
    worker_args: Vec<PathBuf>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run the full pipeline headlessly on a local PNG with latency & PeakWorkingSetSize reporting
    Bench {
        /// Path to input PNG/JPG screenshot
        image_path: PathBuf,
        /// Write ./debug/01_cleaned.png, 02_boxes.png, and 03_masked_for_text.png
        #[arg(long)]
        debug: bool,
        /// OCR mode to benchmark
        #[arg(long, value_enum, default_value_t = CliMode::Mixed)]
        mode: CliMode,
        /// Fail benchmark if any stage exceeds this PeakWorkingSetSize (MB)
        #[arg(long, default_value_t = 400.0)]
        max_ram_mb: f64,
    },
    /// Download all ONNX weights once into the local models directory for 100% offline use
    Download,
    /// Inspect and print exact ONNX input/output tensor names and shapes
    InspectModels,
}

fn main() -> Result<()> {
    platform::enable_per_monitor_dpi_awareness();
    let cli = Cli::parse();
    let models = ModelPaths::resolve(cli.model_dir.as_deref())?;

    // Fast path for isolated child worker subprocesses
    if let Some(ref stage) = cli.worker {
        return match stage.as_str() {
            "mfd" => {
                let img = cli
                    .worker_args
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("--worker mfd requires <image_path>"))?;
                worker_mfd::run_mfd_worker(img, &models.mfd_onnx)
            }
            "text" => {
                let img = cli
                    .worker_args
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("--worker text requires <image_path>"))?;
                worker_text::run_text_worker(img)
            }
            "mfr" => worker_mfr::run_mfr_worker(&cli.worker_args, &models),
            other => bail!("Unknown worker stage: {other}"),
        };
    }

    let preproc = PreprocessConfig {
        black_pt: cli.black_pt,
        white_pt: cli.white_pt,
        remove_underlines: true,
    };

    match cli.command {
        Some(Commands::Bench {
            image_path,
            debug,
            mode,
            max_ram_mb,
        }) => run_benchmark(&image_path, &models, mode.into(), preproc, debug, max_ram_mb),
        Some(Commands::Download) => {
            models.ensure_all_downloaded()?;
            println!("All ONNX models ready in {}", models.root.display());
            Ok(())
        }
        Some(Commands::InspectModels) => inspect_onnx_models(&models),
        None => daemon::run_daemon(models, preproc),
    }
}
