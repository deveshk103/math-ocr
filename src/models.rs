use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

/// Pinned Hugging Face revision for breezedeus/pix2text-mfr-1.5
const MFR_REV: &str = "1cef9f0bdcd6a4c63df7de1311fb0894593340cc";

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub root: PathBuf,
    pub mfd_onnx: PathBuf,
    pub mfr_encoder: PathBuf,
    pub mfr_decoder: PathBuf,
    pub mfr_tokenizer: PathBuf,
    pub mfr_config: PathBuf,
}

impl ModelPaths {
    pub fn resolve(override_dir: Option<&Path>) -> Result<Self> {
        let root = if let Some(dir) = override_dir {
            dir.to_path_buf()
        } else if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            PathBuf::from(local_app_data)
                .join("math-snip-ocr")
                .join("models")
        } else {
            PathBuf::from("./models")
        };

        Ok(Self {
            mfd_onnx: root.join("mfd").join("pix2text-mfd-1.5.onnx"),
            mfr_encoder: root.join("mfr").join("encoder_model.onnx"),
            mfr_decoder: root.join("mfr").join("decoder_model.onnx"),
            mfr_tokenizer: root.join("mfr").join("tokenizer.json"),
            mfr_config: root.join("mfr").join("config.json"),
            root,
        })
    }

    pub fn ensure_all_downloaded(&self) -> Result<()> {
        let files: &[(&Path, String)] = &[
            (
                &self.mfd_onnx,
                "https://huggingface.co/breezedeus/pix2text-mfd-1.5/resolve/main/pix2text-mfd-1.5.onnx".to_string(),
            ),
            (
                &self.mfr_encoder,
                format!("https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/{MFR_REV}/encoder_model.onnx"),
            ),
            (
                &self.mfr_decoder,
                format!("https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/{MFR_REV}/decoder_model.onnx"),
            ),
            (
                &self.mfr_tokenizer,
                format!("https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/{MFR_REV}/tokenizer.json"),
            ),
            (
                &self.mfr_config,
                format!("https://huggingface.co/breezedeus/pix2text-mfr-1.5/resolve/{MFR_REV}/config.json"),
            ),
        ];

        for (dest, url) in files {
            if !dest.exists() {
                download_atomic(url, dest)?;
            }
        }
        Ok(())
    }
}

/// Builds a low-RAM CPU-only ONNX Runtime session (`ort 2.0.0-rc.13`).
/// Maps `!Send + !Sync` `ort::Error<SessionBuilder>` into `anyhow::Error` cleanly.
pub fn build_cpu_session(model_path: &Path, intra_threads: usize) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(|e| anyhow!("ORT Session::builder() failed: {e}"))?
        .with_intra_threads(intra_threads)
        .map_err(|e| anyhow!("ORT with_intra_threads({intra_threads}) failed: {e}"))?
        .with_inter_threads(1)
        .map_err(|e| anyhow!("ORT with_inter_threads(1) failed: {e}"))?
        .with_memory_pattern(false)
        .map_err(|e| anyhow!("ORT with_memory_pattern(false) failed: {e}"))?;

    builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow!("Failed to load ONNX model {}: {e}", model_path.display()))
}

fn download_atomic(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    eprintln!("[models] Downloading {} -> {}", url, dest.display());
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP GET failed for {url}"))?;

    let tmp_path = dest.with_extension("part");
    {
        let mut reader = resp.into_reader();
        let mut out = File::create(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?;
        io::copy(&mut reader, &mut out)
            .with_context(|| format!("Failed while writing {}", tmp_path.display()))?;
    }
    fs::rename(&tmp_path, dest)?;
    Ok(())
}

/// Inspects and prints actual ONNX input/output tensor names and shapes for every downloaded model.
pub fn inspect_onnx_models(paths: &ModelPaths) -> Result<()> {
    paths.ensure_all_downloaded()?;
    for (label, path) in [
        ("MFD (pix2text-mfd-1.5)", &paths.mfd_onnx),
        ("MFR Encoder (pix2text-mfr-1.5)", &paths.mfr_encoder),
        ("MFR Decoder (pix2text-mfr-1.5)", &paths.mfr_decoder),
    ] {
        let session = build_cpu_session(path, 1)?;

        println!("=== {label} ({}) ===", path.display());
        println!("  Inputs:");
        for inp in session.inputs() {
            println!("    - name: {:?}, dtype: {:?}", inp.name(), inp.dtype());
        }
        println!("  Outputs:");
        for out in session.outputs() {
            println!("    - name: {:?}, dtype: {:?}", out.name(), out.dtype());
        }
    }
    Ok(())
}
