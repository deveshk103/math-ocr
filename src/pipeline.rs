use crate::models::ModelPaths;
use crate::platform::peak_working_set_mb;
use crate::preprocess::{
    condition_formula_crop, mask_formula_boxes, preprocess_image, render_debug_boxes,
    PreprocessConfig,
};
use crate::stitch::{clean_latex, stitch_markdown};
use crate::types::{
    Box2D, FormulaKind, MfdOutput, MfrOutput, SnipMode, StageMetrics, TextOutput,
};
use anyhow::{anyhow, Context, Result};
use image::RgbImage;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tempfile::TempDir;

#[derive(Debug, Clone)]
pub struct PipelineOptions {
    pub mode: SnipMode,
    pub preproc: PreprocessConfig,
    pub debug_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub markdown: String,
    pub stages: Vec<StageMetrics>,
    pub parent_peak_mb: f64,
}

impl PipelineResult {
    pub fn max_peak_mb(&self) -> f64 {
        self.stages
            .iter()
            .map(|s| s.peak_working_set_mb)
            .fold(self.parent_peak_mb, f64::max)
    }
}

/// Orchestrates preprocessing in the parent process and spawns sequential short-lived
/// worker subprocesses (`current_exe() --worker <stage>`) so Windows reclaims 100% of
/// each ONNX Runtime arena before the next model loads.
pub fn run_pipeline(
    raw_img: &RgbImage,
    models: &ModelPaths,
    opts: &PipelineOptions,
) -> Result<PipelineResult> {
    models.ensure_all_downloaded()?;

    let tmp = TempDir::new().context("Failed to create temp IPC directory")?;
    let mut stages: Vec<StageMetrics> = Vec::new();

    // 0. Preprocess in parent process
    let t0 = Instant::now();
    let cleaned = preprocess_image(raw_img, opts.preproc);
    let cleaned_path = tmp.path().join("01_cleaned.png");
    cleaned.save(&cleaned_path)?;
    stages.push(StageMetrics {
        stage: "preprocess (parent)".to_string(),
        elapsed_ms: t0.elapsed().as_millis() as u64,
        peak_working_set_mb: peak_working_set_mb(),
    });

    if let Some(ref dbg) = opts.debug_dir {
        fs::create_dir_all(dbg)?;
        cleaned.save(dbg.join("01_cleaned.png"))?;
    }

    match opts.mode {
        SnipMode::Math => {
            // Pure Math mode (Alt + M): skip MFD and Text OCR, send conditioned full crop to MFR
            let full_box = Box2D {
                x1: 0.0,
                y1: 0.0,
                x2: cleaned.width() as f32,
                y2: cleaned.height() as f32,
            };
            let crop = condition_formula_crop(&cleaned, full_box, 10);
            let crop_path = tmp.path().join("crop_0.png");
            crop.save(&crop_path)?;

            let mfr: MfrOutput = spawn_worker_json(&[
                "--worker",
                "mfr",
                "--model-dir",
                &models.root.to_string_lossy(),
                &crop_path.to_string_lossy(),
            ])?;
            stages.push(mfr.metrics);
            let latex = mfr
                .formulas
                .first()
                .map(|s| clean_latex(s))
                .unwrap_or_default();

            Ok(PipelineResult {
                markdown: latex,
                stages,
                parent_peak_mb: peak_working_set_mb(),
            })
        }
        SnipMode::Text => {
            // Pure Text mode (Alt + T): run Stage 2 only
            let text_out: TextOutput = spawn_worker_json(&[
                "--worker",
                "text",
                "--model-dir",
                &models.root.to_string_lossy(),
                &cleaned_path.to_string_lossy(),
            ])?;
            stages.push(text_out.metrics);
            let plain = stitch_markdown(&text_out.spans, &[], &[]);

            Ok(PipelineResult {
                markdown: plain,
                stages,
                parent_peak_mb: peak_working_set_mb(),
            })
        }
        SnipMode::Mixed => {
            // Stage 1: MFD (Formula Detection)
            let mfd: MfdOutput = spawn_worker_json(&[
                "--worker",
                "mfd",
                "--model-dir",
                &models.root.to_string_lossy(),
                &cleaned_path.to_string_lossy(),
            ])?;
            stages.push(mfd.metrics);

            // Mask formula boxes to white (shrunk by 1 px) before Stage 2 text OCR
            let masked = mask_formula_boxes(&cleaned, &mfd.boxes, 1);
            let masked_path = tmp.path().join("03_masked_for_text.png");
            masked.save(&masked_path)?;
            if let Some(ref dbg) = opts.debug_dir {
                masked.save(dbg.join("03_masked_for_text.png"))?;
            }

            // Stage 2: Text OCR on masked image
            let text_out: TextOutput = spawn_worker_json(&[
                "--worker",
                "text",
                "--model-dir",
                &models.root.to_string_lossy(),
                &masked_path.to_string_lossy(),
            ])?;
            stages.push(text_out.metrics);

            if let Some(ref dbg) = opts.debug_dir {
                let boxes_vis = render_debug_boxes(&cleaned, &text_out.spans, &mfd.boxes);
                boxes_vis.save(dbg.join("02_boxes.png"))?;
            }

            // Stage 3: MFR on conditioned formula crops (only if formulas were detected)
            let formula_latex = if mfd.boxes.is_empty() {
                Vec::new()
            } else {
                let mut crop_args: Vec<String> = vec![
                    "--worker".into(),
                    "mfr".into(),
                    "--model-dir".into(),
                    models.root.to_string_lossy().to_string(),
                ];
                for (idx, fb) in mfd.boxes.iter().enumerate() {
                    let pad = if fb.kind == FormulaKind::Isolated { 12 } else { 10 };
                    let crop = condition_formula_crop(&cleaned, fb.bbox, pad);
                    let crop_path = tmp.path().join(format!("crop_{idx}.png"));
                    crop.save(&crop_path)?;
                    crop_args.push(crop_path.to_string_lossy().to_string());
                }
                let arg_refs: Vec<&str> = crop_args.iter().map(String::as_str).collect();
                let mfr: MfrOutput = spawn_worker_json(&arg_refs)?;
                stages.push(mfr.metrics);
                mfr.formulas
            };

            let markdown = stitch_markdown(&text_out.spans, &mfd.boxes, &formula_latex);
            Ok(PipelineResult {
                markdown,
                stages,
                parent_peak_mb: peak_working_set_mb(),
            })
        }
    }
}

fn spawn_worker_json<T: serde::de::DeserializeOwned>(args: &[&str]) -> Result<T> {
    let exe = std::env::current_exe().context("Failed to locate current executable")?;
    let output = Command::new(&exe)
        .args(args)
        .output()
        .with_context(|| format!("Failed to spawn worker subprocess: {args:?}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "Worker subprocess failed ({}):\n{}",
            output.status,
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Parse the last non-empty JSON line (in case ORT wrote any info lines to stdout)
    let json_line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .ok_or_else(|| anyhow!("Worker produced no JSON output. Raw stdout:\n{stdout}"))?;

    serde_json::from_str(json_line)
        .with_context(|| format!("Failed to parse worker JSON:\n{json_line}"))
}

/// Helper for `bench` CLI mode to load a local PNG and print full stage metrics + RAM verification.
pub fn run_benchmark(
    image_path: &Path,
    models: &ModelPaths,
    mode: SnipMode,
    preproc: PreprocessConfig,
    debug: bool,
    max_ram_mb: f64,
) -> Result<()> {
    let raw = image::open(image_path)
        .with_context(|| format!("Failed to open benchmark image {}", image_path.display()))?
        .to_rgb8();

    let debug_dir = if debug {
        Some(PathBuf::from("./debug"))
    } else {
        None
    };

    let res = run_pipeline(
        &raw,
        models,
        &PipelineOptions {
            mode,
            preproc,
            debug_dir,
        },
    )?;

    println!("================ MARKDOWN / LATEX OUTPUT ================");
    println!("{}", res.markdown);
    println!("=========================================================");
    println!("\nStage Metrics (Windows PeakWorkingSetSize):");
    println!("  {:<22} {:>10} {:>14}", "Stage", "Latency", "Peak RAM");
    for st in &res.stages {
        println!(
            "  {:<22} {:>8} ms {:>11.2} MB",
            st.stage, st.elapsed_ms, st.peak_working_set_mb
        );
    }
    println!(
        "  {:<22} {:>10} {:>11.2} MB",
        "parent (overall)", "-", res.parent_peak_mb
    );

    let peak = res.max_peak_mb();
    println!("\nMax Peak RAM across all stages: {peak:.2} MB (budget: <= {max_ram_mb:.0} MB)");
    if peak > max_ram_mb {
        return Err(anyhow!(
            "RAM budget exceeded: measured {peak:.2} MB > {max_ram_mb:.0} MB"
        ));
    }
    Ok(())
}
