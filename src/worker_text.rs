use crate::platform::peak_working_set_mb;
use crate::types::{Box2D, StageMetrics, TextOutput, TextSpan};
use anyhow::{Context, Result};
use oar_ocr::core::config::{OrtExecutionProvider, OrtSessionConfig};
use oar_ocr::prelude::*;
use std::path::Path;
use std::time::Instant;

/// Stage 2 Worker (`--worker text`, target peak < 160 MB):
/// Runs `oar-ocr` PP-OCRv6 Small (pure CPU ONNX) on the formula-masked image
/// and prints JSON `TextOutput` to stdout before exiting so Windows reclaims 100% of RAM.
pub fn run_text_worker(image_path: &Path) -> Result<()> {
    let t0 = Instant::now();

    let ort_cfg = OrtSessionConfig::new()
        .with_execution_providers(vec![OrtExecutionProvider::CPU])
        .with_intra_threads(2)
        .with_inter_threads(1);

    let ocr = OAROCRBuilder::pp_ocrv6(PpOcrV6Size::Small)
        .ort_session(ort_cfg)
        .image_batch_size(1)
        .region_batch_size(4)
        .build()
        .context("Failed to initialize oar-ocr PP-OCRv6 Small pipeline")?;

    let img = load_image(image_path)
        .with_context(|| format!("Failed to load masked image {}", image_path.display()))?;
    let results = ocr
        .predict(vec![img])
        .context("oar-ocr text prediction failed")?;

    let mut spans = Vec::new();
    if let Some(res) = results.first() {
        for region in &res.text_regions {
            if let Some((text, confidence)) = region.text_with_confidence() {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let bb = &region.bounding_box;
                spans.push(TextSpan {
                    bbox: Box2D {
                        x1: bb.x_min(),
                        y1: bb.y_min(),
                        x2: bb.x_max(),
                        y2: bb.y_max(),
                    },
                    text: trimmed.to_string(),
                    confidence,
                });
            }
        }
    }

    let out = TextOutput {
        spans,
        metrics: StageMetrics {
            stage: "text".to_string(),
            elapsed_ms: t0.elapsed().as_millis() as u64,
            peak_working_set_mb: peak_working_set_mb(),
        },
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
