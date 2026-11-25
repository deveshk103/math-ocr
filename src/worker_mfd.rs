use crate::models::build_cpu_session;
use crate::platform::peak_working_set_mb;
use crate::types::{Box2D, FormulaBox, FormulaKind, MfdOutput, StageMetrics};
use anyhow::{anyhow, Context, Result};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::time::Instant;

/// Stage 1 Worker (`--worker mfd`, target peak < 180 MB):
/// Runs `pix2text-mfd-1.5.onnx` via `ort` (CPU-only, intra_threads=2, inter_threads=1)
/// and prints JSON `MfdOutput` to stdout before exiting so Windows reclaims 100% of RAM.
pub fn run_mfd_worker(image_path: &Path, model_path: &Path) -> Result<()> {
    let t0 = Instant::now();
    let img = image::open(image_path)
        .with_context(|| format!("Failed to open {}", image_path.display()))?
        .to_rgb8();

    let mut session = build_cpu_session(model_path, 2)?;

    // Inspect actual input tensor name and target resolution dynamically (default 768x768)
    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .unwrap_or_else(|| "images".to_string());
    let (target_h, target_w) = infer_yolo_input_hw(&session).unwrap_or((768, 768));

    let (tensor_data, scale, pad_x, pad_y) = letterbox_rgb(&img, target_w, target_h);
    let input_tensor =
        Tensor::from_array(([1usize, 3, target_h as usize, target_w as usize], tensor_data))?;

    let outputs = session.run(ort::inputs![input_name.as_str() => input_tensor])?;
    let first_out = outputs
        .values()
        .next()
        .ok_or_else(|| anyhow!("MFD session returned no outputs"))?;
    let (out_shape, out_data) = first_out.try_extract_tensor::<f32>()?;

    let raw_boxes = decode_yolo_detections(
        &out_shape,
        out_data,
        scale,
        pad_x,
        pad_y,
        img.width() as f32,
        img.height() as f32,
        0.25,
    )?;

    let nms_boxes = apply_nms(raw_boxes, 0.45);
    let merged_boxes = preserve_piecewise_isolated_blocks(nms_boxes);

    let out = MfdOutput {
        boxes: merged_boxes,
        metrics: StageMetrics {
            stage: "mfd".to_string(),
            elapsed_ms: t0.elapsed().as_millis() as u64,
            peak_working_set_mb: peak_working_set_mb(),
        },
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

fn infer_yolo_input_hw(session: &Session) -> Option<(u32, u32)> {
    let inp = session.inputs().first()?;
    let dims = inp.dtype().tensor_shape()?;
    if dims.len() == 4 && dims[2] > 0 && dims[3] > 0 {
        Some((dims[2] as u32, dims[3] as u32))
    } else {
        None
    }
}

fn letterbox_rgb(img: &RgbImage, target_w: u32, target_h: u32) -> (Vec<f32>, f32, f32, f32) {
    let src_w = img.width() as f32;
    let src_h = img.height() as f32;
    let scale = (target_w as f32 / src_w).min(target_h as f32 / src_h);
    let new_w = (src_w * scale).round().max(1.0) as u32;
    let new_h = (src_h * scale).round().max(1.0) as u32;
    let pad_x = ((target_w - new_w) / 2) as f32;
    let pad_y = ((target_h - new_h) / 2) as f32;

    let resized = image::imageops::resize(img, new_w, new_h, FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(target_w, target_h, Rgb([114, 114, 114]));
    image::imageops::overlay(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    let hw = (target_w * target_h) as usize;
    let mut nchw = vec![0.0f32; 3 * hw];
    for y in 0..target_h as usize {
        for x in 0..target_w as usize {
            let px = canvas.get_pixel(x as u32, y as u32);
            let idx = y * target_w as usize + x;
            nchw[idx] = px[0] as f32 / 255.0;
            nchw[hw + idx] = px[1] as f32 / 255.0;
            nchw[2 * hw + idx] = px[2] as f32 / 255.0;
        }
    }
    (nchw, scale, pad_x, pad_y)
}

/// Decodes YOLOv8/YOLOv11 ONNX output supporting both:
/// - `[1, 4 + C, N]` (raw Ultralytics export, e.g. `[1, 6, 12096]` for 2 classes: inline=0, isolated=1)
/// - `[1, N, 6]` (post-NMS `[x1, y1, x2, y2, score, class_id]`)
#[allow(clippy::too_many_arguments)]
fn decode_yolo_detections(
    shape: &[i64],
    data: &[f32],
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    img_w: f32,
    img_h: f32,
    conf_thresh: f32,
) -> Result<Vec<FormulaBox>> {
    let mut boxes = Vec::new();
    if shape.len() != 3 {
        return Err(anyhow!("Unexpected MFD output rank: {shape:?}"));
    }
    let d1 = shape[1] as usize;
    let d2 = shape[2] as usize;

    // Case A: [1, 4 + num_classes, num_anchors] (standard YOLOv8/v11 ONNX export)
    if d1 <= 16 && d2 > d1 {
        let num_classes = d1.saturating_sub(4);
        for i in 0..d2 {
            let cx = data[0 * d2 + i];
            let cy = data[1 * d2 + i];
            let bw = data[2 * d2 + i];
            let bh = data[3 * d2 + i];

            let mut best_cls = 0usize;
            let mut best_score = 0.0f32;
            for c in 0..num_classes {
                let sc = data[(4 + c) * d2 + i];
                if sc > best_score {
                    best_score = sc;
                    best_cls = c;
                }
            }
            if best_score < conf_thresh {
                continue;
            }
            let x1 = ((cx - bw * 0.5 - pad_x) / scale).clamp(0.0, img_w);
            let y1 = ((cy - bh * 0.5 - pad_y) / scale).clamp(0.0, img_h);
            let x2 = ((cx + bw * 0.5 - pad_x) / scale).clamp(0.0, img_w);
            let y2 = ((cy + bh * 0.5 - pad_y) / scale).clamp(0.0, img_h);
            if x2 - x1 < 4.0 || y2 - y1 < 4.0 {
                continue;
            }
            let kind = if best_cls == 1 {
                FormulaKind::Isolated
            } else {
                FormulaKind::Inline
            };
            boxes.push(FormulaBox {
                bbox: Box2D { x1, y1, x2, y2 },
                kind,
                score: best_score,
            });
        }
    } else if d2 == 6 {
        // Case B: [1, N, 6] => [x1, y1, x2, y2, score, class_id]
        for i in 0..d1 {
            let base = i * 6;
            let score = data[base + 4];
            if score < conf_thresh {
                continue;
            }
            let cls = data[base + 5].round() as i32;
            let x1 = ((data[base] - pad_x) / scale).clamp(0.0, img_w);
            let y1 = ((data[base + 1] - pad_y) / scale).clamp(0.0, img_h);
            let x2 = ((data[base + 2] - pad_x) / scale).clamp(0.0, img_w);
            let y2 = ((data[base + 3] - pad_y) / scale).clamp(0.0, img_h);
            if x2 - x1 < 4.0 || y2 - y1 < 4.0 {
                continue;
            }
            let kind = if cls == 1 {
                FormulaKind::Isolated
            } else {
                FormulaKind::Inline
            };
            boxes.push(FormulaBox {
                bbox: Box2D { x1, y1, x2, y2 },
                kind,
                score,
            });
        }
    } else {
        return Err(anyhow!("Unsupported MFD output shape: {shape:?}"));
    }

    Ok(boxes)
}

fn apply_nms(mut boxes: Vec<FormulaBox>, iou_thresh: f32) -> Vec<FormulaBox> {
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<FormulaBox> = Vec::new();

    for cand in boxes {
        let suppressed = keep.iter().any(|k| k.bbox.iou(&cand.bbox) > iou_thresh);
        if !suppressed {
            keep.push(cand);
        }
    }
    keep
}

/// Ensures tall multi-row piecewise `{ ... }` isolated blocks remain intact as one `Isolated` box
/// by absorbing smaller overlapping fragments inside an isolated formula region.
fn preserve_piecewise_isolated_blocks(boxes: Vec<FormulaBox>) -> Vec<FormulaBox> {
    let mut isolated: Vec<FormulaBox> = boxes
        .iter()
        .filter(|b| b.kind == FormulaKind::Isolated)
        .cloned()
        .collect();
    let inline: Vec<FormulaBox> = boxes
        .into_iter()
        .filter(|b| b.kind == FormulaKind::Inline)
        .collect();

    // Merge overlapping or tightly adjacent isolated boxes (e.g., split left-hand `f(x) =` + cases brace)
    let mut merged_iso: Vec<FormulaBox> = Vec::new();
    for b in isolated.drain(..) {
        if let Some(existing) = merged_iso.iter_mut().find(|m| {
            let v_overlap = m.bbox.vertical_overlap_ratio(&b.bbox) >= 0.5;
            let h_gap = (b.bbox.x1 - m.bbox.x2).max(m.bbox.x1 - b.bbox.x2);
            v_overlap && h_gap <= 18.0
        }) {
            existing.bbox.x1 = existing.bbox.x1.min(b.bbox.x1);
            existing.bbox.y1 = existing.bbox.y1.min(b.bbox.y1);
            existing.bbox.x2 = existing.bbox.x2.max(b.bbox.x2);
            existing.bbox.y2 = existing.bbox.y2.max(b.bbox.y2);
            existing.score = existing.score.max(b.score);
        } else {
            merged_iso.push(b);
        }
    }

    // Drop inline boxes that are mostly contained inside an isolated block
    let mut out = merged_iso.clone();
    for inl in inline {
        let contained = merged_iso.iter().any(|iso| {
            let ix1 = iso.bbox.x1.max(inl.bbox.x1);
            let iy1 = iso.bbox.y1.max(inl.bbox.y1);
            let ix2 = iso.bbox.x2.min(inl.bbox.x2);
            let iy2 = iso.bbox.y2.min(inl.bbox.y2);
            let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
            inter / inl.bbox.area().max(1.0) > 0.5
        });
        if !contained {
            out.push(inl);
        }
    }

    // Sort top-to-bottom, left-to-right
    out.sort_by(|a, b| {
        a.bbox
            .y1
            .total_cmp(&b.bbox.y1)
            .then_with(|| a.bbox.x1.total_cmp(&b.bbox.x1))
    });
    out
}
