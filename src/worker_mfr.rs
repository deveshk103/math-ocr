use crate::models::{build_cpu_session, ModelPaths};
use crate::platform::peak_working_set_mb;
use crate::types::{MfrOutput, StageMetrics};
use anyhow::{anyhow, Context, Result};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use ort::session::Session;
use ort::value::Tensor;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const MFR_INPUT_SIZE: u32 = 384;
const MAX_DECODE_STEPS: usize = 512;

/// Stage 3 Worker (`--worker mfr`, target peak < 350 MB):
/// Runs `breezedeus/pix2text-mfr-1.5` ONNX encoder + autoregressive decoder via `ort` + `tokenizers`.
/// Executes in two memory-bounded phases:
///   Phase A: Load `encoder_model.onnx`, encode all formula crops, then DROP encoder session.
///   Phase B: Load `decoder_model.onnx`, greedy-decode each hidden state, then exit immediately.
pub fn run_mfr_worker(crop_paths: &[PathBuf], models: &ModelPaths) -> Result<()> {
    let t0 = Instant::now();

    let tokenizer = Tokenizer::from_file(&models.mfr_tokenizer)
        .map_err(|e| anyhow!("Failed to load tokenizer {}: {e}", models.mfr_tokenizer.display()))?;

    let (bos_id, eos_id) = read_special_token_ids(&models.mfr_config);

    // Phase A: Encode all crops, then drop encoder_session before loading decoder_model.onnx
    let mut encoded_states: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(crop_paths.len());
    {
        let mut encoder = build_cpu_session(&models.mfr_encoder, 2)?;

        let enc_input_name = encoder
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "pixel_values".to_string());

        for path in crop_paths {
            let img = image::open(path)
                .with_context(|| format!("Failed to open crop {}", path.display()))?
                .to_rgb8();
            let pixel_values = preprocess_mfr_384(&img);
            let inp = Tensor::from_array((
                [1usize, 3, MFR_INPUT_SIZE as usize, MFR_INPUT_SIZE as usize],
                pixel_values,
            ))?;
            let outputs = encoder.run(ort::inputs![enc_input_name.as_str() => inp])?;
            let hidden = outputs
                .values()
                .next()
                .ok_or_else(|| anyhow!("MFR encoder returned no output"))?;
            let (shape, data) = hidden.try_extract_tensor::<f32>()?;
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            encoded_states.push((dims, data.to_vec()));
        }
        // encoder dropped here before decoder loads
    }

    // Phase B: Greedy autoregressive decoding
    let mut decoder = build_cpu_session(&models.mfr_decoder, 2)?;

    let input_names: Vec<String> = decoder
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    let ids_name = input_names
        .iter()
        .find(|n| n.contains("input_ids"))
        .cloned()
        .unwrap_or_else(|| "input_ids".to_string());
    let enc_hidden_name = input_names
        .iter()
        .find(|n| n.contains("encoder_hidden_states"))
        .cloned()
        .unwrap_or_else(|| "encoder_hidden_states".to_string());

    let mut formulas = Vec::with_capacity(encoded_states.len());
    for (enc_dims, enc_data) in encoded_states {
        let latex = decode_greedy(
            &mut decoder,
            &tokenizer,
            &ids_name,
            &enc_hidden_name,
            &enc_dims,
            &enc_data,
            bos_id,
            eos_id,
        )?;
        formulas.push(latex);
    }

    let out = MfrOutput {
        formulas,
        metrics: StageMetrics {
            stage: "mfr".to_string(),
            elapsed_ms: t0.elapsed().as_millis() as u64,
            peak_working_set_mb: peak_working_set_mb(),
        },
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}

/// Renders the formula crop centered onto a 384x384 white canvas (preserving aspect ratio)
/// and normalizes channel-first RGB to `[-1.0, 1.0]` matching `pix2text-mfr-1.5`.
fn preprocess_mfr_384(img: &RgbImage) -> Vec<f32> {
    let scale = (MFR_INPUT_SIZE as f32 / img.width() as f32)
        .min(MFR_INPUT_SIZE as f32 / img.height() as f32);
    let new_w = (img.width() as f32 * scale).round().clamp(1.0, MFR_INPUT_SIZE as f32) as u32;
    let new_h = (img.height() as f32 * scale).round().clamp(1.0, MFR_INPUT_SIZE as f32) as u32;

    let resized = image::imageops::resize(img, new_w, new_h, FilterType::Lanczos3);
    let mut canvas = RgbImage::from_pixel(MFR_INPUT_SIZE, MFR_INPUT_SIZE, Rgb([255, 255, 255]));
    let off_x = ((MFR_INPUT_SIZE - new_w) / 2) as i64;
    let off_y = ((MFR_INPUT_SIZE - new_h) / 2) as i64;
    image::imageops::overlay(&mut canvas, &resized, off_x, off_y);

    let hw = (MFR_INPUT_SIZE * MFR_INPUT_SIZE) as usize;
    let mut nchw = vec![0.0f32; 3 * hw];
    for y in 0..MFR_INPUT_SIZE as usize {
        for x in 0..MFR_INPUT_SIZE as usize {
            let px = canvas.get_pixel(x as u32, y as u32);
            let idx = y * MFR_INPUT_SIZE as usize + x;
            nchw[idx] = (px[0] as f32 / 255.0 - 0.5) / 0.5;
            nchw[hw + idx] = (px[1] as f32 / 255.0 - 0.5) / 0.5;
            nchw[2 * hw + idx] = (px[2] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    nchw
}

fn read_special_token_ids(config_path: &PathBuf) -> (i64, i64) {
    // Verified defaults from breezedeus/pix2text-mfr-1.5 config.json
    let mut bos = 1i64;
    let mut eos = 2i64;
    if let Ok(raw) = fs::read_to_string(config_path) {
        if let Ok(v) = serde_json::from_str::<Value>(&raw) {
            if let Some(id) = v.get("decoder_start_token_id").and_then(|x| x.as_i64()) {
                bos = id;
            }
            if let Some(id) = v.get("eos_token_id").and_then(|x| x.as_i64()) {
                eos = id;
            }
        }
    }
    (bos, eos)
}

#[allow(clippy::too_many_arguments)]
fn decode_greedy(
    decoder: &mut Session,
    tokenizer: &Tokenizer,
    ids_name: &str,
    enc_hidden_name: &str,
    enc_dims: &[usize],
    enc_data: &[f32],
    bos_id: i64,
    eos_id: i64,
) -> Result<String> {
    let mut token_ids: Vec<i64> = vec![bos_id];

    for _step in 0..MAX_DECODE_STEPS {
        let seq_len = token_ids.len();
        let ids_tensor = Tensor::from_array(([1usize, seq_len], token_ids.clone()))?;
        let enc_tensor = Tensor::from_array((enc_dims.to_vec(), enc_data.to_vec()))?;

        let outputs = decoder.run(ort::inputs![
            ids_name => ids_tensor,
            enc_hidden_name => enc_tensor
        ])?;
        let logits_val = outputs
            .values()
            .next()
            .ok_or_else(|| anyhow!("MFR decoder returned no logits"))?;
        let (shape, logits) = logits_val.try_extract_tensor::<f32>()?;
        if shape.len() != 3 {
            return Err(anyhow!("Unexpected MFR logits shape: {shape:?}"));
        }
        let vocab_size = shape[2] as usize;
        let last_pos_offset = (seq_len - 1) * vocab_size;
        let slice = &logits[last_pos_offset..last_pos_offset + vocab_size];

        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in slice.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i;
            }
        }

        if best_id as i64 == eos_id {
            break;
        }
        token_ids.push(best_id as i64);

        // Simple repetition guard against degenerate loops
        let n = token_ids.len();
        if n >= 16 && token_ids[n - 8..n] == token_ids[n - 16..n - 8] {
            break;
        }
    }

    let u32_ids: Vec<u32> = token_ids
        .into_iter()
        .skip(1) // drop BOS
        .map(|id| id as u32)
        .collect();
    let decoded = tokenizer
        .decode(&u32_ids, true)
        .map_err(|e| anyhow!("Tokenizer decode error: {e}"))?;
    Ok(decoded.trim().to_string())
}
