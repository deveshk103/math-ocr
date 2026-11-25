use crate::types::{Box2D, FormulaBox, FormulaKind, TextSpan};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};

#[derive(Debug, Clone, Copy)]
pub struct PreprocessConfig {
    pub black_pt: u8,
    pub white_pt: u8,
    pub remove_underlines: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            black_pt: 45,
            white_pt: 175,
            remove_underlines: true,
        }
    }
}

/// Runs the full parent-process image conditioning pipeline:
/// 1. Dark-mode auto-inversion (median luma < 128)
/// 2. Watermark removal LUT (`black_pt..white_pt` -> `0..255`)
/// 3. Fill-in-the-blank underline cleanup (preserving minus signs & fraction bars)
pub fn preprocess_image(input: &RgbImage, cfg: PreprocessConfig) -> RgbImage {
    let mut img = input.clone();
    auto_invert_dark_mode(&mut img);
    apply_watermark_lut(&mut img, cfg.black_pt, cfg.white_pt);
    if cfg.remove_underlines {
        remove_blank_underlines(&mut img);
    }
    img
}

/// Compute median grayscale luma via 256-bin histogram; invert if dark mode (< 128).
pub fn auto_invert_dark_mode(img: &mut RgbImage) {
    let mut hist = [0u64; 256];
    let total = (img.width() as u64) * (img.height() as u64);
    if total == 0 {
        return;
    }

    for px in img.pixels() {
        let luma = (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
            .round()
            .clamp(0.0, 255.0) as usize;
        hist[luma] += 1;
    }

    let half = total / 2;
    let mut acc = 0u64;
    let mut median = 255u8;
    for (val, &count) in hist.iter().enumerate() {
        acc += count;
        if acc >= half {
            median = val as u8;
            break;
        }
    }

    if median < 128 {
        for px in img.pixels_mut() {
            px[0] = 255 - px[0];
            px[1] = 255 - px[1];
            px[2] = 255 - px[2];
        }
    }
}

/// Remap channel intensities using a linear LUT from `[black_pt, white_pt]` to `[0, 255]`.
/// Pushes light-grey diagonal PDF watermarks (typically luma 185–235) cleanly to 255
/// while preserving anti-aliased thin strokes (minus signs, decimals, fraction bars).
pub fn apply_watermark_lut(img: &mut RgbImage, black_pt: u8, white_pt: u8) {
    let mut lut = [0u8; 256];
    let span = (white_pt as i32 - black_pt as i32).max(1) as f32;
    for v in 0..256 {
        let vu = v as u8;
        lut[v] = if vu <= black_pt {
            0
        } else if vu >= white_pt {
            255
        } else {
            (((vu - black_pt) as f32) * 255.0 / span)
                .round()
                .clamp(0.0, 255.0) as u8
        };
    }

    for px in img.pixels_mut() {
        px[0] = lut[px[0] as usize];
        px[1] = lut[px[1] as usize];
        px[2] = lut[px[2] as usize];
    }
}

/// Erases long fill-in-the-blank underscores (`________`) while leaving short minus signs
/// and fraction bars (which have numerator/denominator ink directly above/below) intact.
pub fn remove_blank_underlines(img: &mut RgbImage) {
    let w = img.width() as usize;
    let h = img.height() as usize;
    if w < 50 || h < 10 {
        return;
    }

    let mut dark = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let px = img.get_pixel(x as u32, y as u32);
            let luma = (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as u8;
            dark[y * w + x] = luma < 180;
        }
    }

    let row_dark_ratio = |dark_buf: &[bool], row: usize, x0: usize, x1: usize| -> f32 {
        let mut cnt = 0usize;
        for x in x0..x1 {
            if dark_buf[row * w + x] {
                cnt += 1;
            }
        }
        cnt as f32 / (x1 - x0).max(1) as f32
    };

    let band_dark_ratio =
        |dark_buf: &[bool], y0: usize, y1: usize, x0: usize, x1: usize| -> f32 {
            if y1 <= y0 || x1 <= x0 {
                return 0.0;
            }
            let mut cnt = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    if dark_buf[y * w + x] {
                        cnt += 1;
                    }
                }
            }
            cnt as f32 / ((y1 - y0) * (x1 - x0)) as f32
        };

    for y in 0..h {
        let mut x = 0usize;
        while x < w {
            if dark[y * w + x] {
                let x0 = x;
                while x < w && dark[y * w + x] {
                    x += 1;
                }
                let run_w = x - x0;
                if run_w >= 45 {
                    let mut y_top = y;
                    while y_top > 0 && row_dark_ratio(&dark, y_top - 1, x0, x) > 0.7 {
                        y_top -= 1;
                    }
                    let mut y_bot = y;
                    while y_bot + 1 < h && row_dark_ratio(&dark, y_bot + 1, x0, x) > 0.7 {
                        y_bot += 1;
                    }
                    let thickness = y_bot - y_top + 1;
                    if thickness <= 5 {
                        let above_y0 = y_top.saturating_sub(14);
                        let above_y1 = y_top.saturating_sub(2);
                        let below_y0 = (y_bot + 3).min(h);
                        let below_y1 = (y_bot + 14).min(h);
                        let above_ink = band_dark_ratio(&dark, above_y0, above_y1, x0, x);
                        let below_ink = band_dark_ratio(&dark, below_y0, below_y1, x0, x);
                        if above_ink < 0.03 && below_ink < 0.03 {
                            let pad_y0 = y_top.saturating_sub(1);
                            let pad_y1 = (y_bot + 2).min(h);
                            let pad_x0 = x0.saturating_sub(1);
                            let pad_x1 = (x + 1).min(w);
                            for py in pad_y0..pad_y1 {
                                for px in pad_x0..pad_x1 {
                                    img.put_pixel(px as u32, py as u32, Rgb([255, 255, 255]));
                                    dark[py * w + px] = false;
                                }
                            }
                        }
                    }
                }
            } else {
                x += 1;
            }
        }
    }
}

/// Masks detected formula boxes to pure white `(255, 255, 255)` (shrunk by `shrink_px`)
/// before passing the image to Stage 2 text OCR.
pub fn mask_formula_boxes(img: &RgbImage, boxes: &[FormulaBox], shrink_px: i32) -> RgbImage {
    let mut masked = img.clone();
    let w = img.width() as i32;
    let h = img.height() as i32;

    for fb in boxes {
        let x1 = (fb.bbox.x1.floor() as i32 + shrink_px).clamp(0, w);
        let y1 = (fb.bbox.y1.floor() as i32 + shrink_px).clamp(0, h);
        let x2 = (fb.bbox.x2.ceil() as i32 - shrink_px).clamp(0, w);
        let y2 = (fb.bbox.y2.ceil() as i32 - shrink_px).clamp(0, h);
        for y in y1..y2 {
            for x in x1..x2 {
                masked.put_pixel(x as u32, y as u32, Rgb([255, 255, 255]));
            }
        }
    }
    masked
}

/// Crops a formula box, adds `pad_px` (8–12 px) white border padding,
/// and upscales tiny crops (`height < 64 px`) by 2x using Lanczos3.
pub fn condition_formula_crop(img: &RgbImage, bbox: Box2D, pad_px: u32) -> RgbImage {
    let w = img.width() as f32;
    let h = img.height() as f32;
    let x1 = bbox.x1.floor().clamp(0.0, w - 1.0) as u32;
    let y1 = bbox.y1.floor().clamp(0.0, h - 1.0) as u32;
    let x2 = bbox.x2.ceil().clamp((x1 + 1) as f32, w) as u32;
    let y2 = bbox.y2.ceil().clamp((y1 + 1) as f32, h) as u32;

    let cw = (x2 - x1).max(1);
    let ch = (y2 - y1).max(1);
    let crop = image::imageops::crop_imm(img, x1, y1, cw, ch).to_image();

    let mut padded = RgbImage::from_pixel(cw + 2 * pad_px, ch + 2 * pad_px, Rgb([255, 255, 255]));
    image::imageops::overlay(&mut padded, &crop, pad_px as i64, pad_px as i64);

    if padded.height() < 64 {
        image::imageops::resize(
            &padded,
            padded.width() * 2,
            padded.height() * 2,
            FilterType::Lanczos3,
        )
    } else {
        padded
    }
}

/// Renders debug bounding boxes onto a copy of the cleaned image:
/// Blue = text spans, Orange = inline math, Red = isolated math.
pub fn render_debug_boxes(
    img: &RgbImage,
    text_spans: &[TextSpan],
    formula_boxes: &[FormulaBox],
) -> RgbImage {
    let mut canvas = img.clone();
    for ts in text_spans {
        draw_rect(&mut canvas, ts.bbox, Rgb([30, 120, 240]), 2);
    }
    for fb in formula_boxes {
        let color = match fb.kind {
            FormulaKind::Inline => Rgb([245, 145, 20]),
            FormulaKind::Isolated => Rgb([230, 35, 45]),
        };
        draw_rect(&mut canvas, fb.bbox, color, 2);
    }
    canvas
}

fn draw_rect(img: &mut RgbImage, bbox: Box2D, color: Rgb<u8>, thickness: i32) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let x1 = (bbox.x1.round() as i32).clamp(0, w - 1);
    let y1 = (bbox.y1.round() as i32).clamp(0, h - 1);
    let x2 = (bbox.x2.round() as i32).clamp(0, w - 1);
    let y2 = (bbox.y2.round() as i32).clamp(0, h - 1);

    for t in 0..thickness {
        let yt = (y1 + t).clamp(0, h - 1);
        let yb = (y2 - t).clamp(0, h - 1);
        for x in x1..=x2 {
            img.put_pixel(x as u32, yt as u32, color);
            img.put_pixel(x as u32, yb as u32, color);
        }
        let xl = (x1 + t).clamp(0, w - 1);
        let xr = (x2 - t).clamp(0, w - 1);
        for y in y1..=y2 {
            img.put_pixel(xl as u32, y as u32, color);
            img.put_pixel(xr as u32, y as u32, color);
        }
    }
}
