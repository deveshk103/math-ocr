use crate::types::{Box2D, FormulaBox, FormulaKind, TextSpan};
use regex::Regex;

#[derive(Debug, Clone)]
enum LayoutItem {
    Text { bbox: Box2D, content: String },
    InlineMath { bbox: Box2D, latex: String },
    IsolatedMath { bbox: Box2D, latex: String },
}

impl LayoutItem {
    fn bbox(&self) -> Box2D {
        match self {
            LayoutItem::Text { bbox, .. }
            | LayoutItem::InlineMath { bbox, .. }
            | LayoutItem::IsolatedMath { bbox, .. } => *bbox,
        }
    }

    fn is_isolated(&self) -> bool {
        matches!(self, LayoutItem::IsolatedMath { .. })
    }
}

/// Merges Stage 2 text spans and Stage 3 formula strings into reading-order Markdown + LaTeX.
/// - Inline math is embedded as `$...$` within prose lines.
/// - Isolated / multi-line (`\begin{cases}`, etc.) math is emitted on its own `$$...$$` block line.
pub fn stitch_markdown(
    text_spans: &[TextSpan],
    formula_boxes: &[FormulaBox],
    formula_latex: &[String],
) -> String {
    let mut items: Vec<LayoutItem> = Vec::with_capacity(text_spans.len() + formula_boxes.len());

    for ts in text_spans {
        let trimmed = ts.text.trim();
        if !trimmed.is_empty() {
            items.push(LayoutItem::Text {
                bbox: ts.bbox,
                content: trimmed.to_string(),
            });
        }
    }

    for (fb, raw_latex) in formula_boxes.iter().zip(formula_latex.iter()) {
        let cleaned = clean_latex(raw_latex);
        if cleaned.is_empty() {
            continue;
        }
        let is_multiline = cleaned.contains(r"\begin{cases}")
            || cleaned.contains(r"\begin{aligned}")
            || cleaned.contains(r"\\");
        if fb.kind == FormulaKind::Isolated || is_multiline {
            items.push(LayoutItem::IsolatedMath {
                bbox: fb.bbox,
                latex: cleaned,
            });
        } else {
            items.push(LayoutItem::InlineMath {
                bbox: fb.bbox,
                latex: cleaned,
            });
        }
    }

    // Sort roughly top-to-bottom by vertical midpoint, then left-to-right
    items.sort_by(|a, b| {
        let ay = (a.bbox().y1 + a.bbox().y2) * 0.5;
        let by = (b.bbox().y1 + b.bbox().y2) * 0.5;
        ay.total_cmp(&by).then_with(|| a.bbox().x1.total_cmp(&b.bbox().x1))
    });

    // Cluster into reading lines using vertical overlap ratio.
    // Isolated multi-line blocks form their own dedicated block line unless they share a short
    // inline lead-in like "as" directly to their left, in which case emit lead-in + block cleanly.
    let mut lines: Vec<Vec<LayoutItem>> = Vec::new();
    for item in items {
        if item.is_isolated() {
            lines.push(vec![item]);
            continue;
        }
        let mut placed = false;
        for line in lines.iter_mut().rev() {
            if line.iter().any(|it| it.is_isolated()) {
                continue;
            }
            let line_box = merge_line_bbox(line);
            if item.bbox().vertical_overlap_ratio(&line_box) >= 0.42 {
                line.push(item.clone());
                placed = true;
                break;
            }
        }
        if !placed {
            lines.push(vec![item]);
        }
    }

    // Sort lines top-to-bottom by minimum y1, and items within each line left-to-right
    lines.sort_by(|a, b| {
        let ay = a.iter().map(|i| i.bbox().y1).fold(f32::INFINITY, f32::min);
        let by = b.iter().map(|i| i.bbox().y1).fold(f32::INFINITY, f32::min);
        ay.total_cmp(&by)
    });

    let mut output_lines: Vec<String> = Vec::new();
    for mut line in lines {
        line.sort_by(|a, b| a.bbox().x1.total_cmp(&b.bbox().x1));

        if line.len() == 1 {
            if let LayoutItem::IsolatedMath { latex, .. } = &line[0] {
                output_lines.push(format!("\n$$\n{}\n$$\n", latex));
                continue;
            }
        }

        let mut buf = String::new();
        for item in line {
            let chunk = match item {
                LayoutItem::Text { content, .. } => content,
                LayoutItem::InlineMath { latex, .. } => format!("${}$", latex),
                LayoutItem::IsolatedMath { latex, .. } => format!("\n$$\n{}\n$$\n", latex),
            };

            if buf.is_empty() {
                buf.push_str(&chunk);
            } else if starts_with_closing_punct(&chunk) {
                buf.push_str(&chunk);
            } else {
                if !buf.ends_with(' ') && !buf.ends_with('\n') && !chunk.starts_with('\n') {
                    buf.push(' ');
                }
                buf.push_str(&chunk);
            }
        }
        output_lines.push(buf.trim().to_string());
    }

    // Collapse excessive blank lines while preserving block math spacing
    let joined = output_lines.join("\n");
    let multi_nl = Regex::new(r"\n{3,}").unwrap();
    multi_nl.replace_all(joined.trim(), "\n\n").to_string()
}

fn merge_line_bbox(line: &[LayoutItem]) -> Box2D {
    let mut x1 = f32::INFINITY;
    let mut y1 = f32::INFINITY;
    let mut x2 = f32::NEG_INFINITY;
    let mut y2 = f32::NEG_INFINITY;
    for item in line {
        let b = item.bbox();
        x1 = x1.min(b.x1);
        y1 = y1.min(b.y1);
        x2 = x2.max(b.x2);
        y2 = y2.max(b.y2);
    }
    Box2D { x1, y1, x2, y2 }
}

fn starts_with_closing_punct(s: &str) -> bool {
    matches!(
        s.chars().next(),
        Some(',' | '.' | ';' | ':' | '?' | '!' | ')' | ']')
    )
}

/// Strips noisy LaTeX artifacts (`\nonumber`, `\boldmath`, excessive `\!`,
/// outer `$` delimiters, stray `\left.`/`\right.` mismatches) and normalizes whitespace.
pub fn clean_latex(raw: &str) -> String {
    let mut s = raw.trim().to_string();

    // Strip surrounding $$ or $ if model emitted them
    if s.starts_with("$$") && s.ends_with("$$") && s.len() >= 4 {
        s = s[2..s.len() - 2].trim().to_string();
    } else if s.starts_with('$') && s.ends_with('$') && s.len() >= 2 {
        s = s[1..s.len() - 1].trim().to_string();
    }

    // Remove noisy macros
    for noise in &[
        r"\nonumber",
        r"\notag",
        r"\boldmath",
        r"\displaystyle",
        r"\textstyle",
        r"\!",
    ] {
        s = s.replace(noise, "");
    }

    // Normalize common spacing macros & whitespace
    let multi_space = Regex::new(r"[ \t]{2,}").unwrap();
    s = multi_space.replace_all(&s, " ").to_string();

    // Balance stray \left / \right mismatches
    let left_count = s.matches(r"\left").count();
    let right_count = s.matches(r"\right").count();
    if left_count > right_count {
        for _ in 0..(left_count - right_count) {
            s.push_str(r" \right.");
        }
    } else if right_count > left_count {
        // Strip unmatched trailing \right. first, otherwise prepend \left.
        let mut diff = right_count - left_count;
        while diff > 0 && s.ends_with(r"\right.") {
            s = s.trim_end_matches(r"\right.").trim().to_string();
            diff -= 1;
        }
        for _ in 0..diff {
            s = format!(r"\left. {}", s);
        }
    }

    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_latex_strips_noise_and_balances_delimiters() {
        let raw = r"$$ \displaystyle \boldmath f(x) = \left\{ \begin{cases} x; & 0 < x \le 1 \\ 2 - x; & 1 \le x \le 2 \\ 0; & \text{Otherwise} \end{cases} \nonumber \!\! $$";
        let cleaned = clean_latex(raw);
        assert!(!cleaned.contains(r"\nonumber"));
        assert!(!cleaned.contains(r"\boldmath"));
        assert!(!cleaned.contains(r"\!"));
        assert_eq!(cleaned.matches(r"\left").count(), cleaned.matches(r"\right").count());
    }

    #[test]
    fn test_stitch_inline_and_block() {
        let text = vec![
            TextSpan {
                bbox: Box2D { x1: 0.0, y1: 10.0, x2: 30.0, y2: 30.0 },
                text: "If".into(),
                confidence: 0.99,
            },
            TextSpan {
                bbox: Box2D { x1: 140.0, y1: 10.0, x2: 360.0, y2: 30.0 },
                text: "is cumulative distribution function".into(),
                confidence: 0.99,
            },
        ];
        let fboxes = vec![FormulaBox {
            bbox: Box2D { x1: 35.0, y1: 10.0, x2: 135.0, y2: 30.0 },
            kind: FormulaKind::Inline,
            score: 0.95,
        }];
        let flatex = vec!["F(X = x)".to_string()];
        let md = stitch_markdown(&text, &fboxes, &flatex);
        assert_eq!(md, "If $F(X = x)$ is cumulative distribution function");
    }
}
