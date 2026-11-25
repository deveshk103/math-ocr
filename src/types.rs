use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SnipMode {
    /// Alt + Q: Mixed English prose + inline/block math -> Markdown + LaTeX
    Mixed,
    /// Alt + M: Pure Math -> skips MFD & text OCR, outputs raw LaTeX
    Math,
    /// Alt + T: Pure Text -> outputs plain text
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FormulaKind {
    Inline,
    Isolated,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Box2D {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl Box2D {
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn iou(&self, other: &Box2D) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 1e-6 {
            0.0
        } else {
            inter / union
        }
    }

    pub fn vertical_overlap_ratio(&self, other: &Box2D) -> f32 {
        let iy1 = self.y1.max(other.y1);
        let iy2 = self.y2.min(other.y2);
        let inter = (iy2 - iy1).max(0.0);
        let min_h = self.height().min(other.height()).max(1.0);
        inter / min_h
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormulaBox {
    pub bbox: Box2D,
    pub kind: FormulaKind,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSpan {
    pub bbox: Box2D,
    pub text: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetrics {
    pub stage: String,
    pub elapsed_ms: u64,
    pub peak_working_set_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfdOutput {
    pub boxes: Vec<FormulaBox>,
    pub metrics: StageMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextOutput {
    pub spans: Vec<TextSpan>,
    pub metrics: StageMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfrOutput {
    pub formulas: Vec<String>,
    pub metrics: StageMetrics,
}
