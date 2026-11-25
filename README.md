# `snip_ocr` — Low-RAM Offline Windows 11 Snip-to-Markdown/LaTeX OCR Tool

A native Rust desktop snipping tool for **Windows 11 (x86_64, CPU-only)** that captures any selected screen region and copies clean **Markdown + LaTeX** (`$...$` inline, `$$...$$` block/piecewise) directly to the Windows clipboard, engineered for low-RAM laptops (`<= 15 MB` idle, `<= 400 MB` peak).

## Architecture & Why Path B Was Chosen

Before implementing, we evaluated `oar-ocr`'s built-in **Indra OCR** (`IndraOcrBuilder`). Upstream `oar-ocr` benchmarks report **~3 GB peak RSS** even with 1 CPU worker, which exceeds the `400 MB` hard cap.

Instead, `snip_ocr` uses a **3-stage, self-spawning subprocess-isolated ONNX pipeline** so the Windows kernel reclaims 100% of each ONNX Runtime memory arena before the next model loads:

| Stage | Subprocess Flag | Model / Engine | ONNX Disk Size | Target Peak RAM |
| :--- | :--- | :--- | :--- | :--- |
| Parent Daemon | *(default)* | Win32 overlay, `global-hotkey`, `arboard`, `image` preprocessing | — | `4–10 MB` (`<= 15 MB`) |
| Stage 1 (MFD) | `--worker mfd` | [`breezedeus/pix2text-mfd-1.5`](https://huggingface.co/breezedeus/pix2text-mfd-1.5/tree/main) via `ort` | 80.3 MB | `< 180 MB` |
| Stage 2 (Text) | `--worker text` | [`oar-ocr`](https://github.com/GreatV/oar-ocr) PP-OCRv6 Small (CPU ONNX) | auto-cached | `< 160 MB` |
| Stage 3 (MFR) | `--worker mfr` | [`breezedeus/pix2text-mfr-1.5`](https://huggingface.co/breezedeus/pix2text-mfr-1.5/tree/main) (encoder dropped before decoder loads) | 87.5 + 32.0 MB | `< 350 MB` |

Every ONNX session is configured with `intra_threads(2)`, `inter_threads(1)`, and `with_memory_pattern(false)` (CPU EP only).

## Preprocessing & Layout Features

- **Per-Monitor V2 DPI Awareness:** Calls `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)` at startup so overlay coordinates and `BitBlt` pixels match 1:1 at 125%/150% scaling.
- **Dark Mode Auto-Inversion:** Automatically inverts dark-background captures (median grayscale luma `< 128`).
- **Diagonal Watermark Removal:** Configurable LUT (`--black-pt 45 --white-pt 175`) eliminates light-grey PDF watermarks (luma `185–235`) while keeping thin anti-aliased strokes crisp.
- **Fill-in-the-Blank Underline Cleanup:** Removes long isolated horizontal underlines (`width >= 45 px`, `thickness <= 5 px`, clear vertical bands above/below) so trailing `________` blanks never hallucinate symbols, while preserving minus signs and fraction bars.
- **Formula Crop Conditioning:** Adds `10–12 px` white border padding and `2x` Lanczos3 upscaling for tiny crops (`height < 64 px`).

## Build & Run on Windows 11 (x86_64)

### 1. Prerequisites
- Rust stable toolchain (`rustup target add x86_64-pc-windows-msvc`)
- Visual Studio C++ Build Tools (MSVC)

### 2. Compile Optimized Release Binary (LTO + Strip)
```powershell
cargo build --release
```
The stripped binary is produced at `target\release\snip_ocr.exe`.

### 3. One-Time Model Download & Tensor Inspection
```powershell
# Downloads pix2text-mfd-1.5 + pix2text-mfr-1.5 into %LOCALAPPDATA%\math-snip-ocr\models
.\target\release\snip_ocr.exe download

# Inspect actual ONNX input/output tensor names & shapes
.\target\release\snip_ocr.exe inspect-models
```
After this step (and the first PP-OCRv6 Small cache fetch by `oar-ocr`), inference runs **100% offline**.

### 4. Run Acceptance Benchmark & Debug Mode
```powershell
cargo run --release -- bench path\to\test_screenshot.png --debug
```
This outputs:
- `./debug/01_cleaned.png` (watermark & underline removed)
- `./debug/02_boxes.png` (blue = text, orange = inline math, red = isolated math)
- `./debug/03_masked_for_text.png` (formula boxes masked white before Stage 2)
- Final Markdown/LaTeX, per-stage latency (ms), and measured `PeakWorkingSetSize` (MB) via Win32 `K32GetProcessMemoryInfo` (exits non-zero if any stage exceeds `400 MB`).

### 5. Run Background Daemon Mode
```powershell
.\target\release\snip_ocr.exe
```
- **`Alt + Q`** — Mixed Text + Math (green box) → Markdown + LaTeX
- **`Alt + M`** — Pure Math (red box) → raw LaTeX
- **`Alt + T`** — Pure Text (blue box) → plain text
- **`Esc`** (or right-click) — Cancel active snip cleanly without wedging the hotkey lock
