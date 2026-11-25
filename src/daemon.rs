use crate::models::ModelPaths;
use crate::overlay::select_and_capture;
use crate::pipeline::{run_pipeline, PipelineOptions};
use crate::platform::{copy_to_clipboard, current_working_set_mb};
use crate::preprocess::PreprocessConfig;
use crate::types::SnipMode;
use anyhow::Result;
use global_hotkey::hotkey::{Code, HotKey, Modifiers};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

struct AtomicLockGuard(Arc<AtomicBool>);

impl Drop for AtomicLockGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Runs the lightweight background hotkey daemon (target idle Working Set: 4–10 MB, <= 15 MB).
/// Hotkeys:
/// - `Alt + Q`: Mixed Text + Math (green box) -> Markdown + LaTeX
/// - `Alt + M`: Pure Math (red box) -> raw LaTeX
/// - `Alt + T`: Pure Text (blue box) -> plain text
pub fn run_daemon(models: ModelPaths, preproc: PreprocessConfig) -> Result<()> {
    let manager = GlobalHotKeyManager::new()?;

    let hk_mixed = HotKey::new(Some(Modifiers::ALT), Code::KeyQ);
    let hk_math = HotKey::new(Some(Modifiers::ALT), Code::KeyM);
    let hk_text = HotKey::new(Some(Modifiers::ALT), Code::KeyT);

    manager.register(hk_mixed)?;
    manager.register(hk_math)?;
    manager.register(hk_text)?;

    let busy = Arc::new(AtomicBool::new(false));

    eprintln!(
        "[snip_ocr] Daemon ready (idle Working Set: {:.2} MB)",
        current_working_set_mb()
    );
    eprintln!("  Alt + Q : Mixed Text + Math -> Markdown + LaTeX");
    eprintln!("  Alt + M : Pure Math         -> Raw LaTeX");
    eprintln!("  Alt + T : Pure Text         -> Plain Text");
    eprintln!("  Esc     : Cancel active snip");

    let rx = GlobalHotKeyEvent::receiver();

    #[cfg(windows)]
    {
        use windows::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, GetMessageW, TranslateMessage, MSG,
        };
        let mut msg = MSG::default();
        unsafe {
            while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);

                while let Ok(ev) = rx.try_recv() {
                    if ev.state != HotKeyState::Pressed {
                        continue;
                    }
                    let mode = if ev.id == hk_mixed.id() {
                        Some(SnipMode::Mixed)
                    } else if ev.id == hk_math.id() {
                        Some(SnipMode::Math)
                    } else if ev.id == hk_text.id() {
                        Some(SnipMode::Text)
                    } else {
                        None
                    };

                    if let Some(mode) = mode {
                        handle_hotkey(mode, &busy, &models, preproc);
                    }
                }
            }
        }
    }

    #[cfg(not(windows))]
    {
        let _ = (rx, busy, models, preproc);
        anyhow::bail!("Daemon mode requires Windows 11 Win32 event loop");
    }

    Ok(())
}

fn handle_hotkey(
    mode: SnipMode,
    busy: &Arc<AtomicBool>,
    models: &ModelPaths,
    preproc: PreprocessConfig,
) {
    if busy
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    let _guard = AtomicLockGuard(Arc::clone(busy));

    match select_and_capture(mode) {
        Ok(Some(img)) => {
            let opts = PipelineOptions {
                mode,
                preproc,
                debug_dir: None,
            };
            match run_pipeline(&img, models, &opts) {
                Ok(res) => {
                    if let Err(err) = copy_to_clipboard(&res.markdown) {
                        eprintln!("[snip_ocr] Clipboard error: {err:#}");
                    } else {
                        eprintln!(
                            "[snip_ocr] Copied {} chars to clipboard (peak RAM: {:.1} MB)",
                            res.markdown.len(),
                            res.max_peak_mb()
                        );
                    }
                }
                Err(err) => eprintln!("[snip_ocr] Pipeline error: {err:#}"),
            }
        }
        Ok(None) => {
            eprintln!("[snip_ocr] Snip cancelled.");
        }
        Err(err) => {
            eprintln!("[snip_ocr] Capture error: {err:#}");
        }
    }
}
