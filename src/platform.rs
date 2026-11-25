use anyhow::{Context, Result};
use image::RgbImage;

/// Enables Per-Monitor V2 DPI awareness on Windows 11 so overlay coordinates
/// and screen capture pixels match 1:1 on 125%/150% scaled displays.
pub fn enable_per_monitor_dpi_awareness() {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::UI::HiDpi::{
            SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
        };
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
}

/// Returns the current process Peak Working Set Size in MiB via Win32 `K32GetProcessMemoryInfo`.
pub fn peak_working_set_mb() -> f64 {
    #[cfg(windows)]
    unsafe {
        use std::mem::size_of;
        use windows::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        use windows::Win32::System::Threading::GetCurrentProcess;

        let mut pmc = PROCESS_MEMORY_COUNTERS::default();
        let ok = K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut pmc,
            size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        );
        if ok.as_bool() {
            return (pmc.PeakWorkingSetSize as f64) / (1024.0 * 1024.0);
        }
        0.0
    }
    #[cfg(not(windows))]
    {
        0.0
    }
}

/// Returns the current process live Working Set Size in MiB via Win32 `K32GetProcessMemoryInfo`.
pub fn current_working_set_mb() -> f64 {
    #[cfg(windows)]
    unsafe {
        use std::mem::size_of;
        use windows::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        use windows::Win32::System::Threading::GetCurrentProcess;

        let mut pmc = PROCESS_MEMORY_COUNTERS::default();
        let ok = K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut pmc,
            size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        );
        if ok.as_bool() {
            return (pmc.WorkingSetSize as f64) / (1024.0 * 1024.0);
        }
        0.0
    }
    #[cfg(not(windows))]
    {
        0.0
    }
}

/// Captures a rectangular region `(x, y, width, height)` in physical virtual-screen pixels
/// using native Win32 GDI `BitBlt` (`SRCCOPY`).
pub fn capture_screen_region(x: i32, y: i32, width: u32, height: u32) -> Result<RgbImage> {
    #[cfg(windows)]
    unsafe {
        use anyhow::bail;
        use std::mem::size_of;
        use windows::Win32::Graphics::Gdi::{
            BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC,
            GetDIBits, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB,
            DIB_RGB_COLORS, SRCCOPY,
        };

        if width == 0 || height == 0 {
            bail!("Capture region has zero area ({width}x{height})");
        }

        let hdc_screen = GetDC(None);
        if hdc_screen.is_invalid() {
            bail!("GetDC(NULL) failed");
        }
        let hdc_mem = CreateCompatibleDC(Some(hdc_screen));
        let hbm = CreateCompatibleBitmap(hdc_screen, width as i32, height as i32);
        let old_obj = SelectObject(hdc_mem, hbm.into());

        let blt_res = BitBlt(
            hdc_mem,
            0,
            0,
            width as i32,
            height as i32,
            Some(hdc_screen),
            x,
            y,
            SRCCOPY,
        );

        let mut bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width as i32,
                biHeight: -(height as i32), // top-down DIB
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut bgra = vec![0u8; (width as usize) * (height as usize) * 4];
        let lines = GetDIBits(
            hdc_mem,
            hbm,
            0,
            height,
            Some(bgra.as_mut_ptr() as *mut _),
            &mut bmi,
            DIB_RGB_COLORS,
        );

        SelectObject(hdc_mem, old_obj);
        let _ = DeleteObject(hbm.into());
        let _ = DeleteDC(hdc_mem);
        ReleaseDC(None, hdc_screen);

        blt_res.context("BitBlt failed")?;
        if lines == 0 {
            bail!("GetDIBits returned 0 scanlines");
        }

        let mut rgb = vec![0u8; (width as usize) * (height as usize) * 3];
        for (i, chunk) in bgra.chunks_exact(4).enumerate() {
            rgb[i * 3] = chunk[2]; // R
            rgb[i * 3 + 1] = chunk[1]; // G
            rgb[i * 3 + 2] = chunk[0]; // B
        }
        RgbImage::from_raw(width, height, rgb)
            .context("Failed to construct RgbImage from captured pixels")
    }

    #[cfg(not(windows))]
    {
        let _ = (x, y, width, height);
        anyhow::bail!("Screen capture is only supported on Windows 11")
    }
}

/// Copies UTF-8 Markdown/LaTeX text directly to the Windows clipboard via `arboard`.
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut clipboard = arboard::Clipboard::new().context("Failed to open Windows clipboard")?;
    clipboard
        .set_text(text.to_string())
        .context("Failed to write text to clipboard")?;
    Ok(())
}
