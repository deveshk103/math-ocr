use crate::platform::capture_screen_region;
use crate::types::SnipMode;
use anyhow::Result;
use image::RgbImage;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct ScreenRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Displays a transparent layered fullscreen Win32 overlay (`WS_EX_TOPMOST | WS_EX_LAYERED`)
/// with a mode-colored rubber-band selection rectangle (Green=Mixed, Red=Math, Blue=Text).
/// Returns `Ok(None)` if canceled via `Esc` or right-click.
/// Waits 50 ms after hiding/destroying the overlay before grabbing pixels with `BitBlt`.
pub fn select_and_capture(mode: SnipMode) -> Result<Option<RgbImage>> {
    let Some(rect) = run_selection_overlay(mode)? else {
        return Ok(None);
    };
    if rect.width < 8 || rect.height < 8 {
        return Ok(None);
    }
    std::thread::sleep(Duration::from_millis(50));
    let img = capture_screen_region(rect.x, rect.y, rect.width, rect.height)?;
    Ok(Some(img))
}

#[cfg(windows)]
fn run_selection_overlay(mode: SnipMode) -> Result<Option<ScreenRect>> {
    use std::cell::RefCell;
    use windows::core::{w, PCWSTR};
    use windows::Win32::Foundation::{COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
    use windows::Win32::Graphics::Gdi::{
        BeginPaint, CreatePen, CreateSolidBrush, DeleteObject, EndPaint, FillRect, FrameRect,
        InvalidateRect, Rectangle, SelectObject, HBRUSH, PAINTSTRUCT, PS_SOLID,
    };
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        ReleaseCapture, SetCapture, VK_ESCAPE,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetMessageW,
        GetSystemMetrics, LoadCursorW, PostQuitMessage, RegisterClassW, SetLayeredWindowAttributes,
        ShowWindow, TranslateMessage, CS_HREDRAW, CS_VREDRAW, IDC_CROSS, LWA_ALPHA, MSG,
        SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SW_HIDE,
        SW_SHOW, WM_KEYDOWN, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE, WM_PAINT,
        WM_RBUTTONDOWN, WNDCLASSW, WS_EX_LAYERED, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_POPUP,
    };

    #[derive(Default)]
    struct OverlayState {
        dragging: bool,
        start: POINT,
        current: POINT,
        result: Option<ScreenRect>,
        cancelled: bool,
        vx: i32,
        vy: i32,
        border_color: u32,
    }

    thread_local! {
        static STATE: RefCell<OverlayState> = RefCell::new(OverlayState::default());
    }

    // COLORREF is 0x00BBGGRR
    let border_color = match mode {
        SnipMode::Mixed => 0x0032C850, // Green
        SnipMode::Math => 0x003838E8,  // Red
        SnipMode::Text => 0x00E87828,  // Blue
    };

    unsafe extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        match msg {
            WM_KEYDOWN => {
                if wparam.0 as u16 == VK_ESCAPE.0 {
                    STATE.with(|s| s.borrow_mut().cancelled = true);
                    let _ = ReleaseCapture();
                    PostQuitMessage(0);
                    return LRESULT(0);
                }
            }
            WM_RBUTTONDOWN => {
                STATE.with(|s| s.borrow_mut().cancelled = true);
                let _ = ReleaseCapture();
                PostQuitMessage(0);
                return LRESULT(0);
            }
            WM_LBUTTONDOWN => {
                let x = (lparam.0 & 0xFFFF) as i16 as i32;
                let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as i32;
                STATE.with(|s| {
                    let mut st = s.borrow_mut();
                    st.dragging = true;
                    st.start = POINT { x, y };
                    st.current = POINT { x, y };
                });
                SetCapture(hwnd);
                return LRESULT(0);
            }
            WM_MOUSEMOVE => {
                let x = (lparam.0 & 0xFFFF) as i16 as i32;
                let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as i32;
                let redraw = STATE.with(|s| {
                    let mut st = s.borrow_mut();
                    if st.dragging {
                        st.current = POINT { x, y };
                        true
                    } else {
                        false
                    }
                });
                if redraw {
                    let _ = InvalidateRect(Some(hwnd), None, false);
                }
                return LRESULT(0);
            }
            WM_LBUTTONUP => {
                let x = (lparam.0 & 0xFFFF) as i16 as i32;
                let y = ((lparam.0 >> 16) & 0xFFFF) as i16 as i32;
                STATE.with(|s| {
                    let mut st = s.borrow_mut();
                    if st.dragging {
                        st.dragging = false;
                        st.current = POINT { x, y };
                        let x1 = st.start.x.min(st.current.x);
                        let y1 = st.start.y.min(st.current.y);
                        let x2 = st.start.x.max(st.current.x);
                        let y2 = st.start.y.max(st.current.y);
                        st.result = Some(ScreenRect {
                            x: st.vx + x1,
                            y: st.vy + y1,
                            width: (x2 - x1).max(0) as u32,
                            height: (y2 - y1).max(0) as u32,
                        });
                    }
                });
                let _ = ReleaseCapture();
                PostQuitMessage(0);
                return LRESULT(0);
            }
            WM_PAINT => {
                let mut ps = PAINTSTRUCT::default();
                let hdc = BeginPaint(hwnd, &mut ps);
                let bg_brush = CreateSolidBrush(COLORREF(0x00181818));
                FillRect(hdc, &ps.rcPaint, bg_brush);
                let _ = DeleteObject(bg_brush.into());

                STATE.with(|s| {
                    let st = s.borrow();
                    if st.dragging {
                        let rc = RECT {
                            left: st.start.x.min(st.current.x),
                            top: st.start.y.min(st.current.y),
                            right: st.start.x.max(st.current.x),
                            bottom: st.start.y.max(st.current.y),
                        };
                        let fill = CreateSolidBrush(COLORREF(0x00FFFFFF));
                        let pen = CreatePen(PS_SOLID, 3, COLORREF(st.border_color));
                        let old_pen = SelectObject(hdc, pen.into());
                        let old_brush = SelectObject(hdc, fill.into());
                        let _ = Rectangle(hdc, rc.left, rc.top, rc.right, rc.bottom);
                        let border_brush = CreateSolidBrush(COLORREF(st.border_color));
                        FrameRect(hdc, &rc, border_brush);
                        SelectObject(hdc, old_pen);
                        SelectObject(hdc, old_brush);
                        let _ = DeleteObject(pen.into());
                        let _ = DeleteObject(fill.into());
                        let _ = DeleteObject(border_brush.into());
                    }
                });

                let _ = EndPaint(hwnd, &ps);
                return LRESULT(0);
            }
            _ => {}
        }
        DefWindowProcW(hwnd, msg, wparam, lparam)
    }

    unsafe {
        let vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
        let vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
        let vw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        let vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);

        STATE.with(|s| {
            *s.borrow_mut() = OverlayState {
                vx,
                vy,
                border_color,
                ..Default::default()
            };
        });

        let hinstance: HINSTANCE = GetModuleHandleW(None)?.into();
        let class_name: PCWSTR = w!("SnipOcrOverlayClass");

        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wnd_proc),
            hInstance: hinstance,
            hCursor: LoadCursorW(None, IDC_CROSS)?,
            hbrBackground: HBRUSH::default(),
            lpszClassName: class_name,
            ..Default::default()
        };
        let _ = RegisterClassW(&wc);

        let hwnd = CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TOOLWINDOW,
            class_name,
            w!("SnipOCR Overlay"),
            WS_POPUP,
            vx,
            vy,
            vw,
            vh,
            None,
            None,
            Some(hinstance),
            None,
        )?;

        // Semi-transparent veil (alpha = 72 / 255)
        SetLayeredWindowAttributes(hwnd, COLORREF(0), 72, LWA_ALPHA)?;
        let _ = ShowWindow(hwnd, SW_SHOW);

        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        let _ = ShowWindow(hwnd, SW_HIDE);
        let _ = DestroyWindow(hwnd);
    }

    let out = STATE.with(|s| {
        let st = s.borrow();
        if st.cancelled {
            None
        } else {
            st.result
        }
    });
    Ok(out)
}

#[cfg(not(windows))]
fn run_selection_overlay(_mode: SnipMode) -> Result<Option<ScreenRect>> {
    anyhow::bail!("Interactive overlay is only supported on Windows 11")
}
