use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{ERROR_CLASS_ALREADY_EXISTS, GetLastError, HWND};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Controls::*;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

use crate::shaders::{LaunchPreparation, prepare_for_launch};
use crate::util::wide;

struct Window { hwnd: HWND, label: HWND, timer: HWND, bar: HWND }

impl Drop for Window {
    fn drop(&mut self) { unsafe { if IsWindow(self.hwnd) != 0 { DestroyWindow(self.hwnd); } } }
}

impl Window {
    fn open() -> Result<Self, String> {
        unsafe {
            let instance = GetModuleHandleW(null());
            let class = wide("XCLShaderPreparation");
            let wc = WNDCLASSW {
                hInstance: instance, lpfnWndProc: Some(DefWindowProcW),
                hCursor: LoadCursorW(null_mut(), IDC_ARROW),
                hbrBackground: (windows_sys::Win32::Graphics::Gdi::COLOR_WINDOW + 1) as _,
                lpszClassName: class.as_ptr(), ..std::mem::zeroed()
            };
            if RegisterClassW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS {
                return Err(std::io::Error::last_os_error().to_string());
            }
            let init = INITCOMMONCONTROLSEX { dwSize: std::mem::size_of::<INITCOMMONCONTROLSEX>() as u32, dwICC: ICC_PROGRESS_CLASS };
            if InitCommonControlsEx(&init) == 0 { return Err("Could not initialize the progress bar".into()); }
            let hwnd = CreateWindowExW(0, class.as_ptr(), wide("XCL – Preparing shaders").as_ptr(),
                WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT, 510, 155,
                null_mut(), null_mut(), instance, null());
            if hwnd.is_null() { return Err(std::io::Error::last_os_error().to_string()); }
            let label = CreateWindowExW(0, wide("STATIC").as_ptr(), wide("Reading shader records…").as_ptr(),
                WS_CHILD | WS_VISIBLE, 16, 16, 420, 25, hwnd, null_mut(), instance, null());
            let timer = CreateWindowExW(0, wide("STATIC").as_ptr(), null(),
                WS_CHILD | WS_VISIBLE, 16, 47, 465, 23, hwnd, null_mut(), instance, null());
            let bar = CreateWindowExW(0, PROGRESS_CLASSW, null(), WS_CHILD | WS_VISIBLE,
                16, 79, 465, 18, hwnd, null_mut(), instance, null());
            let window = Self { hwnd, label, timer, bar };
            if label.is_null() || timer.is_null() || bar.is_null() { return Err(std::io::Error::last_os_error().to_string()); }
            ShowWindow(hwnd, SW_SHOWNORMAL);
            Ok(window)
        }
    }

    fn pump(&self) -> bool {
        unsafe {
            let mut msg = std::mem::zeroed();
            while PeekMessageW(&mut msg, null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            IsWindow(self.hwnd) != 0
        }
    }

    fn update(&self, done: usize, total: usize, ready: bool) {
        let percent = percent(done, total, ready);
        unsafe {
            SetWindowTextW(self.label, wide(&format!("Preparing shaders: {percent}% ({done}/{total} records)")).as_ptr());
            SendMessageW(self.bar, PBM_SETPOS, percent, 0);
        }
    }
}

#[derive(Default)]
struct StallClock {
    done: usize,
    advanced_at: Duration,
    last_stall_ended: Option<Duration>,
}

impl StallClock {
    fn sample(&mut self, done: usize, now: Duration) -> (u64, Option<u64>) {
        if done != self.done {
            if now - self.advanced_at >= Duration::from_secs(5) {
                self.last_stall_ended = Some(now);
            }
            self.done = done;
            self.advanced_at = now;
        }
        ((now - self.advanced_at).as_secs(), self.last_stall_ended.map(|end| (now - end).as_secs()))
    }
}

fn percent(done: usize, total: usize, ready: bool) -> usize {
    if ready { 100 } else { done.saturating_mul(100).checked_div(total).unwrap_or(0).min(99) }
}

pub(super) fn prepare(root: &Path, adapter: &str, menu: bool) -> Result<(), String> {
    let pack = root.join("shaders.rpf");
    let loose = root.join("shaders");
    if !pack.try_exists().map_err(|e| e.to_string())? && !loose.is_dir() { return Ok(()); }
    let window = Window::open()?;
    let progress = LaunchPreparation { adapter, done: 0.into(), total: 0.into(), cancel: false.into() };
    let start = Instant::now();
    let mut clock = StallClock::default();
    let mut last_times = None;
    let mut last = None;
    let result = wait_for_worker(&progress, || prepare_for_launch(root, &progress), || {
        if !window.pump() { return false; }
        let total = progress.total.load(Ordering::Relaxed);
        let done = progress.done.load(Ordering::Relaxed);
        let times = clock.sample(done, start.elapsed());
        if last_times != Some(times) {
            let mut text = format!("No shader progress for {} s", times.0);
            if let Some(ago) = times.1 { text.push_str(&format!("   |   Last stall ended {ago} s ago")); }
            unsafe { SetWindowTextW(window.timer, wide(&text).as_ptr()); }
            last_times = Some(times);
        }
        if total > 0 && last != Some(done) {
            window.update(done, total, false);
            last = Some(done);
        }
        true
    });
    match &result {
        Ok(()) => {
            let total = progress.total.load(Ordering::Relaxed);
            window.update(total, total, true);
            unsafe {
                windows_sys::Win32::Graphics::Gdi::UpdateWindow(window.label);
                windows_sys::Win32::Graphics::Gdi::UpdateWindow(window.bar);
            }
        }
        Err(message) if !progress.cancel.load(Ordering::Relaxed) && menu => unsafe {
            MessageBoxW(window.hwnd, wide(message).as_ptr(), wide("XCL – Shader preparation failed").as_ptr(), MB_OK | MB_ICONERROR);
        },
        _ => {}
    }
    result
}

fn wait_for_worker(progress: &LaunchPreparation, work: impl FnOnce() -> Result<(), String> + Send,
    mut pump: impl FnMut() -> bool) -> Result<(), String> {
    std::thread::scope(|scope| {
        let worker = scope.spawn(work);
        while !worker.is_finished() {
            if !pump() { progress.cancel.store(true, Ordering::Relaxed); }
            std::thread::sleep(Duration::from_millis(80));
        }
        let result = worker.join().map_err(|_| "Shader preparation worker stopped unexpectedly")?;
        if !pump() { progress.cancel.store(true, Ordering::Relaxed); }
        if progress.cancel.load(Ordering::Relaxed) { return Err("Shader preparation cancelled".into()); }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_waits_for_success_and_propagates_failure_or_close() {
        let progress = LaunchPreparation { adapter: "auto", done: 0.into(), total: 0.into(), cancel: false.into() };
        let called = std::sync::atomic::AtomicBool::new(false);
        wait_for_worker(&progress, || {
            while !called.load(Ordering::Acquire) { std::thread::yield_now(); }
            Ok(())
        }, || { called.store(true, Ordering::Release); true }).unwrap();
        assert_eq!(wait_for_worker(&progress, || Err("bad shader".into()), || true).unwrap_err(), "bad shader");
        let cancelled = wait_for_worker(&progress, || {
            while !progress.cancel.load(Ordering::Relaxed) { std::thread::yield_now(); }
            Ok(())
        }, || false);
        assert_eq!(cancelled.unwrap_err(), "Shader preparation cancelled");
    }

    #[test]
    fn clock_advances_during_a_stall_and_remembers_recovery() {
        let mut clock = StallClock::default();
        let s = Duration::from_secs;
        assert_eq!(clock.sample(0, s(1)), (1, None));
        assert_eq!(clock.sample(0, s(5)), (5, None));
        assert_eq!(clock.sample(2, s(6)), (0, Some(0)));
        assert_eq!(clock.sample(2, s(8)), (2, Some(2)));
        assert_eq!(clock.sample(3, s(9)), (0, Some(3)));
        assert_eq!(clock.sample(3, s(15)), (6, Some(9)));
        assert_eq!(clock.sample(4, s(16)), (0, Some(0)));
    }

    #[test]
    fn only_a_successful_join_releases_the_last_percent() {
        assert_eq!(percent(0, 0, false), 0);
        assert_eq!(percent(1, 4, false), 25);
        assert_eq!(percent(4, 4, false), 99);
        assert_eq!(percent(4, 4, true), 100);
    }
}
