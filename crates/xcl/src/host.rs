//! The host process: started by the launcher with the game's command line. Sets up
//! logging, the VFS, hooks and HLE, maps the title and jumps to its entry point.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::Boot;

pub static HOST_MODE: AtomicBool = AtomicBool::new(false);

/// The system's commit charge and its limit in bytes (RAM plus page files): what all programs'
/// memory, and all video memory, is counted against. (0, 0): not known.
pub fn system_commit() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    unsafe {
        let mut p: PERFORMANCE_INFORMATION = std::mem::zeroed();
        p.cb = std::mem::size_of_val(&p) as u32;
        if K32GetPerformanceInfo(&mut p, p.cb) == 0 {
            return (0, 0);
        }
        (p.CommitTotal as u64 * p.PageSize as u64, p.CommitLimit as u64 * p.PageSize as u64)
    }
}

/// The host's exit code when it ended because its launcher was gone.
const LAUNCHER_GONE: u32 = 0x7C1_0003;

/// The launcher, by a synchronize-only handle it opened on itself and this host inherited
/// (`XCL_PREPARATION_PARENT`; None: started some other way).
fn launcher_handle() -> Option<std::os::windows::io::OwnedHandle> {
    use std::os::windows::io::FromRawHandle;
    let h = std::env::var("XCL_PREPARATION_PARENT").ok().and_then(|s| s.parse::<usize>().ok()).filter(|&h| h != 0);
    std::env::remove_var("XCL_PREPARATION_PARENT");
    // SAFETY: the launcher put an inherited handle's value there for this process alone
    h.map(|h| unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(h as _) })
}

/// `gone` runs once `launcher` is signalled: the launcher ended (killed, or its console closed)
/// and nobody waits for this process, reads its output or summarizes the run any more.
fn watch_launcher(launcher: std::os::windows::io::OwnedHandle, gone: impl FnOnce() + Send + 'static) -> std::io::Result<std::thread::JoinHandle<()>> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
    use windows_sys::Win32::System::Threading::{WaitForSingleObject, INFINITE};
    std::thread::Builder::new().name("xcl-launcher-watch".into()).spawn(move || {
        if unsafe { WaitForSingleObject(launcher.as_raw_handle(), INFINITE) } == WAIT_OBJECT_0 {
            gone();
        }
    })
}

pub fn host_main(boot_path: &Path) -> i32 {
    let boot: Boot = match std::fs::read_to_string(boot_path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("xcl host: cannot read {}: {e}", boot_path.display());
            return 0x7C1_0001;
        }
    };
    crate::log::init(
        Some(&boot.logs_dir().join("xcl.log")),
        crate::log::Level::parse(&boot.config.log_level),
        boot.config.log_to_stderr,
    );
    crate::log::log_panics();
    crate::xlog_info!("XCL {} – host for '{}'", env!("CARGO_PKG_VERSION"), boot.title);
    crate::xlog_info!("game: {} (read-only folder: {})", boot.exe_path.display(), boot.game_root.display());
    crate::xlog_info!("XCL data: {}", boot.data_dir.display());
    crate::xlog_info!(
        "console: {} (title memory {} MiB, mode {})",
        if boot.config.is_series_s() { "Xbox Series S (lockhart)" } else { "Xbox Series X (anaconda)" },
        boot.config.title_memory_mb,
        boot.config.title_memory
    );

    if std::env::var_os("XCL_UMA_MB").is_none() && boot.config.uma_mb != 0 {
        std::env::set_var("XCL_UMA_MB", boot.config.uma_mb.to_string());
    }
    crate::state::set_boot(boot.clone());
    let launcher = launcher_handle();
    if let Err(e) = crate::hle::d3d12x::present::preparation::before_guest(&boot, launcher.as_ref()) {
        crate::xlog_error!("Shader preparation: {e}");
        eprintln!("xcl: {e}");
        return 2;
    }
    if let Some(launcher) = launcher {
        let _ = watch_launcher(launcher, || {
            let (shown, presented) = crate::hle::d3d12x::present::counters();
            crate::xlog_warn!("the launcher (xcl run) is gone – ending the title ({shown} frames shown, {presented} PresentX calls)");
            crate::report::write_final(None);
            crate::log::flush();
            crate::hle::d3d12x::real::end_title(LAUNCHER_GONE);
        });
    }
    let _ = std::thread::Builder::new().name("xcl-split-counts".into()).spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(10));
        crate::hle::d3d12x::real::log_dispatch_chain_counts();
    });
    // `XCL_ACTIVITY=1`: every 2 s what the title did since (a timeline of where it stops)
    if std::env::var_os("XCL_ACTIVITY").is_some() {
        let _ = std::thread::Builder::new().name("xcl-activity".into()).spawn(|| {
            let snap = || {
                let (p, e) = crate::hle::d3d12x::present::activity();
                let (d, x) = crate::hle::d3d12x::bind::activity();
                let f = crate::hle::d3d12x::frame::activity();
                let (r, b, fail) = crate::hle::dstorage::counters();
                [p, e, d, x, f[0], f[1], f[2], f[3], r, b >> 20, fail]
            };
            let mut last = snap();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = snap();
                let d: Vec<u64> = now.iter().zip(last).map(|(a, b)| a - b).collect();
                crate::xlog_info!(
                    "activity (2 s): {} presents, {} submits, {} draws, {} dispatches, {} frame-event waits ({} timed out), {} frame statistics ({} not flipped); DirectStorage {} requests, {} MB, {} failed",
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]
                );
                // where the memory is: the process's commit, video memory, and XCL's parts of it
                let private = crate::hle::kernelx::process_private_bytes();
                let (local, non_local) = crate::hle::d3d12x::real::video_memory().unwrap_or((0, 0));
                let (arenas, arenas_made) = crate::hle::d3d12x::vram::arena_bytes();
                let (pools, pools_mapped) = crate::hle::d3d12x::sparse::pool_bytes();
                let (pages, page_pools) = crate::hle::d3d12x::vram::page_bytes();
                let mb = |b: u64| b >> 20;
                crate::xlog_info!(
                    "memory: {} MB committed by the process; video memory {} MB local, {} MB non-local; target pages {} MB (pools {} MB); arenas {} MB in use, {} MB made; sparse pools {} MB ({} MB mapped)",
                    mb(private as u64),
                    mb(local),
                    mb(non_local),
                    mb(pages as u64),
                    mb(page_pools as u64),
                    mb(arenas as u64),
                    mb(arenas_made as u64),
                    mb(pools as u64),
                    mb(pools_mapped as u64)
                );
                last = now;
            }
        });
    }
    // `XCL_THREAD_DUMP_AT=s1,s2…`: where every thread is at those seconds (a stalled title)
    if let Ok(v) = std::env::var("XCL_THREAD_DUMP_AT") {
        let at: Vec<f64> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let _ = std::thread::Builder::new().name("xcl-thread-dump".into()).spawn(move || {
            let t0 = std::time::Instant::now();
            for s in at {
                let left = s - t0.elapsed().as_secs_f64();
                if left > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(left));
                }
                crate::eh::dump_threads();
            }
        });
    }
    // `XCL_RUN_SECONDS=n`: the title ends itself after n seconds – the GPU drained, the log and
    // report written (a process killed from outside with GPU work in flight can hang in the driver)
    if let Some(secs) = std::env::var("XCL_RUN_SECONDS").ok().and_then(|v| v.parse::<u64>().ok()) {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            let (shown, presented) = crate::hle::d3d12x::present::counters();
            crate::xlog_info!("XCL_RUN_SECONDS: {secs} s – ending the title ({shown} frames shown, {presented} PresentX calls)");
            crate::report::write_final(None);
            crate::log::flush();
            crate::hle::d3d12x::real::end_title(0);
        });
    }
    crate::stubs::init_db(&boot.config.stub_returns);
    crate::hle::init(&boot.config);
    crate::hle::kernelx::fine_timer();
    crate::loader::set_cache_dir(boot.cache_dir());
    let mut dirs = vec![boot.game_root.clone()];
    if let Some(p) = boot.exe_path.parent() {
        if p != boot.game_root {
            dirs.insert(0, p.to_path_buf());
        }
    }
    crate::resolver::set_search_dirs(dirs);

    let mounts: Vec<(String, String)> = boot.config.mounts.iter().map(|(k, v)| (k.clone(), boot.expand(v))).collect();
    // (the shader cache is XCL's too: on a drive a console mount names it must stay where it is)
    crate::vfs::init(crate::vfs::Vfs::new(&boot.game_root, &boot.data_dir, mounts).own_folder(&crate::gpu::recomp::dxc::cache_dir()));
    let _ = std::fs::create_dir_all(boot.data_dir.join("temp"));

    if let Err(e) = crate::vfs_hooks::install() {
        crate::xlog_error!("VFS: not all hooks installed: {e}");
    }
    if let Err(e) = crate::eh::install_pc_hook() {
        crate::xlog_error!("hook RtlPcToFileHeader: {e} – C++ exceptions in the game may not work");
    }
    crate::eh::install_crash_handler();
    crate::eh::start_probes();
    HOST_MODE.store(true, Ordering::Release);

    let main = match crate::loader::load_main(&boot.exe_path) {
        Ok(m) => m,
        Err(e) => {
            crate::xlog_error!("loading the game failed: {e}");
            crate::report::write_final(Some(format!("loader: {e}")));
            return 0x7C1_0002;
        }
    };
    unsafe { crate::hle::kernelx::adopt_title_hooks(&main) };
    crate::report::write_final(None);
    let code = crate::loader::run_main(&main);
    crate::xlog_info!("game entry point returned code {code}");
    crate::report::write_final(None);
    crate::log::flush();
    code as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_commit_is_known_and_under_its_limit() {
        let (commit, limit) = system_commit();
        assert!(commit > 0 && commit <= limit, "{commit} of {limit}");
    }

    /// The watch acts once the launcher's handle is signalled (an event stands for the process
    /// here: both are waited on the same way), and not before.
    #[test]
    fn the_title_ends_when_the_launcher_is_gone() {
        use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
        use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent};
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let event = unsafe { OwnedHandle::from_raw_handle(event) };
        let raw = event.as_raw_handle() as usize;
        let launcher = event.try_clone().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let watch = watch_launcher(launcher, move || tx.send(()).unwrap()).unwrap();
        let early = rx.recv_timeout(std::time::Duration::from_millis(300)).is_ok();
        unsafe { SetEvent(raw as _) };
        let ended = rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok();
        assert!(!early, "the title ended while the launcher was there");
        assert!(ended, "the launcher is gone and the title goes on");
        watch.join().unwrap();
    }
}
