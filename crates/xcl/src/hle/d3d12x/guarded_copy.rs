//! A copy out of the title's memory that survives the title taking the memory away under it.
//!
//! Textures are read from the title's memory on XCL's threads while the title's own threads
//! decommit and unmap pages (a retail game moves its streaming pool's pages about as it
//! loads). `ReadProcessMemory` on the process itself tolerates that and is what other hosts
//! use (`title_textures::copy_committed_region`), but Wine's server suspends the whole process
//! for each call, so on a Mac the bytes were copied directly – and a page that went mid-copy
//! ended the process (an access violation in `memcpy`, once in about ten loading runs).
//! Here the copy is a leaf routine of plain loads and stores, and a vectored handler sends a
//! fault inside it to its end: it returns how many bytes it did not copy, and the caller goes
//! on behind the page that went. A fault there costs one exception; a copy without one costs
//! nothing more than the loop.

use std::sync::Once;

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};

// rcx: destination, rdx: source, r8: bytes; returns in rax the bytes not copied. No stack, no
// calls: wherever a load faults, `xcl_guarded_copy_end` is a valid place to go on, and r8 is
// the count at the start of the step that faulted (a step stores nothing before all its loads).
std::arch::global_asm!(
    r#"
    .globl xcl_guarded_copy
xcl_guarded_copy:
    cmp r8, 32
    jb 3f
2:
    mov rax, [rdx]
    mov r9, [rdx + 8]
    mov r10, [rdx + 16]
    mov r11, [rdx + 24]
    mov [rcx], rax
    mov [rcx + 8], r9
    mov [rcx + 16], r10
    mov [rcx + 24], r11
    add rdx, 32
    add rcx, 32
    sub r8, 32
    cmp r8, 32
    jae 2b
3:
    test r8, r8
    jz 4f
    mov al, [rdx]
    mov [rcx], al
    inc rdx
    inc rcx
    dec r8
    jmp 3b
4:
    .globl xcl_guarded_copy_end
xcl_guarded_copy_end:
    mov rax, r8
    ret
"#
);

extern "C" {
    fn xcl_guarded_copy(dst: *mut u8, src: *const u8, len: usize) -> usize;
    fn xcl_guarded_copy_end();
}

const PAGE: usize = 0x1000;

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS {
        return 0;
    }
    let ctx = &mut *(*info).ContextRecord;
    let (start, end) = (xcl_guarded_copy as usize, xcl_guarded_copy_end as usize);
    if !(start..end).contains(&(ctx.Rip as usize)) {
        return 0;
    }
    ctx.Rip = end as u64;
    -1
}

/// `len` bytes from `src` to `dst`; the 4 KiB pages of the source that cannot be read (now, or
/// no longer while it runs) are left out: `dst` keeps what it had there. True: all were copied.
///
/// # Safety
/// `dst` is writable for `len` bytes and does not overlap the source.
pub(crate) unsafe fn copy(src: usize, dst: *mut u8, len: usize) -> bool {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        AddVectoredExceptionHandler(1, Some(handler));
    });
    let (mut done, mut all) = (0usize, true);
    while done < len {
        let left = xcl_guarded_copy(dst.add(done), (src + done) as *const u8, len - done);
        done = len - left;
        if left > 0 {
            all = false;
            // The step that faulted started at `src + done`. A 32-byte step that reaches into
            // the page that went starts in the page before it: those last bytes are still
            // there, byte by byte. Then on behind the page that went.
            let next = ((src + done) & !(PAGE - 1)) + PAGE;
            let tail = (next - (src + done)).min(len - done);
            if tail < 32 && xcl_guarded_copy(dst.add(done), (src + done) as *const u8, tail) == 0 {
                done += tail;
                continue;
            }
            done = (next - src).min(len);
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

    /// What can be read is copied byte for byte – also at odd offsets and lengths – and a page
    /// with no memory is left out without a fault reaching the caller.
    #[test]
    fn pages_without_memory_are_left_out() {
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            for page in [0usize, 1, 3] {
                VirtualAlloc((base + page * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE);
            }
            let pattern = |i: usize| (i * 131 + 7) as u8;
            for page in [0usize, 1, 3] {
                for i in page * PAGE..(page + 1) * PAGE {
                    *((base + i) as *mut u8) = pattern(i);
                }
            }
            // all there: every offset / length shape of the loop
            for (from, len) in [(0usize, 2 * PAGE), (3, 61), (5, 32), (7, 31), (PAGE - 9, 18), (1, 0)] {
                let mut out = vec![0xEEu8; len];
                assert!(copy(base + from, out.as_mut_ptr(), len), "{from}+{len}");
                assert!(out.iter().enumerate().all(|(i, &b)| b == pattern(from + i)), "{from}+{len}");
            }
            // page 2 has no memory: left as it was, the pages around it copied
            let (from, len) = (PAGE - 100, 2 * PAGE + 300);
            let mut out = vec![0xEEu8; len];
            assert!(!copy(base + from, out.as_mut_ptr(), len));
            for (i, &b) in out.iter().enumerate() {
                let at = from + i;
                if (2 * PAGE..3 * PAGE).contains(&at) {
                    assert_eq!(b, 0xEE, "byte {at:#x} of the page without memory");
                } else {
                    assert_eq!(b, pattern(at), "byte {at:#x}");
                }
            }
            // a page decommitted later is left out the same way
            VirtualFree((base + PAGE) as *mut c_void, PAGE, MEM_DECOMMIT);
            let mut out = vec![0xEEu8; 2 * PAGE];
            assert!(!copy(base, out.as_mut_ptr(), 2 * PAGE));
            assert!(out[..PAGE].iter().enumerate().all(|(i, &b)| b == pattern(i)) && out[PAGE..].iter().all(|&b| b == 0xEE));
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A page that goes while other threads copy it never ends the process.
    #[test]
    fn a_page_decommitted_during_copies_is_survived() {
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 16 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    let stop = stop.clone();
                    std::thread::spawn(move || {
                        let mut out = vec![0u8; 16 * PAGE];
                        let mut holes = 0u32;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            if !copy(base, out.as_mut_ptr(), 16 * PAGE) {
                                holes += 1;
                            }
                        }
                        holes
                    })
                })
                .collect();
            for round in 0..400usize {
                let page = base + (round % 16) * PAGE;
                VirtualFree(page as *mut c_void, PAGE, MEM_DECOMMIT);
                std::thread::yield_now();
                VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE);
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            let holes: u32 = readers.into_iter().map(|h| h.join().unwrap()).sum();
            println!("copies that met a page without memory: {holes}");
            VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
