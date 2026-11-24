//! Delta-debug minimiser for differential test failures.
//!
//! Phase 1: binary search on `Knobs.ops` to find the smallest program where the
//! difference persists (the generator is deterministic per seed).
//!
//! Phase 2: NOP individual instructions in the minimised program (replace with
//! `s_nop 0`, 0xBF800000, 4 bytes; 8-byte instructions become two s_nop). Skip the
//! prologue (v1 = v0 << 2) and s_endpgm. After each NOP, check that the program still
//! translates (structural validity) and that the difference persists.

use super::gen;
use xcl::gpu::rdna2::disassemble;
use xcl::gpu::recomp::{translate, Options};

const S_NOP: [u8; 4] = 0xBF80_0000u32.to_le_bytes();

/// Phase 1: binary search on ops count.
pub fn minimise_ops(base: &gen::Knobs, check: &dyn Fn(&gen::Knobs) -> bool) -> gen::Knobs {
    if !check(base) { return base.clone(); }
    let min_ops = (base.loops as usize * base.loop_depth as usize * 2).max(1);
    let mut lo = min_ops;
    let mut hi = base.ops;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let k = gen::Knobs { ops: mid.max(min_ops), ..base.clone() };
        if check(&k) { hi = mid; } else { lo = mid + 1; }
    }
    gen::Knobs { ops: lo.max(min_ops), ..base.clone() }
}

/// Phase 2: NOP individual instructions, back to front, keeping those whose removal
/// preserves both translatability and the detected difference.
pub fn minimise_instructions(
    code: &[u8], opts: &Options, check_code: &dyn Fn(&[u8]) -> bool,
) -> Vec<u8> {
    let lines = disassemble(code, 0, true);
    if lines.len() < 3 { return code.to_vec(); } // prologue + endpgm at minimum
    let mut best = code.to_vec();
    // Try NOPing each instruction except the first (prologue) and last (s_endpgm)
    let mut i = lines.len() - 2; // skip s_endpgm
    while i >= 1 { // skip prologue at index 0
        let l = &lines[i];
        if keeps_structure(&l.text) {
            if i == 1 { break; }
            i -= 1;
            continue;
        }
        let start = l.pc as usize;
        let len = l.bytes.len();
        // Build a patched copy with this instruction NOPed
        let mut patched = best.clone();
        for off in (start..start + len).step_by(4) {
            if off + 4 <= patched.len() {
                patched[off..off + 4].copy_from_slice(&S_NOP);
            }
        }
        // Check structural validity: does it still translate?
        if let Ok(t) = translate(&patched, opts) {
            if t.complete() && check_code(&patched) {
                best = patched;
            }
        }
        if i == 1 { break; }
        i -= 1;
    }
    best
}

/// Branches, exec masks, the compares that feed them and the loop counters (s80 and up) are
/// kept: NOPing one of them changes the program's shape (an endless loop, a branch on a stale
/// mask) and turns a translation difference into a different program.
fn keeps_structure(text: &str) -> bool {
    let t = text.trim();
    t.starts_with("s_cbranch") || t.starts_with("s_branch") || t.contains("saveexec")
        || t.contains("exec") || t.starts_with("v_cmp") || t.starts_with("s_cmp")
        || t.split(|c: char| !c.is_ascii_alphanumeric()).any(|w| {
            w.strip_prefix('s').and_then(|n| n.parse::<u32>().ok()).is_some_and(|n| (80..96).contains(&n))
        })
}

/// Print a minimised program's disassembly to stderr for use as a fixed regression test.
pub fn print_minimised(knobs: &gen::Knobs, code: &[u8]) {
    eprintln!("--- minimised program (seed={}, ops={}) ---", knobs.seed, knobs.ops);
    for l in disassemble(code, 0, true) {
        eprintln!("  {:#06x}: {}", l.pc, l.text);
    }
    eprintln!("---");
}
