//! Lane-by-lane comparison of two KernelResults.
//!
//! Integer and float results: exact bit match. Both paths run on the same GPU hardware,
//! so any float difference (NaN payloads, denormal handling) is a real translation bug.
//! SGPR slot 125 (KERNEL_SCC_SLOT) stores SCC; it is compared once via the scc field.

use xcl::gpu::recomp::runner::KernelResult;
use xcl::gpu::recomp::{KERNEL_LANES, KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_VGPRS};

/// One register or memory dword that differs between path A and path B.
pub struct Diff {
    /// "sgpr", "vgpr", "scc", or "mem"
    pub reg: &'static str,
    pub lane: usize,
    pub index: usize,
    pub a: u32,
    pub b: u32,
}

pub fn compare(a: &KernelResult, b: &KernelResult) -> Vec<Diff> {
    let mut diffs = Vec::new();
    for i in 0..KERNEL_SGPRS {
        if i == KERNEL_SCC_SLOT { continue; }
        if a.sgpr[i] != b.sgpr[i] {
            diffs.push(Diff { reg: "sgpr", lane: 0, index: i, a: a.sgpr[i], b: b.sgpr[i] });
        }
    }
    if a.scc != b.scc {
        diffs.push(Diff { reg: "scc", lane: 0, index: 0, a: a.scc, b: b.scc });
    }
    for lane in 0..KERNEL_LANES {
        if lane >= a.vgpr.len() || lane >= b.vgpr.len() { break; }
        for reg in 0..KERNEL_VGPRS {
            if reg >= a.vgpr[lane].len() || reg >= b.vgpr[lane].len() { break; }
            if a.vgpr[lane][reg] != b.vgpr[lane][reg] {
                diffs.push(Diff { reg: "vgpr", lane, index: reg, a: a.vgpr[lane][reg], b: b.vgpr[lane][reg] });
            }
        }
    }
    let mem_len = a.mem.len().min(b.mem.len());
    for i in 0..mem_len {
        if a.mem[i] != b.mem[i] {
            diffs.push(Diff { reg: "mem", lane: 0, index: i, a: a.mem[i], b: b.mem[i] });
        }
    }
    diffs
}

pub fn format_diffs(diffs: &[Diff], max: usize) -> String {
    let mut s = String::new();
    for (i, d) in diffs.iter().take(max).enumerate() {
        if i > 0 { s.push('\n'); }
        match d.reg {
            "vgpr" => s.push_str(&format!("  v{}[lane {}]: {:#010x} vs {:#010x}", d.index, d.lane, d.a, d.b)),
            "sgpr" => s.push_str(&format!("  s{}: {:#010x} vs {:#010x}", d.index, d.a, d.b)),
            "scc" => s.push_str(&format!("  scc: {} vs {}", d.a, d.b)),
            "mem" => s.push_str(&format!("  mem[{}]: {:#010x} vs {:#010x}", d.index, d.a, d.b)),
            _ => {}
        }
    }
    if diffs.len() > max {
        s.push_str(&format!("\n  ... and {} more", diffs.len() - max));
    }
    s
}
