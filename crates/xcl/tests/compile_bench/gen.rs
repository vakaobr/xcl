//! Generated RDNA2 programs for compile-time measurements: no game data, the same program for
//! the same knobs and seed. The code is meant to be translated and compiled, not run: register
//! values and addresses are not set up to compute anything.
//!
//! Register use: v0 is the lane id (read only), v1 the lane's byte offset for memory, v2.. the
//! `vgprs` pool. s[2:3] is the memory base, s4.. the `sgprs` pool, s[64 + 2d] the EXEC saved by
//! an `if` at depth d, s[80 + d] the counter of a loop at depth d.

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::Options;

#[derive(Debug, Clone, PartialEq)]
pub struct Knobs {
    pub seed: u64,
    /// operations (ALU and memory instructions); the control flow comes on top
    pub ops: usize,
    /// how deep divergent `if`s (v_cmp → s_and_saveexec → s_cbranch_execz) nest; 0: none
    pub if_depth: u32,
    /// percent of operation slots that open an `if` while `if_depth` allows one
    pub divergent_pct: u32,
    /// uniform loops (counter in an SGPR, backward s_cbranch_scc1), each `loop_depth` deep
    pub loops: u32,
    pub loop_depth: u32,
    /// trips of every loop, 1..=64 (an inline constant)
    pub loop_trips: u32,
    /// wave64 (64-bit EXEC) or wave32 program
    pub wave64: bool,
    /// VGPRs the operations use, 2..=254; SGPRs, 1..=16
    pub vgprs: u32,
    pub sgprs: u32,
    /// percent of operations that are LDS reads / writes, and global or scalar memory accesses
    pub lds_pct: u32,
    pub mem_pct: u32,
}

impl Default for Knobs {
    fn default() -> Self {
        Knobs { seed: 1, ops: 200, if_depth: 2, divergent_pct: 10, loops: 0, loop_depth: 1, loop_trips: 4, wave64: true, vgprs: 16, sgprs: 8, lds_pct: 5, mem_pct: 10 }
    }
}

impl Knobs {
    /// The recompiler options the program is meant for (kernel mode, its wave size).
    pub fn options(&self) -> Options {
        Options { wave64: self.wave64, ..Options::default() }
    }
}

const VCC: u64 = 106;
const EXEC: u64 = 126;
const LIT: u64 = 255;
const V: u64 = 256;

fn inst(name: &str, enc: &str, fields: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    encode(name, enc, fields, lit).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// splitmix64: small, and the same sequence on every platform
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

struct Gen<'a> {
    k: &'a Knobs,
    rng: Rng,
}

impl Gen<'_> {
    fn vgpr(&mut self) -> u64 {
        2 + self.rng.below(self.k.vgprs as u64)
    }

    fn sgpr(&mut self) -> u64 {
        4 + self.rng.below(self.k.sgprs as u64)
    }

    fn op(&mut self) -> Vec<u8> {
        let r = self.rng.below(100) as u32;
        if r < self.k.mem_pct {
            return match self.rng.below(3) {
                0 => inst("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", self.vgpr()), ("ADDR", 1), ("SADDR", 2), ("OFFSET", 4 * self.rng.below(256))], None),
                1 => inst("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", self.vgpr()), ("SADDR", 2), ("OFFSET", 4 * self.rng.below(256))], None),
                _ => inst("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", self.sgpr()), ("SBASE", 1), ("OFFSET", 4 * self.rng.below(256))], None),
            };
        }
        if r < self.k.mem_pct + self.k.lds_pct {
            return if self.rng.below(2) == 0 {
                inst("DS_WRITE_B32", "ENC_DS", &[("ADDR", 1), ("DATA0", self.vgpr())], None)
            } else {
                inst("DS_READ_B32", "ENC_DS", &[("VDST", self.vgpr()), ("ADDR", 1)], None)
            };
        }
        // the rest: about one scalar operation for three vector ones, as in console programs
        if self.rng.below(4) == 0 {
            let name = ["S_ADD_U32", "S_XOR_B32", "S_MUL_I32"][self.rng.below(3) as usize];
            return inst(name, "ENC_SOP2", &[("SDST", self.sgpr()), ("SSRC0", self.sgpr()), ("SSRC1", self.sgpr())], None);
        }
        let (d, a, b, c) = (self.vgpr(), self.vgpr(), self.vgpr(), self.vgpr());
        match self.rng.below(6) {
            0 => inst("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)], None),
            1 => inst("V_XOR_B32", "ENC_VOP2", &[("VDST", d), ("SRC0", self.sgpr()), ("VSRC1", b)], None),
            2 => inst("V_MUL_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", LIT), ("VSRC1", b)], Some(0x3F80_0000 + (self.rng.below(1 << 16) as u32))),
            3 => inst("V_FMAC_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)], None),
            4 => inst("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", d), ("SRC0", V + a), ("SRC1", V + b)], None),
            _ => inst("V_FMA_F32", "ENC_VOP3", &[("VDST", d), ("SRC0", V + a), ("SRC1", V + b), ("SRC2", V + c)], None),
        }
    }

    /// `n` operations, some of them inside divergent `if`s from depth `depth` on.
    fn seq(&mut self, mut n: usize, depth: u32) -> Vec<u8> {
        let mut v = Vec::new();
        while n > 0 {
            if depth < self.k.if_depth && (self.rng.below(100) as u32) < self.k.divergent_pct {
                let inner = n.min(4 + self.rng.below(12) as usize);
                v.extend(self.divergent_if(inner, depth));
                n -= inner;
            } else {
                v.extend(self.op());
                n -= 1;
            }
        }
        v
    }

    fn divergent_if(&mut self, n: usize, depth: u32) -> Vec<u8> {
        let save = 64 + 2 * depth as u64;
        let (and_saveexec, or) = if self.k.wave64 { ("S_AND_SAVEEXEC_B64", "S_OR_B64") } else { ("S_AND_SAVEEXEC_B32", "S_OR_B32") };
        let body = self.seq(n, depth + 1);
        let mut v = inst("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", LIT), ("VSRC1", self.vgpr())], Some(self.rng.next() as u32));
        v.extend(inst(and_saveexec, "ENC_SOP1", &[("SDST", save), ("SSRC0", VCC)], None));
        v.extend(inst("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", (body.len() / 4) as u64)], None));
        v.extend(body);
        v.extend(inst(or, "ENC_SOP2", &[("SDST", EXEC), ("SSRC0", EXEC), ("SSRC1", save)], None));
        v
    }

    /// A loop `levels` deep: each level runs `per_level` operations, then the next level.
    fn nested_loop(&mut self, levels: u32, per_level: usize, depth: u32) -> Vec<u8> {
        let counter = 80 + depth as u64;
        let mut v = inst("S_MOV_B32", "ENC_SOP1", &[("SDST", counter), ("SSRC0", 128 + self.k.loop_trips as u64)], None);
        let head = v.len();
        v.extend(self.seq(per_level, 0));
        if levels > 1 {
            v.extend(self.nested_loop(levels - 1, per_level, depth + 1));
        }
        v.extend(inst("S_SUB_U32", "ENC_SOP2", &[("SDST", counter), ("SSRC0", counter), ("SSRC1", 129)], None));
        v.extend(inst("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", counter), ("SSRC1", 128)], None));
        // the offset counts dwords from the instruction after the branch
        let back = -(((v.len() - head) / 4 + 1) as i64);
        v.extend(inst("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", back as i16 as u16 as u64)], None));
        v
    }
}

/// The program for `k`, ending in s_endpgm. Half of the operations go into the loops when
/// there are any, spread evenly over their levels; the rest runs before, between and after them.
pub fn program(k: &Knobs) -> Vec<u8> {
    assert!((2..=254).contains(&k.vgprs) && (1..=16).contains(&k.sgprs), "{k:?}");
    assert!((1..=64).contains(&k.loop_trips) && k.loop_depth >= 1 && k.if_depth <= 8 && k.loop_depth <= 8, "{k:?}");
    assert!(k.lds_pct + k.mem_pct <= 100 && k.divergent_pct <= 100, "{k:?}");
    let mut g = Gen { k, rng: Rng(k.seed) };
    let mut v = inst("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)], None); // v1 = lane * 4
    let (loops, levels) = (k.loops as usize, k.loop_depth as usize);
    // every loop level runs at least one operation, so the program would get more than `ops`
    assert!(k.ops / 2 >= loops * levels, "too few operations: ops {} for {loops} loops × {levels} levels needs ops ≥ {}", k.ops, 2 * loops * levels);
    let per_level = (k.ops / 2).checked_div(loops * levels).unwrap_or(0);
    let outside = k.ops.saturating_sub(per_level * levels * loops);
    for i in 0..=loops {
        // the operations outside the loops, split into loops + 1 runs
        v.extend(g.seq(outside * (i + 1) / (loops + 1) - outside * i / (loops + 1), 0));
        if i < loops {
            v.extend(g.nested_loop(k.loop_depth, per_level, 0));
        }
    }
    v.extend(inst("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}
