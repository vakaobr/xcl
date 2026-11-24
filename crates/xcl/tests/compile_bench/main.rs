//! Compile-time benchmark on generated RDNA2 programs (issue #1269): no game data, so it can
//! run anywhere, CI included. This part checks the generator: what each knob does is counted
//! in the disassembly of the program, and every program the knobs allow translates completely.

mod gen;

use gen::{program, Knobs};
use xcl::gpu::rdna2::{disassemble, Line};
use xcl::gpu::recomp::{translate, Options};

fn mnemonic(l: &Line) -> &str {
    l.text.split_whitespace().next().unwrap_or("")
}

fn lines(code: &[u8]) -> Vec<Line> {
    let ls = disassemble(code, 0, true);
    assert!(ls.iter().all(|l| l.ok), "undecoded: {:?}", ls.iter().find(|l| !l.ok).map(|l| &l.text));
    assert_eq!(ls.iter().map(|l| l.bytes.len()).sum::<usize>(), code.len(), "the disassembly ends before the code");
    ls
}

/// Target of a SOPP branch: SIMM16 dwords from the next instruction.
fn target(l: &Line) -> u64 {
    let simm = u16::from_le_bytes([l.bytes[0], l.bytes[1]]) as i16 as i64;
    (l.pc as i64 + 4 + 4 * simm) as u64
}

/// Instructions that the control flow and the prologue add (the rest are the operations).
const CONTROL: &[&str] = &[
    "v_lshlrev_b32", "v_cmp_gt_u32", "s_and_saveexec_b64", "s_and_saveexec_b32", "s_cbranch_execz", "s_or_b64", "s_or_b32",
    "s_mov_b32", "s_sub_u32", "s_cmp_lg_u32", "s_cbranch_scc1", "s_endpgm",
];

#[derive(Debug, Default)]
struct Shape {
    ops: usize,
    ifs: usize,
    if_depth: u32,
    loops: usize,
    loop_depth: u32,
    lds: usize,
    mem: usize,
    max_vgpr: u32,
}

fn shape(code: &[u8]) -> Shape {
    let ls = lines(code);
    let at = |pc: u64| ls.iter().position(|l| l.pc == pc).unwrap_or_else(|| panic!("no instruction at {pc:#x}"));
    let mut s = Shape::default();
    let mut depth = 0u32;
    let mut loops = Vec::new();
    for (i, l) in ls.iter().enumerate() {
        let m = mnemonic(l);
        match m {
            "s_and_saveexec_b64" | "s_and_saveexec_b32" => {
                s.ifs += 1;
                depth += 1;
                s.if_depth = s.if_depth.max(depth);
            }
            "s_or_b64" | "s_or_b32" => depth -= 1,
            // an if's skip lands on its EXEC restore
            "s_cbranch_execz" => assert!(mnemonic(&ls[at(target(l))]).starts_with("s_or_b"), "execz at {:#x}", l.pc),
            // a loop's branch goes back to the instruction after its counter is set
            "s_cbranch_scc1" => {
                let head = at(target(l));
                assert!(head < i && mnemonic(&ls[head - 1]) == "s_mov_b32", "scc1 at {:#x}", l.pc);
                loops.push((ls[head].pc, l.pc));
            }
            _ => {}
        }
        if !CONTROL.contains(&m) {
            s.ops += 1;
            s.lds += m.starts_with("ds_") as usize;
            s.mem += (m.starts_with("global_") || m.starts_with("s_load")) as usize;
        }
        for tok in l.text.split(|c: char| !c.is_ascii_alphanumeric()) {
            if let Some(n) = tok.strip_prefix('v').and_then(|n| n.parse::<u32>().ok()) {
                s.max_vgpr = s.max_vgpr.max(n);
            }
        }
    }
    assert_eq!(depth, 0, "every if restores EXEC");
    s.loops = loops.len();
    s.loop_depth = loops.iter().map(|&(h, b)| loops.iter().filter(|&&(h2, b2)| h2 <= h && b <= b2).count() as u32).max().unwrap_or(0);
    s
}

#[test]
fn same_knobs_and_seed_give_the_same_program() {
    let k = Knobs { ops: 500, loops: 2, loop_depth: 2, ..Knobs::default() };
    assert_eq!(program(&k), program(&k));
    assert_ne!(program(&k), program(&Knobs { seed: 2, ..k }));
}

#[test]
fn operations_and_control_flow_follow_the_knobs() {
    for (ops, if_depth, loops, loop_depth) in [(200, 0, 0, 1), (400, 1, 1, 1), (600, 3, 2, 2), (2000, 8, 3, 4), (120, 2, 5, 3), (12, 0, 2, 3)] {
        let k = Knobs { ops, if_depth, loops, loop_depth, divergent_pct: 30, ..Knobs::default() };
        let s = shape(&program(&k));
        assert_eq!(s.ops, ops, "{k:?}");
        assert!(s.if_depth <= if_depth && (if_depth == 0) == (s.ifs == 0), "{k:?}: {s:?}");
        assert_eq!((s.loops, s.loop_depth), ((loops * loop_depth) as usize, if loops > 0 { loop_depth } else { 0 }), "{k:?}");
    }
    // always divergent: the ifs nest as deep as allowed
    for if_depth in [1, 4, 8] {
        let s = shape(&program(&Knobs { ops: 1000, if_depth, divergent_pct: 100, ..Knobs::default() }));
        assert_eq!(s.if_depth, if_depth);
    }
    let none = shape(&program(&Knobs { ops: 1000, divergent_pct: 0, ..Knobs::default() }));
    assert_eq!(none.ifs, 0);
}

#[test]
#[should_panic(expected = "too few operations")]
fn knobs_with_fewer_operations_than_loop_levels_are_refused() {
    // 2 × 3 levels need 12 operations: the program would get more than the 10 asked for
    program(&Knobs { ops: 10, loops: 2, loop_depth: 3, ..Knobs::default() });
}

#[test]
fn wave_size_picks_the_exec_instructions() {
    for wave64 in [true, false] {
        let code = program(&Knobs { ops: 500, if_depth: 3, divergent_pct: 50, wave64, ..Knobs::default() });
        let ms: Vec<String> = lines(&code).iter().map(|l| mnemonic(l).to_string()).collect();
        let (want, other) = if wave64 { ("_b64", "_b32") } else { ("_b32", "_b64") };
        assert!(ms.iter().any(|m| *m == format!("s_and_saveexec{want}")));
        assert!(!ms.iter().any(|m| *m == format!("s_and_saveexec{other}") || *m == format!("s_or{other}")), "wave64 {wave64}");
    }
}

#[test]
fn lds_and_memory_shares_follow_the_knobs() {
    let at = |lds_pct, mem_pct| shape(&program(&Knobs { ops: 2000, lds_pct, mem_pct, ..Knobs::default() }));
    let s = at(0, 0);
    assert_eq!((s.lds, s.mem), (0, 0));
    let s = at(100, 0);
    assert_eq!((s.lds, s.mem), (2000, 0));
    let s = at(0, 100);
    assert_eq!((s.lds, s.mem), (0, 2000));
    // 10 % and 30 % of 2000 operations, give or take what the seed draws
    let s = at(10, 30);
    assert!((150..250).contains(&s.lds) && (500..700).contains(&s.mem), "{s:?}");
}

#[test]
fn vgprs_stay_in_the_pool() {
    for vgprs in [2, 16, 200] {
        let s = shape(&program(&Knobs { ops: 3000, vgprs, ..Knobs::default() }));
        // v0 and v1 are the lane id and offset; the pool is v2..v(vgprs + 1)
        assert_eq!(s.max_vgpr, vgprs + 1, "vgprs {vgprs}");
    }
}

#[test]
fn every_program_translates_completely() {
    let mut n = 0;
    for wave64 in [true, false] {
        for (if_depth, loops, loop_depth) in [(0, 0, 1), (2, 1, 1), (4, 2, 3), (8, 4, 2)] {
            for (lds_pct, mem_pct) in [(0, 0), (5, 10), (20, 40)] {
                let k = Knobs { seed: n, ops: 300, if_depth, divergent_pct: 25, loops, loop_depth, wave64, lds_pct, mem_pct, ..Knobs::default() };
                let code = program(&k);
                let insts = lines(&code).len();
                for structured in [true, false] {
                    let t = translate(&code, &Options { structured, ..k.options() }).unwrap_or_else(|e| panic!("{k:?}: {e}"));
                    assert!(t.complete(), "{k:?}: {:?}", t.unsupported_why);
                    assert_eq!(t.instructions, insts, "{k:?}");
                }
                n += 1;
            }
        }
    }
}
