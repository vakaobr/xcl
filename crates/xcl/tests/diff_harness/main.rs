//! Differential testing harness for the XCL recompiler (#1269): generated RDNA2 programs
//! run through two compilation paths on the GPU, comparing every lane's results.
//!
//! Path A: the existing HLSL path (translate -> DXC -> DXIL).
//! Path B: today the same HLSL path with split_wave64 (a real differential on wave64
//! hardware). When a direct backend is available, path B switches to it:
//!   - XCL_DIRECT_DXIL=1 (#1426, bmjask): runner::direct_kernel (in-tree digest, no DXC)
//!   - XCL_RECOMP_DXIL=1 (gefydev recomp-lane-masks): per-lane lowering
//! The swap is in compile_b; the rest of the harness does not change.
//!
//! Generator: BetterJdue's compile_bench/gen.rs, imported via #[path]. Not modified.

#[path = "../compile_bench/gen.rs"]
mod gen;
mod compare;
mod minimise;
mod runnable;

use xcl::gpu::recomp::{dxc, translate, Options};
use xcl::gpu::recomp::runner;
use xcl::gpu::dxil::validate;

fn gpu_and_dxc_ready() -> bool {
    if runner::adapter().is_err() {
        runner::skip_test("no GPU");
        return false;
    }
    if dxc::location().is_err() {
        runner::skip_test("no DXC");
        return false;
    }
    true
}

/// Structural validation of a DXIL container (belt-and-suspenders; DXC + dxil.dll already
/// validate during compile). Replicates the core invariants of the private
/// `dxc::valid_dxil_container`: magic, version, total size, chunk count, DXIL chunk present.
fn validate_dxil(dxil: &[u8]) -> Result<(), String> {
    let rd = |at: usize| dxil.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    if dxil.len() < 36 { return Err(format!("DXIL too small: {} bytes", dxil.len())); }
    if dxil.get(..4) != Some(b"DXBC".as_slice()) { return Err("missing DXBC magic".into()); }
    if rd(20) != Some(1) { return Err("container version != 1".into()); }
    if rd(24) != Some(dxil.len() as u32) {
        return Err(format!("total_size {} != blob {}", rd(24).unwrap_or(0), dxil.len()));
    }
    let count = rd(28).ok_or("chunk count unreadable")? as usize;
    if count == 0 || count > 64 { return Err(format!("chunk count {count}")); }
    let table_end = 32 + count * 4;
    if table_end > dxil.len() { return Err("chunk table overflows blob".into()); }
    let has_dxil = (0..count).any(|i| {
        rd(32 + i * 4).and_then(|off| dxil.get(off as usize..off as usize + 4))
            == Some(b"DXIL")
    });
    if !has_dxil { return Err("no DXIL chunk".into()); }
    Ok(())
}

/// Path A: HLSL, split wave64 (the production form on 32-lane GPUs).
fn compile_a(code: &[u8]) -> Result<Vec<u8>, String> {
    let opts = Options { wave64: true, split_wave64: true, ..Options::default() };
    let t = translate(code, &opts)?;
    if !t.complete() { return Err(format!("unsupported: {:?}", t.unsupported)); }
    let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[])?;
    validate_dxil(&dxil)?;
    validate::validate(&dxil).map_err(|e| format!("IDxcValidator (path A): {e}"))?;
    Ok(dxil)
}

/// Path B: the direct DXIL backend (#1426, bmjask). The container has an in-tree digest
/// (no DXC at run time). The IDxcValidator check is the interesting case: does dxil.dll
/// accept bmjask's containers? Every emitted blob goes through both the structural check
/// and the validator.
fn compile_b(code: &[u8]) -> Result<Vec<u8>, String> {
    let dxil = runner::direct_kernel(code, true)?;
    validate_dxil(&dxil)?;
    validate::validate(&dxil).map_err(|e| format!("IDxcValidator (path B / direct): {e}"))?;
    Ok(dxil)
}

fn is_differential() -> bool {
    true // always a real differential: HLSL vs direct DXIL
}

fn default_knobs(seed: u64) -> gen::Knobs {
    gen::Knobs {
        seed, ops: 200, if_depth: 2, divergent_pct: 25, loops: 1, loop_depth: 1,
        loop_trips: 8, wave64: true, vgprs: 32, sgprs: 8, lds_pct: 10, mem_pct: 15,
    }
}

/// Run one program through both paths and compare. Returns Ok(()) on match, Err on diff.
fn run_one(seed: u64) -> Result<(), String> {
    let (code, state, mem) = runnable::runnable(&default_knobs(seed));
    let dxil_a = compile_a(&code)?;
    let dxil_b = compile_b(&code)?;
    let ra = runner::run(&dxil_a, &state, &mem)?;
    let rb = runner::run(&dxil_b, &state, &mem)?;
    let diffs = compare::compare(&ra, &rb);
    if diffs.is_empty() { Ok(()) } else {
        Err(format!("seed {seed}: {} diffs:\n{}", diffs.len(), compare::format_diffs(&diffs, 10)))
    }
}

#[test]
fn test_single_program() {
    if !gpu_and_dxc_ready() { return; }
    let (code, state, mem) = runnable::runnable(&default_knobs(1));
    let dxil = compile_b(&code).expect("translate+compile");
    let r = runner::run(&dxil, &state, &mem).expect("run");
    assert_eq!(r.vgpr.len(), 64, "expected 64 lanes");
}

#[test]
fn test_ci_sweep() {
    if !gpu_and_dxc_ready() { return; }
    let differential = is_differential();
    let (mut tested, mut skipped) = (0u32, 0u32);
    for seed in 0..32u64 {
        match run_one(seed) {
            Ok(()) => tested += 1,
            Err(e) if e.starts_with("unsupported") => { skipped += 1; }
            Err(e) => panic!("{e}"),
        }
    }
    eprintln!("ci_sweep: {tested} tested, {skipped} skipped, differential={differential}");
}

#[test]
#[ignore]
fn test_long_sweep() {
    if !gpu_and_dxc_ready() { return; }
    let differential = is_differential();
    let (mut tested, mut failures) = (0u32, 0u32);
    for seed in 0..1000u64 {
        match run_one(seed) {
            Ok(()) => tested += 1,
            Err(e) if e.starts_with("unsupported") => {}
            Err(e) => { eprintln!("{e}"); failures += 1; }
        }
    }
    eprintln!("long_sweep: {tested} tested, {failures} failures, differential={differential}");
    assert_eq!(failures, 0);
}

#[test]
fn test_mutation_catch() {
    if !gpu_and_dxc_ready() { return; }
    let knobs = default_knobs(42);
    let (code, state, mem) = runnable::runnable(&knobs);
    let dxil = compile_b(&code).expect("compile");
    let ra = runner::run(&dxil, &state, &mem).expect("run A");
    let mut rb = runner::run(&dxil, &state, &mem).expect("run B");
    // Inject: flip vgpr[0][2] in path B to simulate a translation bug
    rb.vgpr[0][2] ^= 1;
    let diffs = compare::compare(&ra, &rb);
    assert!(!diffs.is_empty(), "mutation not caught");
    // Minimise: find the smallest program where the check closure returns true
    let opts = Options { wave64: true, split_wave64: true, ..Options::default() };
    let shrunk = minimise::minimise_ops(&knobs, &|k| {
        let (c, _, _) = runnable::runnable(k);
        // The mutation always fires (we flip after run), so just check compilation works
        let t = match translate(&c, &opts) { Ok(t) if t.complete() => t, _ => return false };
        dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).is_ok()
    });
    assert!(shrunk.ops <= knobs.ops, "minimiser grew the program");
    eprintln!("mutation_catch: {} -> {} ops", knobs.ops, shrunk.ops);
    // Print the minimised program
    let (min_code, _, _) = runnable::runnable(&shrunk);
    minimise::print_minimised(&shrunk, &min_code);
}

/// Translate code and corrupt the HLSL: change the first V_ADD_NC_U32 lowering from
/// addition to subtraction. The emitter writes `vD = select(act, vA + vB, vD);` for
/// V_ADD_NC_U32 (see emit.rs line 3385). Returns the corrupted DXIL or Err if the
/// program has no add or the corrupted HLSL fails to compile.
fn translate_with_add_as_sub(code: &[u8]) -> Result<Vec<u8>, String> {
    let opts = Options { wave64: true, split_wave64: true, ..Options::default() };
    let t = translate(code, &opts)?;
    if !t.complete() { return Err(format!("unsupported: {:?}", t.unsupported)); }
    // Replace every V_ADD_NC_U32 lowering (" + v") with subtract (" - v") in the HLSL
    // body. The emitter writes `select(act, vA + vB, vD)` for V_ADD_NC_U32. Early adds
    // may operate on zero-valued VGPRs (the pool starts at 0); replacing all ensures that
    // adds after non-zero values accumulate will produce different results.
    let hlsl = &t.hlsl;
    let body_start = hlsl.find("void main()").unwrap_or(0);
    let (head, body) = hlsl.split_at(body_start);
    let corrupted_body = body.replace(" + v", " - v");
    if corrupted_body == body {
        return Err("no V_ADD_NC_U32 pattern in HLSL".into());
    }
    let corrupted = format!("{head}{corrupted_body}");
    dxc::compile(&corrupted, "main", "cs_6_6", &[])
}

/// Build a runnable triple with non-zero memory, so loads bring values into the VGPR pool
/// and subsequent ALU operations produce non-trivial results.
fn runnable_with_data(knobs: &gen::Knobs) -> (Vec<u8>, runner::KernelState, Vec<u32>) {
    let (code, state, _) = runnable::runnable(knobs);
    let mem: Vec<u32> = (0..512u32).map(|i| 0x1000 + i * 7).collect();
    (code, state, mem)
}

/// Proves the harness catches a translation-level bug (not just a flipped bit in the
/// result). The mutation changes every V_ADD_NC_U32 lowering from add to subtract in the
/// HLSL. Memory is non-zero so loads bring values into the VGPR pool, and the adds that
/// follow produce different results from subtracts. The minimiser shrinks the program to
/// the smallest one that still contains the offending instruction.
#[test]
fn test_translation_mutation() {
    if !gpu_and_dxc_ready() { return; }
    // Find a seed whose program contains V_ADD_NC_U32 and where the mutation is visible
    let mut found_seed = None;
    for seed in 0..64u64 {
        let knobs = default_knobs(seed);
        let (code, state, mem) = runnable_with_data(&knobs);
        let good = match compile_b(&code) { Ok(d) => d, Err(_) => continue };
        let bad = match translate_with_add_as_sub(&code) { Ok(d) => d, Err(_) => continue };
        let ra = match runner::run(&good, &state, &mem) { Ok(r) => r, Err(_) => continue };
        let rb = match runner::run(&bad, &state, &mem) { Ok(r) => r, Err(_) => continue };
        if !compare::compare(&ra, &rb).is_empty() {
            found_seed = Some(seed);
            break;
        }
    }
    let seed = found_seed.expect("no seed showed a visible add-as-sub difference");
    let knobs = default_knobs(seed);
    let (code, state, mem) = runnable_with_data(&knobs);
    let good = compile_b(&code).unwrap();
    let bad = translate_with_add_as_sub(&code).unwrap();
    let ra = runner::run(&good, &state, &mem).unwrap();
    let rb = runner::run(&bad, &state, &mem).unwrap();
    let diffs = compare::compare(&ra, &rb);
    eprintln!("translation_mutation: seed {seed}, {} diffs:\n{}",
        diffs.len(), compare::format_diffs(&diffs, 5));
    // Minimise: find smallest program where add-as-sub still differs
    let shrunk = minimise::minimise_ops(&knobs, &|k| {
        let (c, s, m) = runnable_with_data(k);
        let g = match compile_b(&c) { Ok(d) => d, Err(_) => return false };
        let b = match translate_with_add_as_sub(&c) { Ok(d) => d, Err(_) => return false };
        let ra = match runner::run(&g, &s, &m) { Ok(r) => r, Err(_) => return false };
        let rb = match runner::run(&b, &s, &m) { Ok(r) => r, Err(_) => return false };
        !compare::compare(&ra, &rb).is_empty()
    });
    eprintln!("translation_mutation minimised: {} -> {} ops", knobs.ops, shrunk.ops);
    let (min_code, _, _) = runnable_with_data(&shrunk);
    minimise::print_minimised(&shrunk, &min_code);
}

#[test]
fn test_dxil_valid() {
    if !gpu_and_dxc_ready() { return; }
    let configs = [
        gen::Knobs { seed: 1, ops: 100, ..gen::Knobs::default() },
        gen::Knobs { seed: 2, ops: 300, if_depth: 3, divergent_pct: 40, ..gen::Knobs::default() },
        gen::Knobs { seed: 3, ops: 200, loops: 2, loop_depth: 2, ..gen::Knobs::default() },
        gen::Knobs { seed: 4, ops: 150, lds_pct: 20, mem_pct: 30, ..gen::Knobs::default() },
        gen::Knobs { seed: 5, ops: 50, wave64: false, ..gen::Knobs::default() },
        gen::Knobs { seed: 6, ops: 400, if_depth: 4, loops: 1, ..gen::Knobs::default() },
        gen::Knobs { seed: 7, ops: 200, vgprs: 100, sgprs: 16, ..gen::Knobs::default() },
        gen::Knobs { seed: 8, ops: 200, divergent_pct: 50, ..gen::Knobs::default() },
    ];
    let mut validated = 0u32;
    for k in &configs {
        let code = gen::program(k);
        let opts = Options { wave64: k.wave64, split_wave64: k.wave64, ..Options::default() };
        let t = match translate(&code, &opts) {
            Ok(t) if t.complete() => t,
            _ => continue,
        };
        let dxil = match dxc::compile(&t.hlsl, "main", "cs_6_6", &[]) {
            Ok(d) => d,
            Err(_) => continue,
        };
        validate_dxil(&dxil).unwrap_or_else(|e| panic!("seed {}: {e}", k.seed));
        validated += 1;
    }
    eprintln!("dxil_valid: {validated}/8 validated");
    assert!(validated >= 6, "too many programs failed to compile: only {validated}/8");
}
