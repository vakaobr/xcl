//! Wraps the generator output into a runnable (code, state, memory) triple.
//!
//! Memory is at address 0: s[2:3] = 0 in the default state, and the translator maps
//! GLOBAL_LOAD/STORE through the mem UAV (u2), indexed by ADDR + OFFSET. Maximum byte
//! address the generator produces: 252 (v1 max) + 1020 (OFFSET max) = 1272, so a
//! 512-dword (2048-byte) buffer covers it with margin. LDS addresses (v1 = lane * 4,
//! range 0..252 bytes) stay well below the split-wave64 exchange area.
//!
//! The generator does not emit f64 or 64-bit atomics. If it is extended to do so, those
//! must be gated: D3DMetal (CrossOver) produces no output for them (#983, #787).

use super::gen;
use xcl::gpu::recomp::runner::KernelState;
use xcl::gpu::recomp::KERNEL_LANES;

pub fn runnable(knobs: &gen::Knobs) -> (Vec<u8>, KernelState, Vec<u32>) {
    let code = gen::program(knobs);
    let mut state = KernelState::default();
    for lane in 0..KERNEL_LANES {
        state.vgpr[lane][0] = lane as u32; // v0 = lane id (generator reads it)
    }
    (code, state, vec![0u32; 512])
}
