//! AVX / AVX2 masked stores (VMASKMOVPS, VMASKMOVPD, VPMASKMOVD, VPMASKMOVQ with a memory
//! destination) of a title, replaced when its image is loaded on a host whose translator
//! cannot resume them after a fault.
//!
//! Rosetta 2 runs a masked store an element at a time and can restart it only from its first
//! element: a fault there is delivered and resumed like any other, a fault on a later element
//! ends the process ("rosetta error: unexpectedly need to EmulateForward on a synchronous
//! exception"). That is a store which starts on a page that can be written and goes on into one
//! that cannot – what a write watch kept by page protection (Wine on macOS) makes of every
//! page boundary inside a buffer the title fills with such stores.
//!
//! Each store becomes `call helper` (five bytes; these instructions have at least five) and
//! NOPs. The helper stores the selected elements one at a time with ordinary `mov`s, which
//! fault and resume as any store does, and changes nothing else: no register, no flag
//! (it has no instruction that writes one), no byte of an element whose mask bit is clear –
//! such an element is not touched at all, so a store whose tail is masked out may end at the
//! last byte of a mapping as before.
//!
//! This module holds the parts that do not depend on Windows:
//!
//! * `insn_len` – the length of an x86-64 instruction, to walk a function from its start.
//! * `decode` – the four instructions with any memory operand.
//! * `helper` – the code a site calls.
//! * `plan` – the sites of an image: byte pattern, then the walk from the start of the function
//!   the unwind data names (or from the end of the one before, for a leaf function that has no
//!   unwind data), which must arrive exactly at the site.
//!
//! A later change applies a plan in `loader` (after relocation, before the sections are
//! protected); nothing calls this module yet.

/// Length of the x86-64 instruction at the start of `c`. `None`: the bytes run out, or it is
/// something this walk does not know and must not guess at – EVEX, XOP, 3DNow!, an opcode
/// that is not valid in 64-bit mode (which in a function means data).
pub fn insn_len(c: &[u8]) -> Option<usize> {
    let (mut i, mut p66, mut p67, mut pf2, mut legacy) = (0usize, false, false, false, false);
    loop {
        match *c.get(i)? {
            0x66 => p66 = true,
            0x67 => p67 = true,
            0xF2 => pf2 = true,
            0xF3 | 0xF0 => legacy = true,
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 => {}
            _ => break,
        }
        i += 1;
        if i > 14 {
            return None;
        }
    }
    let (mut rex, mut rex_w) = (false, false);
    if *c.get(i)? & 0xF0 == 0x40 {
        rex = true;
        rex_w = *c.get(i)? & 8 != 0;
        i += 1;
    }
    let op = *c.get(i)?;
    i += 1;
    // an immediate of the operand size, at most four bytes
    let immz = if rex_w || !p66 { 4 } else { 2 };
    // (has ModRM, immediate bytes)
    let (modrm, mut imm): (bool, usize) = match op {
        0xC4 | 0xC5 => {
            // VEX takes no REX, LOCK or 66 / F2 / F3 in front
            if rex || p66 || pf2 || legacy {
                return None;
            }
            let map = if op == 0xC5 {
                i += 1;
                1
            } else {
                i += 2;
                *c.get(i - 2)? & 0x1F
            };
            let o = *c.get(i)?;
            i += 1;
            match map {
                1 => (o != 0x77, matches!(o, 0x70..=0x73 | 0xC2 | 0xC4..=0xC6) as usize),
                2 => (true, 0),
                3 => (true, 1),
                _ => return None,
            }
        }
        0x0F => {
            let o = *c.get(i)?;
            i += 1;
            match o {
                0x38 | 0x3A => {
                    c.get(i)?;
                    i += 1;
                    (true, (o == 0x3A) as usize)
                }
                0x05..=0x09 | 0x0B | 0x30..=0x35 | 0x37 | 0x77 | 0xA0..=0xA2 | 0xA8..=0xAA | 0xC8..=0xCF => (false, 0),
                0x80..=0x8F => (false, 4),
                0x70..=0x73 | 0xA4 | 0xAC | 0xBA | 0xC2 | 0xC4..=0xC6 => (true, 1),
                // SSE4a EXTRQ / INSERTQ with length and index bytes
                0x78 if p66 || pf2 => (true, 2),
                0x04 | 0x0A | 0x0C | 0x0E | 0x0F | 0x24..=0x27 | 0x36 | 0x39 | 0x3B..=0x3F | 0x7A | 0x7B | 0xA6 | 0xA7 | 0xFF => return None,
                _ => (true, 0),
            }
        }
        0x00..=0x3F => match op & 7 {
            0..=3 => (true, 0),
            4 => (false, 1),
            5 => (false, immz),
            _ => return None,
        },
        0x50..=0x5F | 0x6C..=0x6F | 0x90..=0x99 | 0x9B..=0x9F | 0xA4..=0xA7 | 0xAA..=0xAF => (false, 0),
        0xC3 | 0xC9 | 0xCB | 0xCC | 0xCF | 0xD7 | 0xEC..=0xEF | 0xF1 | 0xF4 | 0xF5 | 0xF8..=0xFD => (false, 0),
        // MOV to or from a segment register: there are six
        0x8C | 0x8E if (*c.get(i)? >> 3) & 7 > 5 => return None,
        0x63 | 0x84..=0x8E | 0xD0..=0xD3 | 0xD8..=0xDF | 0xFE | 0xFF => (true, 0),
        // POP r/m; any other /r is XOP
        0x8F if *c.get(i)? & 0x38 == 0 => (true, 0),
        0x68 | 0xA9 => (false, immz),
        0x69 | 0x81 | 0xC7 => (true, immz),
        0x6A | 0x70..=0x7F | 0xA8 | 0xB0..=0xB7 | 0xCD | 0xE0..=0xE7 | 0xEB => (false, 1),
        0x6B | 0x80 | 0x83 | 0xC0 | 0xC1 | 0xC6 => (true, 1),
        0xA0..=0xA3 => (false, if p67 { 4 } else { 8 }),
        0xB8..=0xBF => (false, if rex_w { 8 } else { immz }),
        0xC2 | 0xCA => (false, 2),
        0xC8 => (false, 3),
        0xE8 | 0xE9 => (false, 4),
        // TEST r/m, imm is /0 and /1 of the group
        0xF6 => (true, (*c.get(i)? & 0x30 == 0) as usize),
        0xF7 => (true, if *c.get(i)? & 0x30 == 0 { immz } else { 0 }),
        _ => return None,
    };
    if modrm {
        let m = *c.get(i)?;
        i += 1;
        let (md, rm) = (m >> 6, m & 7);
        if md != 3 {
            let mut base = rm;
            if rm == 4 {
                base = *c.get(i)? & 7;
                i += 1;
            }
            imm += match md {
                0 if base == 5 => 4,
                0 => 0,
                1 => 1,
                _ => 4,
            };
        }
    }
    i += imm;
    (i <= 15 && i <= c.len()).then_some(i)
}

/// Which of the four instructions: the element size and what the log calls it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// VEX.66.0F38.W0 2E /r: VMASKMOVPS m, mask, src
    Ps,
    /// VEX.66.0F38.W0 2F /r: VMASKMOVPD m, mask, src
    Pd,
    /// VEX.66.0F38.W0 8E /r: VPMASKMOVD m, mask, src
    D,
    /// VEX.66.0F38.W1 8E /r: VPMASKMOVQ m, mask, src
    Q,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Ps, Kind::Pd, Kind::D, Kind::Q];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Ps => "VMASKMOVPS",
            Kind::Pd => "VMASKMOVPD",
            Kind::D => "VPMASKMOVD",
            Kind::Q => "VPMASKMOVQ",
        }
    }

    /// Bytes of one element.
    pub fn element(self) -> usize {
        match self {
            Kind::Ps | Kind::D => 4,
            Kind::Pd | Kind::Q => 8,
        }
    }
}

/// The memory operand: base + index * 2^scale + disp, or the next instruction's address + disp.
/// Registers are the hardware's numbers (4 RSP, 8–15 R8–R15).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mem {
    pub base: Option<u8>,
    pub index: Option<u8>,
    pub scale: u8,
    pub disp: i32,
    pub rip: bool,
}

/// A masked store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Store {
    pub kind: Kind,
    /// Length in bytes (5..=10).
    pub len: u8,
    /// 256 bits (YMM); else 128 (XMM).
    pub wide: bool,
    /// The register stored from (ModRM reg) and the mask register (VEX.vvvv).
    pub src: u8,
    pub mask: u8,
    pub mem: Mem,
}

impl Store {
    /// Number of elements.
    pub fn elements(&self) -> usize {
        (if self.wide { 32 } else { 16 }) / self.kind.element()
    }
}

/// The first five bytes of every masked store: three-byte VEX with map 0F38 and prefix 66, one
/// of the three opcodes, a ModRM byte with a memory operand.
pub fn is_candidate(c: &[u8]) -> bool {
    matches!(c, [0xC4, b1, b2, 0x2E | 0x2F | 0x8E, m, ..] if b1 & 0x1F == 2 && b2 & 3 == 1 && m >> 6 != 3)
}

/// Decodes the masked store at the start of `c` (`None`: not one, or cut short). A prefix in
/// front (a segment override, 32-bit addressing) makes it another instruction start, which
/// this does not take.
pub fn decode(c: &[u8]) -> Option<Store> {
    if !is_candidate(c) {
        return None;
    }
    let (b1, b2, m) = (c[1], c[2], c[4]);
    let (r, x, b, w) = ((!b1 >> 7) & 1, (!b1 >> 6) & 1, (!b1 >> 5) & 1, b2 >> 7);
    let kind = match (c[3], w) {
        (0x2E, 0) => Kind::Ps,
        (0x2F, 0) => Kind::Pd,
        (0x8E, 0) => Kind::D,
        (0x8E, _) => Kind::Q,
        _ => return None, // W1 of the floating-point forms is not an instruction
    };
    let (md, rm) = (m >> 6, m & 7);
    let mut i = 5;
    let (mut base, mut index, mut scale, mut rip) = (None, None, 0, false);
    let mut disp_len = match md {
        1 => 1,
        2 => 4,
        _ => 0,
    };
    if rm == 4 {
        let sib = *c.get(i)?;
        i += 1;
        scale = sib >> 6;
        let ix = ((sib >> 3) & 7) | x << 3;
        if ix != 4 {
            index = Some(ix); // index 4 without VEX.X: none (R12 is an index)
        }
        if sib & 7 == 5 && md == 0 {
            disp_len = 4; // disp32 without a base
        } else {
            base = Some((sib & 7) | b << 3);
        }
    } else if rm == 5 && md == 0 {
        rip = true;
        disp_len = 4;
    } else {
        base = Some(rm | b << 3);
    }
    let disp = match disp_len {
        1 => *c.get(i)? as i8 as i32,
        4 => i32::from_le_bytes([*c.get(i)?, *c.get(i + 1)?, *c.get(i + 2)?, *c.get(i + 3)?]),
        _ => 0,
    };
    i += disp_len;
    Some(Store { kind, len: i as u8, wide: b2 & 4 != 0, src: ((m >> 3) & 7) | r << 3, mask: (!b2 >> 3) & 15, mem: Mem { base, index, scale, disp, rip } })
}

/// Largest helper `helper` writes.
pub const HELPER_MAX: usize = 192;

/// What a helper has between the site's RSP and its own while it stores: the return address,
/// four saved registers and 64 bytes for the two vectors.
const FRAME: i64 = 8 + 32 + 64;

/// `push rax; push rcx; push rdx; push r8; lea rsp, [rsp - 64]` – every helper starts with it,
/// which is what `UNWIND_INFO` describes.
const PROLOGUE: [u8; 10] = [0x50, 0x51, 0x52, 0x41, 0x50, 0x48, 0x8D, 0x64, 0x24, 0xC0];
/// `lea rsp, [rsp + 64]; pop r8; pop rdx; pop rcx; pop rax; ret`
const EPILOGUE: [u8; 11] = [0x48, 0x8D, 0x64, 0x24, 0x40, 0x41, 0x58, 0x5A, 0x59, 0x58, 0xC3];

/// The unwind data all helpers share (UNWIND_INFO, version 1): a ten-byte prologue that
/// allocates 64 bytes after pushing R8, RDX, RCX and RAX. With it a fault in a helper that no
/// one repairs is unwound into the title's function like one at the store itself.
pub const UNWIND_INFO: [u8; 16] = [
    0x01, PROLOGUE.len() as u8, 5, 0, // version 1, prologue size, five codes, no frame register
    10, 0x72, // after `lea rsp`: UWOP_ALLOC_SMALL, (7 + 1) * 8 bytes
    5, 0x80, // after `push r8`: UWOP_PUSH_NONVOL r8
    3, 0x20, // rdx
    2, 0x10, // rcx
    1, 0x00, // rax
    0, 0,
];

/// `vmovdqu [rsp + disp], reg` (XMM, or YMM when `wide`).
fn spill(out: &mut Vec<u8>, reg: u8, wide: bool, disp: u8) {
    out.extend_from_slice(&[0xC5, (if reg < 8 { 0x80 } else { 0 }) | 0x7A | (wide as u8) << 2, 0x7F]);
    if disp == 0 {
        out.extend_from_slice(&[0x04 | (reg & 7) << 3, 0x24]);
    } else {
        out.extend_from_slice(&[0x44 | (reg & 7) << 3, 0x24, disp]);
    }
}

/// Appends to `out` position-independent code that does what the store `s` does and returns:
/// the code a `call` written over the store reaches. `next` is the address of the instruction
/// after the store (what a RIP-relative operand counts from). `false`: nothing was appended –
/// an RSP-relative displacement that does not fit once the helper's frame is added.
///
/// The address is taken first, into R8, while every register still holds the title's value
/// (RSP is lower by `FRAME`, which the displacement makes up for). The source and the mask go
/// to the stack with two unaligned vector stores, and each element is then
/// `mov eax, mask[i]; cdq; mov ecx, edx; jrcxz skip; mov eax, src[i]; mov [r8 + 4 i], eax`
/// (with RAX / CQO for 8-byte elements): CDQ copies the mask's sign bit into every bit of EDX,
/// JRCXZ branches on RCX without a flag. Nothing in it writes a flag or a vector register.
pub fn helper(s: &Store, next: u64, out: &mut Vec<u8>) -> bool {
    let start = out.len();
    out.extend_from_slice(&PROLOGUE);
    let Mem { base, index, scale, disp, rip } = s.mem;
    if rip {
        // mov r8, imm64
        out.extend_from_slice(&[0x49, 0xB8]);
        out.extend_from_slice(&next.wrapping_add(disp as i64 as u64).to_le_bytes());
    } else {
        let disp = disp as i64 + if base == Some(4) { FRAME } else { 0 };
        let Ok(disp) = i32::try_from(disp) else {
            out.truncate(start);
            return false;
        };
        // lea r8, [base + index * 2^scale + disp]
        out.extend_from_slice(&[0x4C | index.map_or(0, |x| x >> 3) << 1 | base.map_or(0, |b| b >> 3), 0x8D]);
        let sib = scale << 6 | index.map_or(4, |x| x & 7) << 3;
        match base {
            None => {
                out.extend_from_slice(&[0x04, sib | 5]);
                out.extend_from_slice(&disp.to_le_bytes());
            }
            Some(b) => {
                // RBP / R13 as a base have no form without a displacement
                let md = if disp == 0 && b & 7 != 5 {
                    0
                } else if i8::try_from(disp).is_ok() {
                    1
                } else {
                    2
                };
                if index.is_some() || b & 7 == 4 {
                    out.extend_from_slice(&[md << 6 | 4, sib | (b & 7)]);
                } else {
                    out.push(md << 6 | (b & 7));
                }
                match md {
                    1 => out.push(disp as u8),
                    2 => out.extend_from_slice(&disp.to_le_bytes()),
                    _ => {}
                }
            }
        }
    }
    spill(out, s.src, s.wide, 0);
    spill(out, s.mask, s.wide, 0x20);
    for i in 0..s.elements() {
        let d = (i * s.kind.element()) as u8;
        if s.kind.element() == 4 {
            out.extend_from_slice(&[0x8B, 0x44, 0x24, 0x20 + d, 0x99, 0x89, 0xD1, 0xE3, 0x08]);
            out.extend_from_slice(&[0x8B, 0x44, 0x24, d, 0x41, 0x89, 0x40, d]);
        } else {
            out.extend_from_slice(&[0x48, 0x8B, 0x44, 0x24, 0x20 + d, 0x48, 0x99, 0x89, 0xD1, 0xE3, 0x09]);
            out.extend_from_slice(&[0x48, 0x8B, 0x44, 0x24, d, 0x49, 0x89, 0x40, d]);
        }
    }
    out.extend_from_slice(&EPILOGUE);
    debug_assert!(out.len() - start <= HELPER_MAX);
    true
}

/// The bytes that replace a store of `len` bytes at `site`: `call helper`, then NOPs (the
/// first `len` of the array). `None`: the helper is out of the call's ±2 GiB reach.
pub fn call_site(site: u64, len: u8, helper: u64) -> Option<[u8; 10]> {
    if !(5..=10).contains(&len) {
        return None;
    }
    let rel = i32::try_from(helper.wrapping_sub(site.wrapping_add(5)) as i64).ok()?;
    let mut b = [0x90u8; 10];
    b[0] = 0xE8;
    b[1..5].copy_from_slice(&rel.to_le_bytes());
    Some(b)
}

/// Why a byte pattern was left as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// In no function of the unwind data, and further than `GAP_MAX` from the end of one.
    NoFunction,
    /// An instruction on the way there (at this RVA) is not one `insn_len` knows.
    Undecodable(u32),
    /// The walk passes over it: the bytes are part of another instruction (or follow a prefix).
    NotAtBoundary,
    /// At a boundary but not a store `decode` and `helper` take.
    Form,
}

/// How far behind the end of a function with unwind data a leaf function's store may be.
pub const GAP_MAX: u32 = 0x1_0000;

/// A store to replace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub rva: u32,
    pub store: Store,
    /// Offset of its helper in `Plan::code`.
    pub helper: u32,
    /// Not in a function of the unwind data: found by the walk from the end of the one before.
    pub leaf: bool,
}

/// What to do to an image.
#[derive(Default, Debug)]
pub struct Plan {
    pub sites: Vec<Site>,
    pub skipped: Vec<(u32, Skip)>,
    /// The helpers, each at a multiple of 16; stores with the same bytes share one.
    pub code: Vec<u8>,
    /// (start, end) of each helper in `code`, ascending.
    pub helpers: Vec<(u32, u32)>,
}

/// Finds the masked stores in the executable sections of a mapped image and writes their
/// helpers. `image` is the image as mapped (relocated), `base` its address, `sections` the
/// (RVA, length) of each executable section and `functions` the (start, end) RVAs of the
/// unwind data's entries.
///
/// A byte pattern alone is not taken: the instructions are walked from the start of the
/// function that contains it, and the walk has to arrive exactly at it. A pattern outside
/// every function is walked to from the end of the function before it (a leaf function has no
/// unwind data; the padding between functions is one-byte instructions).
pub fn plan(image: &[u8], base: u64, sections: &[(u32, u32)], functions: &[(u32, u32)]) -> Plan {
    let mut p = Plan::default();
    let mut funcs: Vec<(u32, u32)> = functions.iter().copied().filter(|f| f.0 < f.1).collect();
    funcs.sort_unstable();
    // helpers by the store's bytes (and, for a RIP-relative operand, its place)
    let mut shared: std::collections::HashMap<([u8; 10], u32), u32> = std::collections::HashMap::new();
    for &(s_rva, s_len) in sections {
        let (s_start, s_end) = (s_rva as usize, (s_rva as usize).saturating_add(s_len as usize).min(image.len()));
        let Some(code) = image.get(s_start..s_end) else { continue };
        // where the last walk started and how far it got
        let (mut anchor, mut cursor) = (usize::MAX, 0usize);
        let mut at = 0;
        while let Some(k) = code[at..].iter().position(|&b| b == 0xC4) {
            let site = at + k;
            at = site + 1;
            if !is_candidate(&code[site..]) {
                continue;
            }
            let rva = (s_start + site) as u32;
            let before = funcs.partition_point(|f| f.0 <= rva);
            let (from, leaf) = match before.checked_sub(1).map(|i| funcs[i]) {
                Some((start, end)) if rva < end => (start as usize, false),
                Some((_, end)) if end as usize >= s_start && rva - end <= GAP_MAX => (end as usize, true),
                _ => {
                    p.skipped.push((rva, Skip::NoFunction));
                    continue;
                }
            };
            if from < s_start {
                p.skipped.push((rva, Skip::NoFunction));
                continue;
            }
            let from = from - s_start;
            if anchor != from || cursor > site {
                (anchor, cursor) = (from, from);
            }
            let mut bad = None;
            while cursor < site {
                match insn_len(&code[cursor..]) {
                    Some(n) => cursor += n,
                    None => {
                        bad = Some(Skip::Undecodable((s_start + cursor) as u32));
                        anchor = usize::MAX;
                        break;
                    }
                }
            }
            if bad.is_none() && cursor != site {
                bad = Some(Skip::NotAtBoundary);
            }
            if let Some(why) = bad {
                p.skipped.push((rva, why));
                continue;
            }
            let Some(store) = decode(&code[site..]).filter(|s| insn_len(&code[site..]) == Some(s.len as usize)) else {
                p.skipped.push((rva, Skip::Form));
                continue;
            };
            let mut key = ([0u8; 10], 0);
            key.0[..store.len as usize].copy_from_slice(&code[site..site + store.len as usize]);
            if store.mem.rip {
                key.1 = rva;
            }
            let helper_at = match shared.get(&key) {
                Some(&h) => h,
                None => {
                    while p.code.len() % 16 != 0 {
                        p.code.push(0xCC);
                    }
                    let h = p.code.len() as u32;
                    if !helper(&store, base + rva as u64 + store.len as u64, &mut p.code) {
                        p.skipped.push((rva, Skip::Form));
                        continue;
                    }
                    p.helpers.push((h, p.code.len() as u32));
                    shared.insert(key, h);
                    h
                }
            };
            p.sites.push(Site { rva, store, helper: helper_at, leaf });
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lengths as an assembler gives them, for the kinds of instruction a function has on the
    /// way to a masked store – and `None` for what must stop a walk.
    #[test]
    fn instruction_lengths() {
        let cases: &[(&[u8], usize)] = &[
            (&[0x90], 1),                                                       // nop
            (&[0xC3], 1),                                                       // ret
            (&[0x55], 1),                                                       // push rbp
            (&[0x41, 0x57], 2),                                                 // push r15
            (&[0x48, 0x83, 0xEC, 0x28], 4),                                     // sub rsp, 0x28
            (&[0x48, 0x81, 0xEC, 0x88, 0x00, 0x00, 0x00], 7),                   // sub rsp, 0x88
            (&[0x48, 0x8B, 0x05, 1, 2, 3, 4], 7),                               // mov rax, [rip + disp32]
            (&[0x48, 0x8B, 0x44, 0x24, 0x08], 5),                               // mov rax, [rsp + 8]
            (&[0x4A, 0x8D, 0x04, 0xC5, 0, 0, 0, 0], 8),                         // lea rax, [r8 * 8 + disp32]
            (&[0x48, 0xB8, 1, 2, 3, 4, 5, 6, 7, 8], 10),                        // mov rax, imm64
            (&[0xB8, 1, 2, 3, 4], 5),                                           // mov eax, imm32
            (&[0x66, 0xB8, 1, 2], 4),                                           // mov ax, imm16
            (&[0x66, 0xC7, 0x00, 1, 2], 5),                                     // mov word [rax], imm16
            (&[0xC7, 0x04, 0x25, 0, 0, 0, 0, 1, 2, 3, 4], 11),                  // mov dword [0], imm32
            (&[0xF6, 0xC1, 0x01], 3),                                           // test cl, 1
            (&[0xF6, 0xD9], 2),                                                 // neg cl
            (&[0xF7, 0xC1, 1, 2, 3, 4], 6),                                     // test ecx, imm32
            (&[0x48, 0xF7, 0xD8], 3),                                           // neg rax
            (&[0xE8, 1, 2, 3, 4], 5),                                           // call
            (&[0x74, 0x10], 2),                                                 // je
            (&[0x0F, 0x84, 1, 2, 3, 4], 6),                                     // je rel32
            (&[0xF2, 0xC3], 2),                                                 // bnd ret
            (&[0xFF, 0x15, 1, 2, 3, 4], 6),                                     // call [rip + disp32]
            (&[0x48, 0xA1, 1, 2, 3, 4, 5, 6, 7, 8], 10),                        // mov rax, [moffs64]
            (&[0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0, 0, 0, 0], 11),      // nop word cs:[rax + rax + disp32]
            (&[0x0F, 0x1F, 0x44, 0x00, 0x00], 5),                               // nop dword [rax + rax + 0]
            (&[0x0F, 0xBA, 0xE0, 0x07], 4),                                     // bt eax, 7
            (&[0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x08], 6),                         // palignr xmm0, xmm1, 8
            (&[0xF2, 0x0F, 0x78, 0xC1, 0x08, 0x10], 6),                         // insertq xmm0, xmm1, 8, 16
            (&[0x0F, 0x01, 0xFC], 3),                                           // clzero
            (&[0xF3, 0x48, 0x0F, 0xB8, 0xC1], 5),                               // popcnt rax, rcx
            (&[0xC5, 0xF8, 0x77], 3),                                           // vzeroupper
            (&[0xC5, 0xFE, 0x7F, 0x04, 0x24], 5),                               // vmovdqu [rsp], ymm0
            (&[0xC5, 0xF9, 0x70, 0xC1, 0x1B], 5),                               // vpshufd xmm0, xmm1, 0x1B
            (&[0xC4, 0xE2, 0x7D, 0x21, 0x05, 1, 2, 3, 4], 9),                   // vpmovsxbd ymm0, [rip + disp32]
            (&[0xC4, 0xE3, 0x7D, 0x39, 0xC1, 0x01], 6),                         // vextracti128 xmm1, ymm0, 1
            (&[0xC4, 0xE3, 0x75, 0x4C, 0xC2, 0x30], 6),                         // vpblendvb ymm0, ymm1, ymm2, ymm3
            (&[0xC4, 0xE2, 0x4D, 0x8E, 0x04, 0x90], 6),                         // vpmaskmovd [rax + 4 * rdx], ymm6, ymm0
            (&[0xC4, 0xE2, 0xF8, 0xF7, 0xC1], 5),                               // bextr rax, rcx, rax
        ];
        for (code, len) in cases {
            assert_eq!(insn_len(code), Some(*len), "{code:02x?}");
            // followed by other bytes it is no longer, cut short it is nothing
            let mut more = code.to_vec();
            more.extend_from_slice(&[0xC4; 16]);
            assert_eq!(insn_len(&more), Some(*len), "{code:02x?} with a tail");
            assert_eq!(insn_len(&code[..len - 1]), None, "{code:02x?} cut short");
        }
        // EVEX, an opcode 64-bit mode does not have, XOP, 3DNow!, a far jump, REX before VEX, and
        // the first word of a jump table a function kept in its code (8E /6: no such register)
        for code in [&[0x62, 0xF1, 0x7C, 0x48, 0x10, 0x00][..], &[0x06], &[0x8F, 0xE8, 0x78, 0xC2, 0xC1, 0x04], &[0x0F, 0x0F, 0xC1, 0x9E], &[0xEA], &[0x41, 0xC5, 0xF8, 0x77], &[0x8E, 0xB2, 0x5E, 0x05, 0x98, 0xB2]] {
            assert_eq!(insn_len(code), None, "{code:02x?}");
        }
    }

    const NO_MEM: Mem = Mem { base: None, index: None, scale: 0, disp: 0, rip: false };

    /// Every shape of the instruction a retail game has (1,130 sites: lengths 5, 6, 7, 9 and
    /// 10; RSP, R12, R13 as bases; XMM forms), plus the ones it has not: RIP-relative, an
    /// address without a base.
    #[test]
    fn decodes_the_masked_stores() {
        let mem = |base, index, scale, disp| Mem { base, index, scale, disp, rip: false };
        let cases: &[(&[u8], Kind, bool, u8, u8, Mem)] = &[
            // vpmaskmovq [rsp + r8 + 0x40], ymm7, ymm8
            (&[0xC4, 0x22, 0xC5, 0x8E, 0x44, 0x04, 0x40], Kind::Q, true, 8, 7, mem(Some(4), Some(8), 0, 0x40)),
            // vmaskmovps [rcx + 4 * rdx], ymm4, ymm5
            (&[0xC4, 0xE2, 0x5D, 0x2E, 0x2C, 0x91], Kind::Ps, true, 5, 4, mem(Some(1), Some(2), 2, 0)),
            // vpmaskmovd [r12 + 4 * r8], ymm5, ymm6
            (&[0xC4, 0x82, 0x55, 0x8E, 0x34, 0x84], Kind::D, true, 6, 5, mem(Some(12), Some(8), 2, 0)),
            // vpmaskmovd [rcx], ymm7, ymm6
            (&[0xC4, 0xE2, 0x45, 0x8E, 0x31], Kind::D, true, 6, 7, mem(Some(1), None, 0, 0)),
            // vpmaskmovq [rax + 8 * r9], ymm4, ymm5
            (&[0xC4, 0xA2, 0xDD, 0x8E, 0x2C, 0xC8], Kind::Q, true, 5, 4, mem(Some(0), Some(9), 3, 0)),
            // vpmaskmovd [rsi + rcx], ymm12, ymm13
            (&[0xC4, 0x62, 0x1D, 0x8E, 0x2C, 0x0E], Kind::D, true, 13, 12, mem(Some(6), Some(1), 0, 0)),
            // vpmaskmovd [rsp + 4 * rcx + 0x4e0], ymm7, ymm0
            (&[0xC4, 0xE2, 0x45, 0x8E, 0x84, 0x8C, 0xE0, 0x04, 0x00, 0x00], Kind::D, true, 0, 7, mem(Some(4), Some(1), 2, 0x4E0)),
            // vpmaskmovd [rsp + 0x90], ymm1, ymm0;  vpmaskmovd [rsp], ymm1, ymm0
            (&[0xC4, 0xE2, 0x75, 0x8E, 0x84, 0x24, 0x90, 0x00, 0x00, 0x00], Kind::D, true, 0, 1, mem(Some(4), None, 0, 0x90)),
            (&[0xC4, 0xE2, 0x75, 0x8E, 0x04, 0x24], Kind::D, true, 0, 1, mem(Some(4), None, 0, 0)),
            // vmaskmovpd [rdx + 8 * rcx], ymm5, ymm6
            (&[0xC4, 0xE2, 0x55, 0x2F, 0x34, 0xCA], Kind::Pd, true, 6, 5, mem(Some(2), Some(1), 3, 0)),
            // vpmaskmovq [r10 + 8 * r11 - 8], ymm4, ymm5
            (&[0xC4, 0x82, 0xDD, 0x8E, 0x6C, 0xDA, 0xF8], Kind::Q, true, 5, 4, mem(Some(10), Some(11), 3, -8)),
            // vpmaskmovd [r13], xmm2, xmm1 (R13 needs a displacement byte);  vpmaskmovd [rbp + 0x10], xmm15, xmm9
            (&[0xC4, 0xC2, 0x69, 0x8E, 0x4D, 0x00], Kind::D, false, 1, 2, mem(Some(13), None, 0, 0)),
            (&[0xC4, 0x62, 0x01, 0x8E, 0x4D, 0x10], Kind::D, false, 9, 15, mem(Some(5), None, 0, 0x10)),
            // vpmaskmovd [rip + 0x1000], ymm1, ymm0
            (&[0xC4, 0xE2, 0x75, 0x8E, 0x05, 0x00, 0x10, 0x00, 0x00], Kind::D, true, 0, 1, Mem { disp: 0x1000, rip: true, ..NO_MEM }),
            // vmaskmovps [4 * rcx + 0x12345678], xmm1, xmm0;  vpmaskmovq [0x12345678], ymm1, ymm0
            (&[0xC4, 0xE2, 0x71, 0x2E, 0x04, 0x8D, 0x78, 0x56, 0x34, 0x12], Kind::Ps, false, 0, 1, mem(None, Some(1), 2, 0x1234_5678)),
            (&[0xC4, 0xE2, 0xF5, 0x8E, 0x04, 0x25, 0x78, 0x56, 0x34, 0x12], Kind::Q, true, 0, 1, mem(None, None, 0, 0x1234_5678)),
        ];
        for (code, kind, wide, src, mask, mem) in cases {
            let want = Store { kind: *kind, len: code.len() as u8, wide: *wide, src: *src, mask: *mask, mem: *mem };
            assert_eq!(decode(code), Some(want), "{code:02x?}");
            assert_eq!(insn_len(code), Some(code.len()), "{code:02x?}");
            assert_eq!(decode(&code[..code.len() - 1]), None, "{code:02x?} cut short");
        }
        assert_eq!(decode(&cases[0].0[..]).unwrap().elements(), 4);
        assert_eq!(decode(&cases[11].0[..]).unwrap().elements(), 4);
        assert_eq!(decode(&cases[14].0[..]).unwrap().elements(), 4);
        assert_eq!(decode(&cases[15].0[..]).unwrap().elements(), 4);
        assert_eq!(decode(&cases[1].0[..]).unwrap().elements(), 8);
        // the loads (8C, 2C, 2D), a register "destination", VMASKMOVPS with W1, another map: not stores
        for code in [[0xC4, 0xE2, 0x4D, 0x8C, 0x04, 0x90], [0xC4, 0xE2, 0x5D, 0x2C, 0x2C, 0x91], [0xC4, 0xE2, 0x4D, 0x8E, 0xC1, 0x90], [0xC4, 0xE2, 0xDD, 0x2E, 0x2C, 0x91], [0xC4, 0xE3, 0x4D, 0x8E, 0x04, 0x90], [0xC4, 0xE2, 0x4E, 0x8E, 0x04, 0x90]] {
            assert_eq!(decode(&code), None, "{code:02x?}");
        }
    }

    /// A helper as bytes: the prologue the unwind data describes, the address into R8 with the
    /// frame added to an RSP-relative displacement, both vectors to the stack, one guarded
    /// `mov` per element, the epilogue. And the call that reaches it.
    #[test]
    fn a_helper_is_the_address_two_spills_and_a_store_per_element() {
        // vpmaskmovq [rsp + r8 + 0x40], ymm7, ymm8
        let s = decode(&[0xC4, 0x22, 0xC5, 0x8E, 0x44, 0x04, 0x40]).unwrap();
        let mut out = Vec::new();
        assert!(helper(&s, 0, &mut out));
        let mut want = PROLOGUE.to_vec();
        want.extend_from_slice(&[0x4E, 0x8D, 0x84, 0x04, 0xA8, 0x00, 0x00, 0x00]); // lea r8, [rsp + r8 + 0x40 + 104]
        want.extend_from_slice(&[0xC5, 0x7E, 0x7F, 0x04, 0x24]); // vmovdqu [rsp], ymm8
        want.extend_from_slice(&[0xC5, 0xFE, 0x7F, 0x7C, 0x24, 0x20]); // vmovdqu [rsp + 0x20], ymm7
        for d in [0u8, 8, 16, 24] {
            want.extend_from_slice(&[0x48, 0x8B, 0x44, 0x24, 0x20 + d, 0x48, 0x99, 0x89, 0xD1, 0xE3, 0x09, 0x48, 0x8B, 0x44, 0x24, d, 0x49, 0x89, 0x40, d]);
        }
        want.extend_from_slice(&EPILOGUE);
        assert_eq!(out, want);
        // the largest: eight 4-byte elements and a RIP-relative operand (an absolute address)
        let s = decode(&[0xC4, 0xE2, 0x75, 0x8E, 0x05, 0x00, 0x10, 0x00, 0x00]).unwrap();
        let mut out = vec![0xCC; 3];
        assert!(helper(&s, 0x1_4000_2009, &mut out));
        assert_eq!(out[3 + 10..3 + 20], [0x49, 0xB8, 0x09, 0x30, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00]);
        assert!(out.len() - 3 <= HELPER_MAX && out.len() - 3 == 10 + 10 + 11 + 8 * 17 + 11);
        // vpmaskmovd [r13], xmm2, xmm1: R13 keeps its displacement byte, an XMM store has four elements
        let s = decode(&[0xC4, 0xC2, 0x69, 0x8E, 0x4D, 0x00]).unwrap();
        let mut out = Vec::new();
        assert!(helper(&s, 0, &mut out));
        assert_eq!(out[10..14], [0x4D, 0x8D, 0x45, 0x00]); // lea r8, [r13 + 0]
        assert_eq!(out[14..18], [0xC5, 0xFA, 0x7F, 0x0C]); // vmovdqu [rsp], xmm1
        assert_eq!(out.len(), 10 + 4 + 11 + 4 * 17 + 11);
        // an RSP-relative displacement that no longer fits is declined, and nothing is left behind
        let s = Store { mem: Mem { base: Some(4), disp: i32::MAX - 50, ..NO_MEM }, ..s };
        assert!(!helper(&s, 0, &mut out) && out.len() == 10 + 4 + 11 + 4 * 17 + 11);
        // the prologue the unwind data counts: 4 pushes (5 bytes) and the allocation end at byte 10
        assert_eq!((UNWIND_INFO[1], UNWIND_INFO[2], UNWIND_INFO[4]), (10, 5, 10));
        assert_eq!(FRAME, 8 + 4 * 8 + ((UNWIND_INFO[5] >> 4) as i64 + 1) * 8);
        let call = call_site(0x1_4007_9D67, 6, 0x1_4B07_0000).unwrap();
        assert_eq!(call[..6], [0xE8, 0x94, 0x62, 0xFF, 0x0A, 0x90]);
        assert_eq!(call_site(0x1_4007_9D67, 6, 0x2_4B07_0000), None, "out of reach");
        assert_eq!(call_site(0x1_4007_9D67, 4, 0x1_4B07_0000), None, "too short for a call");
    }

    /// The sites of a made-up image: a store in a function, one in a leaf function behind it,
    /// the same bytes inside another instruction, behind bytes that are no instruction, and
    /// far from any function.
    #[test]
    fn a_site_is_a_pattern_the_walk_from_its_function_arrives_at() {
        const STORE: [u8; 6] = [0xC4, 0xE2, 0x4D, 0x8E, 0x04, 0x90]; // vpmaskmovd [rax + 4 * rdx], ymm6, ymm0
        const OTHER: [u8; 5] = [0xC4, 0xE2, 0x45, 0x8E, 0x31]; // vpmaskmovd [rcx], ymm7, ymm6
        let mut image = vec![0xCCu8; 0x4_0000];
        let mut put = |at: usize, bytes: &[u8]| image[at..at + bytes.len()].copy_from_slice(bytes);
        // function 0x1000..0x1020: push rbx; sub rsp, 0x20; STORE; STORE; add rsp, 0x20; pop rbx; ret
        put(0x1000, &[0x53, 0x48, 0x83, 0xEC, 0x20]);
        put(0x1005, &STORE);
        put(0x100B, &STORE);
        put(0x1011, &[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]);
        // a leaf function behind its padding: xor eax, eax; OTHER; ret
        put(0x1030, &[0x31, 0xC0]);
        put(0x1032, &OTHER);
        put(0x1037, &[0xC3]);
        // function 0x2000..0x2040: mov rax, imm64 whose immediate is the pattern; then a real one
        put(0x2000, &[0x48, 0xB8]);
        put(0x2002, &STORE);
        put(0x2008, &[0x90, 0x90]);
        put(0x200A, &OTHER);
        put(0x200F, &[0xC3]);
        // function 0x3000..0x3040: an EVEX instruction before the pattern
        put(0x3000, &[0x62, 0xF1, 0x7C, 0x48, 0x10, 0x00]);
        put(0x3006, &STORE);
        // far behind the last function
        put(0x2_8000, &STORE);
        let functions = [(0x3000, 0x3040), (0x1000, 0x1017), (0x2000, 0x2040)];
        let p = plan(&image, 0x1_4000_0000, &[(0x1000, 0x3_F000)], &functions);
        let sites: Vec<(u32, bool, u32)> = p.sites.iter().map(|s| (s.rva, s.leaf, s.helper)).collect();
        let second = p.helpers[1].0;
        assert_eq!(sites, [(0x1005, false, 0), (0x100B, false, 0), (0x1032, true, second), (0x200A, false, second)]);
        assert_eq!(p.skipped, [(0x2002, Skip::NotAtBoundary), (0x3006, Skip::Undecodable(0x3000)), (0x2_8000, Skip::NoFunction)]);
        assert_eq!(p.helpers.len(), 2, "stores with the same bytes share a helper");
        assert!(second % 16 == 0 && p.helpers[0] == (0, 10 + 4 + 11 + 8 * 17 + 11) && p.helpers[1].1 as usize == p.code.len());
        // a section that ends inside the pattern, and one with no function before it
        assert!(plan(&image, 0, &[(0x1000, 8)], &functions).sites.is_empty());
        assert_eq!(plan(&image, 0, &[(0x1000, 0x100)], &[]).skipped.len(), 3);
    }
}

/// The helpers run against the instructions they stand for, on a CPU (or translator) that has
/// AVX2: a made-up register file is loaded, the code is called, the register file is taken
/// back, and the two runs are compared with each other and with what the instruction is
/// defined to do.
#[cfg(all(test, windows, target_arch = "x86_64"))]
pub(crate) mod exec {
    use super::*;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, RemoveVectoredExceptionHandler, EXCEPTION_POINTERS};
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, ResetWriteWatch, VirtualAlloc, VirtualFree, VirtualProtect, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, PAGE_READONLY, PAGE_READWRITE,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Everything a masked store could change but memory: the general registers (the hardware's
    /// order; RSP's slot is not used), the flags, YMM0–15 and 512 bytes of stack right above
    /// the call (`[rsp + 0x18]` at the instruction is its first byte).
    #[repr(C)]
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub(crate) struct State {
        pub gpr: [u64; 16],
        pub flags: u64,
        pub unused: u64,
        pub ymm: [[u8; 32]; 16],
        pub stack: [u8; 512],
    }

    /// `[rsp + STACK_AT]` at the instruction under test is `State::stack[0]`: above the return
    /// address and the two pointers `xcl_maskmov_test_run` keeps there.
    pub(crate) const STACK_AT: i32 = 0x18;

    std::arch::global_asm!(
        r#"
        .globl xcl_maskmov_test_run
    xcl_maskmov_test_run:
        push rbx
        push rbp
        push rsi
        push rdi
        push r12
        push r13
        push r14
        push r15
        sub rsp, 0xA8
        vmovdqu [rsp + 0x00], xmm6
        vmovdqu [rsp + 0x10], xmm7
        vmovdqu [rsp + 0x20], xmm8
        vmovdqu [rsp + 0x30], xmm9
        vmovdqu [rsp + 0x40], xmm10
        vmovdqu [rsp + 0x50], xmm11
        vmovdqu [rsp + 0x60], xmm12
        vmovdqu [rsp + 0x70], xmm13
        vmovdqu [rsp + 0x80], xmm14
        vmovdqu [rsp + 0x90], xmm15
        sub rsp, 0x200
        mov r10, rcx
        mov r11, rdx
        lea rsi, [r10 + 0x290]
        mov rdi, rsp
        mov ecx, 0x200
        cld
        rep movsb
        push r11
        push r10
        vmovdqu ymm0, [r10 + 0x090]
        vmovdqu ymm1, [r10 + 0x0B0]
        vmovdqu ymm2, [r10 + 0x0D0]
        vmovdqu ymm3, [r10 + 0x0F0]
        vmovdqu ymm4, [r10 + 0x110]
        vmovdqu ymm5, [r10 + 0x130]
        vmovdqu ymm6, [r10 + 0x150]
        vmovdqu ymm7, [r10 + 0x170]
        vmovdqu ymm8, [r10 + 0x190]
        vmovdqu ymm9, [r10 + 0x1B0]
        vmovdqu ymm10, [r10 + 0x1D0]
        vmovdqu ymm11, [r10 + 0x1F0]
        vmovdqu ymm12, [r10 + 0x210]
        vmovdqu ymm13, [r10 + 0x230]
        vmovdqu ymm14, [r10 + 0x250]
        vmovdqu ymm15, [r10 + 0x270]
        push qword ptr [r10 + 0x80]
        popfq
        mov rax, [r10 + 0x00]
        mov rcx, [r10 + 0x08]
        mov rdx, [r10 + 0x10]
        mov rbx, [r10 + 0x18]
        mov rbp, [r10 + 0x28]
        mov rsi, [r10 + 0x30]
        mov rdi, [r10 + 0x38]
        mov r8, [r10 + 0x40]
        mov r9, [r10 + 0x48]
        mov r11, [r10 + 0x58]
        mov r12, [r10 + 0x60]
        mov r13, [r10 + 0x68]
        mov r14, [r10 + 0x70]
        mov r15, [r10 + 0x78]
        mov r10, [r10 + 0x50]
        call qword ptr [rsp + 8]
        xchg r10, [rsp]
        mov [r10 + 0x00], rax
        mov [r10 + 0x08], rcx
        mov [r10 + 0x10], rdx
        mov [r10 + 0x18], rbx
        mov [r10 + 0x28], rbp
        mov [r10 + 0x30], rsi
        mov [r10 + 0x38], rdi
        mov [r10 + 0x40], r8
        mov [r10 + 0x48], r9
        mov [r10 + 0x58], r11
        mov [r10 + 0x60], r12
        mov [r10 + 0x68], r13
        mov [r10 + 0x70], r14
        mov [r10 + 0x78], r15
        pop rax
        mov [r10 + 0x50], rax
        pushfq
        pop rax
        mov [r10 + 0x80], rax
        vmovdqu [r10 + 0x090], ymm0
        vmovdqu [r10 + 0x0B0], ymm1
        vmovdqu [r10 + 0x0D0], ymm2
        vmovdqu [r10 + 0x0F0], ymm3
        vmovdqu [r10 + 0x110], ymm4
        vmovdqu [r10 + 0x130], ymm5
        vmovdqu [r10 + 0x150], ymm6
        vmovdqu [r10 + 0x170], ymm7
        vmovdqu [r10 + 0x190], ymm8
        vmovdqu [r10 + 0x1B0], ymm9
        vmovdqu [r10 + 0x1D0], ymm10
        vmovdqu [r10 + 0x1F0], ymm11
        vmovdqu [r10 + 0x210], ymm12
        vmovdqu [r10 + 0x230], ymm13
        vmovdqu [r10 + 0x250], ymm14
        vmovdqu [r10 + 0x270], ymm15
        pop rax
        lea rdi, [r10 + 0x290]
        mov rsi, rsp
        mov ecx, 0x200
        cld
        rep movsb
        add rsp, 0x200
        vmovdqu xmm6, [rsp + 0x00]
        vmovdqu xmm7, [rsp + 0x10]
        vmovdqu xmm8, [rsp + 0x20]
        vmovdqu xmm9, [rsp + 0x30]
        vmovdqu xmm10, [rsp + 0x40]
        vmovdqu xmm11, [rsp + 0x50]
        vmovdqu xmm12, [rsp + 0x60]
        vmovdqu xmm13, [rsp + 0x70]
        vmovdqu xmm14, [rsp + 0x80]
        vmovdqu xmm15, [rsp + 0x90]
        vzeroupper
        add rsp, 0xA8
        pop r15
        pop r14
        pop r13
        pop r12
        pop rdi
        pop rsi
        pop rbp
        pop rbx
        ret
    "#
    );

    extern "C" {
        /// Loads `state`, calls `code` (which must return with RSP as it found it), stores
        /// the registers back.
        fn xcl_maskmov_test_run(state: *mut State, code: *const u8);
    }

    /// `code` run on a copy of `state`.
    pub(crate) unsafe fn run(code: usize, state: &State) -> State {
        let mut s = state.clone();
        xcl_maskmov_test_run(&mut s, code as *const u8);
        s
    }

    pub(crate) fn has_avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    /// A small generator (xorshift): the same made-up values on every run.
    pub(crate) struct Rng(pub u64);

    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        pub fn fill(&mut self, b: &mut [u8]) {
            for c in b.chunks_mut(8) {
                let v = self.next().to_le_bytes();
                c.copy_from_slice(&v[..c.len()]);
            }
        }

        /// Registers, vectors and stack bytes from the generator; the arithmetic flags all set
        /// or all clear.
        pub fn state(&mut self, flags_set: bool) -> State {
            let mut s = State { gpr: [0; 16], flags: 0x202 | if flags_set { 0x8D5 } else { 0 }, unused: 0, ymm: [[0; 32]; 16], stack: [0; 512] };
            for g in &mut s.gpr {
                *g = self.next();
            }
            s.gpr[4] = 0;
            for y in &mut s.ymm {
                self.fill(y);
            }
            self.fill(&mut s.stack);
            s
        }
    }

    /// Executable memory that is written once and never again (a translator caches what it
    /// has run), with writable memory behind it for RIP-relative operands.
    pub(crate) struct Code {
        base: usize,
        used: usize,
    }

    const CODE_SIZE: usize = 0x10_0000;
    const DATA_AT: usize = 0xF_0000;

    impl Code {
        pub unsafe fn new() -> Code {
            let base = VirtualAlloc(std::ptr::null(), CODE_SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) as usize;
            assert!(base != 0);
            let mut old = 0;
            VirtualProtect((base + DATA_AT) as *const c_void, CODE_SIZE - DATA_AT, PAGE_READWRITE, &mut old);
            Code { base, used: 0 }
        }

        /// Writable bytes within a RIP-relative operand's reach of the code.
        pub fn data(&self) -> usize {
            self.base + DATA_AT
        }

        pub unsafe fn put(&mut self, bytes: &[u8]) -> usize {
            let at = self.base + self.used;
            assert!(self.used + bytes.len() <= DATA_AT);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len());
            self.used = (self.used + bytes.len() + 15) & !15;
            at
        }

        /// Where the next `put` goes.
        pub fn next(&self) -> usize {
            self.base + self.used
        }

        /// `store; ret` as the CPU runs it, and the same with the store replaced as `plan`'s
        /// caller does it: `call helper; nop…; ret`. A RIP-relative store is made to point at
        /// `data() + 0x40` from both places.
        pub unsafe fn pair(&mut self, s: &Store) -> (usize, usize) {
            let at = |s: &Store, site: usize, data: usize| {
                let mut s = *s;
                if s.mem.rip {
                    s.mem.disp = (data as i64 - (site as i64 + 9)) as i32;
                }
                s
            };
            let data = self.data() + 0x40;
            let mut code = encode(&at(s, self.next(), data));
            code.push(0xC3);
            let original = self.put(&code);
            let site = self.next();
            let s = at(s, site, data);
            let mut h = Vec::new();
            assert!(helper(&s, (site + s.len as usize) as u64, &mut h));
            let helper_at = site + 16;
            let mut code = call_site(site as u64, s.len, helper_at as u64).unwrap()[..s.len as usize].to_vec();
            code.push(0xC3);
            code.resize(16, 0xCC);
            code.extend_from_slice(&h);
            (original, self.put(&code))
        }
    }

    impl Drop for Code {
        fn drop(&mut self) {
            unsafe { VirtualFree(self.base as *mut c_void, 0, MEM_RELEASE) };
        }
    }

    /// The instruction's bytes (three-byte VEX), the reverse of `decode`.
    pub(crate) fn encode(s: &Store) -> Vec<u8> {
        let Mem { base, index, scale, disp, rip } = s.mem;
        let (r, x, b) = (s.src >> 3, index.map_or(0, |i| i >> 3), base.map_or(0, |b| b >> 3));
        let (op, w) = match s.kind {
            Kind::Ps => (0x2E, 0),
            Kind::Pd => (0x2F, 0),
            Kind::D => (0x8E, 0),
            Kind::Q => (0x8E, 1),
        };
        let mut v = vec![0xC4, (r ^ 1) << 7 | (x ^ 1) << 6 | (b ^ 1) << 5 | 2, w << 7 | (!s.mask & 15) << 3 | (s.wide as u8) << 2 | 1, op];
        let reg = (s.src & 7) << 3;
        let sib = scale << 6 | index.map_or(4, |i| i & 7) << 3;
        match base {
            _ if rip => {
                v.push(reg | 5);
                v.extend_from_slice(&disp.to_le_bytes());
            }
            None => {
                v.extend_from_slice(&[reg | 4, sib | 5]);
                v.extend_from_slice(&disp.to_le_bytes());
            }
            Some(bb) => {
                let md = if disp == 0 && bb & 7 != 5 {
                    0
                } else if (-128..=127).contains(&disp) {
                    1
                } else {
                    2
                };
                if index.is_some() || bb & 7 == 4 {
                    v.extend_from_slice(&[md << 6 | reg | 4, sib | (bb & 7)]);
                } else {
                    v.push(md << 6 | reg | (bb & 7));
                }
                match md {
                    1 => v.push(disp as u8),
                    2 => v.extend_from_slice(&disp.to_le_bytes()),
                    _ => {}
                }
            }
        }
        v
    }

    /// What the store leaves in memory that held `before`, by the instruction's definition:
    /// the source's element where the mask's element has its top bit set, nothing else.
    pub(crate) fn defined(s: &Store, state: &State, before: &[u8]) -> Vec<u8> {
        let (mut after, n) = (before.to_vec(), s.kind.element());
        for i in 0..s.elements() {
            if state.ymm[s.mask as usize][i * n + n - 1] & 0x80 != 0 {
                after[i * n..i * n + n].copy_from_slice(&state.ymm[s.src as usize][i * n..i * n + n]);
            }
        }
        after
    }

    /// Sets the mask register of `s` in `state`: element `i` selected when bit `i` of `bits` is
    /// set; every other bit of the mask stays as it was (they do not count).
    pub(crate) fn set_mask(s: &Store, state: &mut State, bits: u32) {
        let n = s.kind.element();
        for i in 0..s.elements() {
            let top = &mut state.ymm[s.mask as usize][i * n + n - 1];
            *top = (*top & 0x7F) | if bits >> i & 1 != 0 { 0x80 } else { 0 };
        }
    }

    /// Points the operand of `s` at `target` by its base and index registers (an index is
    /// given the value `index`). An RSP-relative operand is left alone: it is what it is.
    pub(crate) fn point(s: &Store, state: &mut State, target: usize, index: i64) {
        let Mem { base, index: ix, scale, disp, rip } = s.mem;
        if rip || base == Some(4) {
            return;
        }
        let mut rest = (target as i64).wrapping_sub(disp as i64);
        match (base, ix) {
            (Some(b), Some(x)) => {
                state.gpr[x as usize] = index as u64;
                state.gpr[b as usize] = rest.wrapping_sub(index << scale) as u64;
            }
            (Some(b), None) => state.gpr[b as usize] = rest as u64,
            (None, Some(x)) => {
                assert!(rest & ((1 << scale) - 1) == 0);
                rest >>= scale;
                state.gpr[x as usize] = rest as u64;
            }
            (None, None) => assert_eq!(rest, 0, "an absolute operand has to be the target"),
        }
    }

    fn mem(base: Option<u8>, index: Option<u8>, scale: u8, disp: i32) -> Mem {
        Mem { base, index, scale, disp, rip: false }
    }

    /// Operand shapes: every one a retail game has (see `decodes_the_masked_stores`), each
    /// register the helper itself uses (RAX, RCX, RDX, R8) as base and as index, RSP with and
    /// without an index and with displacements on both sides of a byte, RBP / R13 (which need
    /// a displacement), R12 (which needs a SIB byte), RIP-relative, an index without a base.
    fn shapes() -> Vec<Mem> {
        let mut v = vec![
            mem(Some(0), None, 0, 0),
            mem(Some(1), None, 0, 0),
            mem(Some(0), Some(2), 2, 0),
            mem(Some(1), Some(2), 2, 0),
            mem(Some(2), Some(1), 3, 0),
            mem(Some(8), Some(0), 0, 0x20),
            mem(Some(2), Some(8), 3, -8),
            mem(Some(12), Some(8), 2, 0),
            mem(Some(0), Some(15), 2, 0x20),
            mem(Some(6), Some(1), 0, 0),
            mem(Some(10), Some(11), 3, -8),
            mem(Some(5), None, 0, 0),
            mem(Some(5), None, 0, 0x10),
            mem(Some(13), None, 0, 0),
            mem(Some(13), Some(9), 1, 0x1234),
            mem(Some(12), None, 0, 0),
            mem(Some(3), Some(12), 0, -0x80),
            mem(Some(7), Some(14), 2, 0x7F),
            mem(Some(14), Some(5), 3, 0x80),
            mem(None, Some(1), 2, 0x100),
            mem(None, Some(8), 3, -0x1000),
            // [rsp + STACK_AT + 0x40] and around it: the store goes to `State::stack[0x40..]`
            mem(Some(4), None, 0, STACK_AT + 0x40),
            mem(Some(4), Some(8), 0, 0x40),
            mem(Some(4), Some(1), 2, 0x20),
            mem(Some(4), Some(2), 3, 0x70 - 0x100),
            mem(Some(4), Some(6), 0, 0),
            mem(Some(4), Some(15), 2, 0x4E0),
        ];
        v.push(Mem { rip: true, ..mem(None, None, 0, 0) });
        v
    }

    /// The index an RSP-relative shape needs for its store to land on `State::stack[0x40..]`.
    fn stack_index(m: &Mem) -> u64 {
        ((STACK_AT + 0x40 - m.disp) as i64 >> m.scale) as u64
    }

    /// Every kind and width over every operand shape, with the registers taken in turn, for
    /// every mask (256 for eight elements) and both settings of the flags: the helper leaves
    /// the registers, the flags, the vectors, the stack and memory exactly as the instruction
    /// does, and memory is what the instruction is defined to leave.
    #[test]
    fn a_helper_does_what_the_instruction_does() {
        if !has_avx2() {
            eprintln!("SKIPPED: this CPU has no AVX2");
            return;
        }
        unsafe {
            let mut code = Code::new();
            let buf = VirtualAlloc(std::ptr::null(), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
            let (mut runs, mut k) = (0usize, 0usize);
            for kind in Kind::ALL {
                for wide in [true, false] {
                    for m in shapes() {
                        // the registers in turn; one shape in seven stores the mask register itself
                        k += 1;
                        let src = (k * 5 % 16) as u8;
                        let s = Store { kind, len: 0, wide, src, mask: if k % 7 == 0 { src } else { (k * 11 % 16) as u8 }, mem: m };
                        let bytes = encode(&s);
                        let s = Store { len: bytes.len() as u8, ..s };
                        assert_eq!(decode(&bytes), Some(s), "{bytes:02x?}");
                        let (original, replaced) = code.pair(&s);
                        let on_stack = m.base == Some(4);
                        let target = if m.rip { code.data() + 0x40 } else { buf + 0x40 };
                        for bits in 0..1u32 << s.elements() {
                            let mut st = rng.state(bits & 1 != 0);
                            set_mask(&s, &mut st, bits);
                            if on_stack {
                                if let Some(x) = m.index {
                                    st.gpr[x as usize] = stack_index(&m);
                                }
                            } else {
                                point(&s, &mut st, target, [3, -2, 0x10][bits as usize % 3]);
                            }
                            let mut before = [0u8; 0x80];
                            rng.fill(&mut before);
                            let mem_at = (target - 0x40) as *mut [u8; 0x80];
                            let mut after = [[0u8; 0x80]; 2];
                            let mut out = Vec::new();
                            for (which, entry) in [original, replaced].into_iter().enumerate() {
                                if !on_stack {
                                    *mem_at = before;
                                }
                                out.push(run(entry, &st));
                                if !on_stack {
                                    after[which] = *mem_at;
                                }
                                runs += 1;
                            }
                            let what = format!("{} {bytes:02x?} mask {bits:#x}", kind.name());
                            assert_eq!(out[0], out[1], "registers, flags, vectors, stack: {what}");
                            let mut want = st.clone();
                            if on_stack {
                                let stored = defined(&s, &st, &st.stack[0x40..0x60]);
                                want.stack[0x40..0x60].copy_from_slice(&stored);
                            } else {
                                assert_eq!(after[0], after[1], "memory: {what}");
                                let mut m = before;
                                let stored = defined(&s, &st, &before[0x40..0x60]);
                                m[0x40..0x60].copy_from_slice(&stored);
                                assert_eq!(after[1], m, "memory as defined: {what}");
                            }
                            assert_eq!(out[1], want, "nothing but the selected elements changes: {what}");
                        }
                    }
                }
            }
            VirtualFree(buf as *mut c_void, 0, MEM_RELEASE);
            eprintln!("{runs} runs, {k} stores");
        }
    }

    /// A store whose last elements are masked out ends at the last byte of a mapping: neither
    /// the instruction nor the helper touches the page behind it, which is not there.
    #[test]
    fn a_masked_out_tail_may_end_where_the_mapping_ends() {
        if !has_avx2() {
            eprintln!("SKIPPED: this CPU has no AVX2");
            return;
        }
        unsafe {
            let mut code = Code::new();
            let pages = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(!VirtualAlloc(pages as *const c_void, 0x1000, MEM_COMMIT, PAGE_READWRITE).is_null());
            let mut rng = Rng(0x1234_5678_9ABC_DEF1);
            for kind in Kind::ALL {
                for wide in [true, false] {
                    let s = Store { kind, len: 0, wide, src: 9, mask: 3, mem: mem(Some(1), Some(2), 0, 0x10) };
                    let s = Store { len: encode(&s).len() as u8, ..s };
                    let (original, replaced) = code.pair(&s);
                    // the first `inside` elements are in the mapping, and none, some or all of them selected
                    for inside in 0..=s.elements() {
                        for bits in [0, 1, (1u32 << inside) - 1, (1u32 << inside) >> 1] {
                            let bits = bits & ((1 << inside) - 1);
                            let target = pages + 0x1000 - inside * kind.element();
                            let mut st = rng.state(false);
                            set_mask(&s, &mut st, bits);
                            point(&s, &mut st, target, 7);
                            let tail = (pages + 0x1000 - 0x40) as *mut [u8; 0x40];
                            let mut before = [0u8; 0x40];
                            rng.fill(&mut before);
                            let mut after = Vec::new();
                            for entry in [original, replaced] {
                                *tail = before;
                                assert_eq!(run(entry, &st), st);
                                after.push(*tail);
                            }
                            assert_eq!(after[0], after[1], "{} with {inside} elements inside, mask {bits:#x}", kind.name());
                            let mut want = [0u8; 0x60];
                            want[..0x40].copy_from_slice(&before);
                            let at = 0x40 - inside * kind.element();
                            let stored = defined(&s, &st, &want[at..at + 0x20]);
                            want[at..at + 0x20].copy_from_slice(&stored);
                            assert_eq!(after[1][..], want[..0x40]);
                        }
                    }
                }
            }
            VirtualFree(pages as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// Pages a handler makes writable when they are written: (start, end), how often it did,
    /// where the last faulting instruction was.
    static REPAIR: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

    unsafe extern "system" fn repair(info: *mut EXCEPTION_POINTERS) -> i32 {
        let rec = &*(*info).ExceptionRecord;
        let at = rec.ExceptionInformation[1];
        if rec.ExceptionCode as u32 != 0xC000_0005 || rec.ExceptionInformation[0] != 1 || !(REPAIR[0].load(Ordering::Relaxed)..REPAIR[1].load(Ordering::Relaxed)).contains(&at) {
            return 0;
        }
        let mut old = 0;
        VirtualProtect((at & !0xFFF) as *const c_void, 0x1000, PAGE_READWRITE, &mut old);
        REPAIR[2].fetch_add(1, Ordering::Relaxed);
        REPAIR[3].store(rec.ExceptionAddress as usize, Ordering::Relaxed);
        -1 // EXCEPTION_CONTINUE_EXECUTION
    }

    /// The store that ends the process under Rosetta 2 when the instruction makes it: it starts
    /// on a page that can be written and goes on into one that cannot. The helper's fault is
    /// on an ordinary `mov`: a handler makes the page writable, the store is resumed and
    /// completes – with a read-only page and an exception handler of the test's, and with a
    /// write watch, where (under Wine) the system itself is the handler.
    #[test]
    fn a_helper_resumes_after_a_fault_on_a_later_element() {
        if !has_avx2() {
            eprintln!("SKIPPED: this CPU has no AVX2");
            return;
        }
        unsafe {
            let mut code = Code::new();
            let mut rng = Rng(0xFEED_FACE_CAFE_BEEF);
            let pages = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize;
            let watched = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert!(pages != 0 && watched != 0);
            REPAIR[0].store(pages, Ordering::Relaxed);
            REPAIR[1].store(pages + 0x2000, Ordering::Relaxed);
            let handler = AddVectoredExceptionHandler(1, Some(repair));
            for kind in Kind::ALL {
                for wide in [true, false] {
                    let s = Store { kind, len: 0, wide, src: 2, mask: 14, mem: mem(Some(0), Some(2), 2, 0) };
                    let s = Store { len: encode(&s).len() as u8, ..s };
                    let (_, replaced) = code.pair(&s);
                    let all = (1u32 << s.elements()) - 1;
                    // the boundary before each element but the first; every element selected, or
                    // only the ones behind the boundary (the first of them is then the first stored)
                    for first_behind in 1..s.elements() {
                        for bits in [all, all & !((1 << first_behind) - 1)] {
                            let what = format!("{} with element {first_behind} first on the second page, mask {bits:#x}", kind.name());
                            let offset = 0x1000 - first_behind * kind.element();
                            let mut st = rng.state(true);
                            set_mask(&s, &mut st, bits);
                            // a read-only second page, repaired by `repair`
                            let window = (pages + 0x1000 - 0x40) as *mut [u8; 0x80];
                            let mut before = [0u8; 0x80];
                            rng.fill(&mut before);
                            *window = before;
                            let mut old = 0;
                            VirtualProtect((pages + 0x1000) as *const c_void, 0x1000, PAGE_READONLY, &mut old);
                            let faults = REPAIR[2].load(Ordering::Relaxed);
                            point(&s, &mut st, pages + offset, 5);
                            assert_eq!(run(replaced, &st), st, "{what}");
                            assert_eq!(REPAIR[2].load(Ordering::Relaxed), faults + 1, "one fault, repaired: {what}");
                            assert!((replaced + 16..replaced + 16 + HELPER_MAX).contains(&REPAIR[3].load(Ordering::Relaxed)), "the fault is in the helper");
                            let at = 0x40 - first_behind * kind.element();
                            let mut want = before;
                            let stored = defined(&s, &st, &before[at..at + 0x20]);
                            want[at..at + 0x20].copy_from_slice(&stored);
                            assert_eq!(*window, want, "{what}");
                            // a write watch: both pages watched again, the first written (so it is
                            // writable), then the store across the boundary
                            let window = (watched + 0x1000 - 0x40) as *mut [u8; 0x80];
                            *window = before;
                            assert_eq!(ResetWriteWatch(watched as *const c_void, 0x2000), 0);
                            (watched as *mut u8).add(0x1000 - 0x40).write_volatile(before[0]);
                            point(&s, &mut st, watched + offset, 5);
                            assert_eq!(run(replaced, &st), st, "{what}");
                            assert_eq!(*window, want, "write-watched: {what}");
                            let (mut written, mut count, mut grain) = ([std::ptr::null_mut::<c_void>(); 4], 4usize, 0u32);
                            assert_eq!(GetWriteWatch(0, watched as *const c_void, 0x2000, written.as_mut_ptr(), &mut count, &mut grain), 0);
                            assert_eq!((count, written[0] as usize, written[1] as usize), (2, watched, watched + 0x1000), "both pages are reported as written: {what}");
                        }
                    }
                }
            }
            RemoveVectoredExceptionHandler(handler);
            VirtualFree(pages as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(watched as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// What the helpers are for, to be run alone and on purpose (`--ignored`): the instruction
    /// itself across the boundary of a written and an unwritten page of write-watched memory.
    /// A CPU completes it. Under Rosetta 2 with a write watch kept by page protection (Wine on
    /// macOS) the process ends here with "rosetta error: unexpectedly need to EmulateForward
    /// on a synchronous exception" and the test never reports.
    #[test]
    #[ignore = "ends the process under Rosetta 2: that is the finding"]
    fn the_instruction_itself_across_a_watched_page_boundary() {
        if !has_avx2() {
            return;
        }
        unsafe {
            let mut code = Code::new();
            let watched = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            let s = Store { kind: Kind::D, len: 0, wide: true, src: 0, mask: 6, mem: mem(Some(0), Some(2), 2, 0) };
            let s = Store { len: encode(&s).len() as u8, ..s };
            let (original, _) = code.pair(&s);
            let mut st = Rng(7).state(false);
            set_mask(&s, &mut st, 0xFF);
            assert_eq!(ResetWriteWatch(watched as *const c_void, 0x2000), 0);
            // first with its first element on the unwritten page: resumed on every host
            point(&s, &mut st, watched + 0x1000, 5);
            assert_eq!(run(original, &st), st);
            eprintln!("a fault on the first element was resumed; now one on the fourth");
            assert_eq!(ResetWriteWatch(watched as *const c_void, 0x2000), 0);
            (watched as *mut u8).add(0x1000 - 0x40).write_volatile(1);
            point(&s, &mut st, watched + 0x1000 - 12, 5);
            assert_eq!(run(original, &st), st);
            assert_eq!(((watched + 0x1000 - 12) as *const [u8; 32]).read(), st.ymm[0]);
            eprintln!("a fault on the fourth element was resumed: this host runs the instruction as a CPU does");
        }
    }
}
