//! Streamed textures whose video memory follows what the title has loaded.
//!
//! A retail game places its streamed textures at full size and maps memory under their levels
//! as it loads them, the small ones first. A committed PC texture takes video memory for every
//! level at once: in one loading run 22,000 textures stood for 7.9 GB of which 1.3 GB had
//! memory behind it. On a host whose video memory is the system's memory (Apple's D3DMetal)
//! that is wired memory nothing uses – on a 24 GB machine the run then spent its time
//! compressing and decompressing memory (2 – 4 GB a second) and showed two frames in 30 s.
//! Here such a texture is a reserved (tiled) resource of its own, and a level's tiles are
//! mapped when an upload first has texels for it (`map_for`, before the copy, on the queue the
//! copy runs on). Each texture has tiles of its own – no two share any, which D3DMetal does
//! not do for textures (`vram::on_pages`).
//!
//! Not done: a level the title unloads keeps its tiles until the texture goes; arrays, cubes
//! and volumes stay committed textures.
//! `XCL_TEXTURE_LEVEL_TILES=1` / `0`; unset: on a native host.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu::texture::Subresource;
use crate::util::HRESULT;

/// tiles of a pool heap (256 MiB)
const HEAP_TILES: u32 = 4096;
/// a freed run of tiles is not given out again before this (work on its old texture may still run)
const COOL: Duration = Duration::from_secs(2);

/// a run of tiles: (heap, first tile, count)
type Run = (u32, u32, u32);

struct Tex {
    dev: usize,
    /// tiles of each standard mip: (width, height, depth) in tiles
    standard: Vec<(u32, u16, u16)>,
    /// tiles of the packed mips (0: none)
    packed: u32,
    /// the tiles of each standard mip, then of the packed mips (None: not mapped)
    mapped: Vec<Option<Run>>,
}

#[derive(Default)]
struct Pool {
    heaps: Vec<(ID3D12Heap, u32, u32)>,
    free: HashMap<u32, Vec<(u32, u32, Instant)>>,
}

#[derive(Default)]
struct State {
    texs: HashMap<usize, Tex>,
    pools: HashMap<usize, Pool>,
    mapped_tiles: u64,
    all_tiles: u64,
}

// SAFETY: heaps are free-threaded COM objects
unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);

pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = match std::env::var("XCL_TEXTURE_LEVEL_TILES") {
            Ok(v) => v.trim() != "0",
            Err(_) => super::real::native_host(),
        };
        if on {
            crate::xlog_info!("D3D12.X: streamed textures are reserved resources whose levels get video memory when the title has loaded them (XCL_TEXTURE_LEVEL_TILES=0: committed textures)");
        }
        on
    })
}

#[cfg(test)]
/// (tiles mapped, the textures' tiles in all, textures)
pub fn stats() -> (u64, u64, usize) {
    STATE.lock().unwrap().as_ref().map_or((0, 0, 0), |s| (s.mapped_tiles, s.all_tiles, s.texs.len()))
}

/// A sampled 2D texture of the title as a reserved resource with nothing mapped. None: not this
/// way (the caller makes a committed texture).
pub(super) unsafe fn create(this: *mut c_void, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    if !enabled() {
        return None;
    }
    reserve(this, desc, state, riid, ppv)
}

/// `create` without the switch.
unsafe fn reserve(this: *mut c_void, desc: &D3D12_RESOURCE_DESC, state: u32, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || desc.DepthOrArraySize != 1
        || desc.SampleDesc.Count != 1
        || desc.Flags != D3D12_RESOURCE_FLAG_NONE
        || desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN
        || desc.MipLevels < 2
        || desc.Width < 512
        || desc.Height < 512
    {
        return None;
    }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, 0, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("D3D12.X: streamed texture {}x{} format {}: not reserved ({hr:#x}) – committed", desc.Width, desc.Height, desc.Format.0);
        }
        *ppv = std::ptr::null_mut();
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mips = desc.MipLevels as u32;
    let (mut all, mut packed, mut n) = (0u32, D3D12_PACKED_MIP_INFO::default(), mips);
    let mut subs = vec![D3D12_SUBRESOURCE_TILING::default(); mips as usize];
    dev.GetResourceTiling(res, Some(&mut all), Some(&mut packed), None, Some(&mut n), 0, subs.as_mut_ptr());
    let n_standard = (packed.NumStandardMips as usize).min(mips as usize);
    let standard: Vec<(u32, u16, u16)> = subs[..n_standard].iter().map(|t| (t.WidthInTiles, t.HeightInTiles, t.DepthInTiles)).collect();
    if n_standard == 0 || standard.iter().any(|&(w, h, d)| w == 0 || h == 0 || d == 0) {
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    s.all_tiles += all as u64;
    s.texs.insert(raw as usize, Tex { dev: this as usize, mapped: vec![None; n_standard + 1], standard, packed: packed.NumTilesForPackedMips });
    Some(hr)
}

unsafe fn take(pool: &mut Pool, dev: &ID3D12Device, n: u32, now: Instant) -> Option<Run> {
    if let Some(list) = pool.free.get_mut(&n) {
        if let Some(i) = list.iter().position(|&(_, _, at)| now.duration_since(at) >= COOL) {
            let (h, t, _) = list.swap_remove(i);
            return Some((h, t, n));
        }
    }
    if let Some((i, (_, used, _))) = pool.heaps.iter_mut().enumerate().rev().find(|(_, (_, used, size))| *used + n <= *size) {
        let t = *used;
        *used += n;
        return Some((i as u32, t, n));
    }
    let size = HEAP_TILES.max(n);
    let hd = D3D12_HEAP_DESC {
        SizeInBytes: size as u64 * 0x1_0000,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&hd, &mut heap)) {
        crate::xlog_warn!("D3D12.X: a heap for streamed textures' levels: {e}");
        return None;
    }
    pool.heaps.push((heap?, n, size));
    Some((pool.heaps.len() as u32 - 1, 0, n))
}

/// Map slot `slot` (a standard mip, or the packed mips) of `res` if it is not.
unsafe fn map_slot(s: &mut State, queue: &ID3D12CommandQueue, res: &ID3D12Resource, slot: usize) {
    let key = res.as_raw() as usize;
    let Some(t) = s.texs.get(&key) else { return };
    if slot >= t.mapped.len() || t.mapped[slot].is_some() {
        return;
    }
    let (coord, region, n) = match t.standard.get(slot) {
        Some(&(w, h, d)) => {
            let n = w * h as u32 * d as u32;
            (D3D12_TILED_RESOURCE_COORDINATE { Subresource: slot as u32, ..Default::default() }, D3D12_TILE_REGION_SIZE { NumTiles: n, UseBox: true.into(), Width: w, Height: h, Depth: d }, n)
        }
        None if t.packed > 0 => (D3D12_TILED_RESOURCE_COORDINATE { Subresource: t.standard.len() as u32, ..Default::default() }, D3D12_TILE_REGION_SIZE { NumTiles: t.packed, UseBox: false.into(), ..Default::default() }, t.packed),
        None => return,
    };
    let d = t.dev as *mut c_void;
    let Some(dev) = ID3D12Device::from_raw_borrowed(&d) else { return };
    let pool = s.pools.entry(t.dev).or_default();
    let Some(run) = take(pool, dev, n, Instant::now()) else { return };
    let (flags, offsets, counts) = ([D3D12_TILE_RANGE_FLAG_NONE], [run.1], [n]);
    queue.UpdateTileMappings(res, 1, Some(&coord), Some(&region), &pool.heaps[run.0 as usize].0, 1, Some(flags.as_ptr()), Some(offsets.as_ptr()), Some(counts.as_ptr()), D3D12_TILE_MAPPING_FLAG_NONE);
    s.texs.get_mut(&key).unwrap().mapped[slot] = Some(run);
    let before = s.mapped_tiles;
    s.mapped_tiles += n as u64;
    if before >> 12 != s.mapped_tiles >> 12 {
        crate::xlog_info!("D3D12.X: streamed textures' levels: {} MB of video memory mapped of the {} MB that {} textures stand for", s.mapped_tiles >> 4, s.all_tiles >> 4, s.texs.len());
    }
}

/// Before a batch of uploads runs on `queue`: every level that has texels gets its tiles. A
/// level of zeros has none: its place in the title's memory has no memory behind it yet (read as
/// zeros), as for `vram::map_levels`; its copy goes to tiles that are not mapped.
pub(crate) unsafe fn map_for<T>(queue: &ID3D12CommandQueue, items: &[(ID3D12Resource, Vec<Subresource>, T)]) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    for (res, subs, _) in items {
        let Some(n_standard) = s.texs.get(&(res.as_raw() as usize)).map(|t| t.standard.len()) else { continue };
        for (i, sub) in subs.iter().enumerate() {
            if !super::title_textures::all_zero(&sub.data) {
                map_slot(s, queue, res, i.min(n_standard));
            }
        }
    }
}

/// The title's GPU work writes `res`: all of it is mapped.
pub(super) unsafe fn map_all(queue: &ID3D12CommandQueue, res: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let Some(slots) = s.texs.get(&res).map(|t| t.mapped.len()) else { return };
    let raw = res as *mut c_void;
    let Some(r) = ID3D12Resource::from_raw_borrowed(&raw) else { return };
    for slot in 0..slots {
        map_slot(s, queue, r, slot);
    }
}

/// The device a level-tiled texture belongs to.
pub(super) fn device_of(res: usize) -> Option<usize> {
    STATE.lock().unwrap().as_ref().and_then(|s| s.texs.get(&res).map(|t| t.dev))
}

/// `res` is gone: its tiles are free after a moment.
pub(super) fn forget(res: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    let Some(t) = s.texs.remove(&res) else { return };
    let now = Instant::now();
    let pool = s.pools.entry(t.dev).or_default();
    let all: u64 = t.standard.iter().map(|&(w, h, d)| w as u64 * h as u64 * d as u64).sum::<u64>() + t.packed as u64;
    s.all_tiles = s.all_tiles.saturating_sub(all);
    for (h, first, n) in t.mapped.into_iter().flatten() {
        pool.free.entry(n).or_default().push((h, first, now));
        s.mapped_tiles = s.mapped_tiles.saturating_sub(n as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
    use windows::Win32::Graphics::Dxgi::*;

    /// A level takes tiles when an upload first has texels for it, the levels without texels
    /// take none, what was uploaded reads back, and the tiles go when the texture goes.
    #[test]
    fn tiles_follow_the_levels_that_have_texels() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            let _ = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32);
            if o.TiledResourcesTier.0 < 1 {
                println!("SKIPPED tiles_follow_the_levels_that_have_texels: no tiled resources on this device");
                return;
            }
            let (size, mips) = (1024u32, 11u16);
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: size as u64, Height: size, DepthOrArraySize: 1, MipLevels: mips, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let mut p = std::ptr::null_mut();
            let hr = reserve(dev.as_raw(), &desc, D3D12_RESOURCE_STATE_COMMON.0 as u32, riid, &mut p);
            assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "a reserved texture");
            let tex = ID3D12Resource::from_raw(p);
            let key = tex.as_raw() as usize;
            let (standard, packed) = {
                let g = STATE.lock().unwrap();
                let t = &g.as_ref().unwrap().texs[&key];
                (t.standard.iter().map(|&(w, h, d)| w as u64 * h as u64 * d as u64).collect::<Vec<u64>>(), t.packed as u64)
            };
            assert!(standard.len() >= 3, "{} standard mips", standard.len());
            let mine = || {
                let g = STATE.lock().unwrap();
                g.as_ref().and_then(|s| s.texs.get(&key)).map_or(0, |t| t.mapped.iter().flatten().map(|r| r.2 as u64).sum::<u64>())
            };
            assert_eq!(mine(), 0, "nothing is mapped before the first texels");

            // texels for every level but the two largest
            let level = |i: usize, with: bool| {
                let w = (size as usize >> i).max(1);
                Subresource { rows: w, row_bytes: w * 4, depth: 1, data: if with { (0..w * w * 4).map(|k| (k * 7 + i) as u8).collect() } else { vec![0; w * w * 4] } }
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let small: Vec<Subresource> = (0..mips as usize).map(|i| level(i, i >= 2)).collect();
            let items = vec![(tex.clone(), small.clone(), vec![D3D12_RESOURCE_STATE_COMMON])];
            map_for(&queue, &items);
            assert_eq!(mine(), standard[2..].iter().sum::<u64>() + packed, "the levels with texels, and no others");
            assert!(crate::gpu::texture::upload_batch(&dev, &queue, &items).unwrap().is_empty(), "the upload");

            // level 2 back from the GPU
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            dev.GetCopyableFootprints(&desc, 2, 1, 0, Some(&mut fp), None, None, Some(&mut total));
            fp.Offset = 0;
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(&tex), Subresource: 2, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
            let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&tex), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 2 } };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut m = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut m)).unwrap();
            let got = std::slice::from_raw_parts(m as *const u8, total as usize);
            let w = size as usize >> 2;
            for y in [0, w / 2, w - 1] {
                assert_eq!(got[y * fp.Footprint.RowPitch as usize..][..w * 4], small[2].data[y * w * 4..][..w * 4], "level 2, row {y}");
            }
            readback.Unmap(0, None);

            // the largest levels arrive: all of it is mapped, and nothing twice
            let all: Vec<Subresource> = (0..mips as usize).map(|i| level(i, true)).collect();
            let items = vec![(tex.clone(), all, vec![D3D12_RESOURCE_STATE_COMMON])];
            map_for(&queue, &items);
            map_for(&queue, &items);
            assert_eq!(mine(), standard.iter().sum::<u64>() + packed);
            assert!(crate::gpu::texture::upload_batch(&dev, &queue, &items).unwrap().is_empty(), "the second upload");

            // gone: its tiles are free again
            let before = stats();
            forget(key);
            let after = stats();
            assert_eq!(before.0 - after.0, standard.iter().sum::<u64>() + packed);
            assert_eq!(before.2 - after.2, 1);
            drop(items);
            drop(tex);
        }
    }
}
