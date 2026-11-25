//! The title's last `Release` of a resource, put off until the GPU is past the work in flight.
//!
//! D3D12 on the PC asks that an object outlives the GPU work that uses it. A console title
//! keeps to that by its own count of finished frames – but XCL's submits are not the title's:
//! texture uploads go out with a later submit, tile mappings on a queue of XCL's own. A retail
//! title's final `Release` of a resource then came while a queue still had work on it (debug
//! layer: "CORRUPTION: … is referenced by GPU operations in-flight", id 921).
//!
//! The last reference is held here instead. Each batch of held resources gets a fence value
//! signalled on every queue with work on it; the batch goes once all of them have passed it.
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// a batch is not closed more often than this (a `Signal` a queue each time)
const SEAL_EVERY: Duration = Duration::from_millis(10);
/// how often the thread here looks for batches that can go
const TICK: Duration = Duration::from_millis(4);

struct Batch {
    dead: Vec<usize>,
    /// (fence, value) a queue: all reached = nothing submitted before the batch closed is left
    marks: Vec<(ID3D12Fence, u64)>,
}

#[derive(Default)]
struct Held {
    /// queue → its fence here and the last value signalled on it
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    open: Vec<usize>,
    sealed: VecDeque<Batch>,
    last_seal: Option<Instant>,
}

// (the interface pointers are only used under the lock)
unsafe impl Send for Held {}

impl Held {
    fn is_empty(&self) -> bool {
        self.open.is_empty() && self.sealed.is_empty()
    }

    /// Close the open batch (at most every `SEAL_EVERY`, or `seal_now`) with a fence value on each of `queues`,
    /// and take out the batches every queue is past: the resources to release now.
    unsafe fn drain(&mut self, queues: &[ID3D12CommandQueue], seal_now: bool) -> Vec<usize> {
        if !self.open.is_empty() && (seal_now || self.last_seal.is_none_or(|t| t.elapsed() >= SEAL_EVERY)) {
            let mut marks = Vec::with_capacity(queues.len());
            let mut ok = true;
            for q in queues {
                let key = q.as_raw() as usize;
                if !self.fences.contains_key(&key) {
                    let mut dev: Option<ID3D12Device> = None;
                    let made = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok());
                    let Some(f) = made else {
                        ok = false;
                        break;
                    };
                    self.fences.insert(key, (f, 0));
                }
                let (fence, value) = self.fences.get_mut(&key).unwrap();
                if q.Signal(&*fence, *value + 1).is_err() {
                    ok = false;
                    break;
                }
                *value += 1;
                marks.push((fence.clone(), *value));
            }
            // (no fence or no signal: the device is gone – the batch stays open, and held)
            if ok {
                self.last_seal = Some(Instant::now());
                let dead = std::mem::take(&mut self.open);
                self.sealed.push_back(Batch { dead, marks });
            }
        }
        let mut gone = Vec::new();
        while self.sealed.front().is_some_and(|b| b.marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v)) {
            gone.extend(self.sealed.pop_front().unwrap().dead);
        }
        gone
    }
}

static HELD: Mutex<Option<Held>> = Mutex::new(None);

/// Whether last releases are held (`XCL_LATE_RELEASE=0` turns it off).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_LATE_RELEASE").is_ok_and(|v| v.trim() == "0"))
}

/// Force the one-time read of `XCL_LATE_RELEASE` on the calling (Wine) thread.
///
/// D3DMetal calls `Release` on adopted resources from a native thread of its own, which has
/// no Wine TEB.  Reading the environment there for the first time never returns.  Calling
/// this from `real::adopt` -- on the Wine thread that creates the object -- makes sure the
/// `OnceLock` inside `enabled()` is already set before the host's thread can reach it.
pub fn init() {
    let _ = enabled();
}

/// how long a last `Release` itself waits for queues with nothing to do to say so
const IDLE_WAIT: Duration = Duration::from_micros(300);

/// `res` (an adopted resource) holds one reference that is now this module's. With no work in
/// flight it goes before this returns (the queues pass a fence value at once): a title that
/// frees the memory under a resource right after its last `Release` finds it unpinned, as
/// before. Otherwise the thread here lets it go.
pub unsafe fn hold(res: usize) {
    HELD.lock().unwrap().get_or_insert_with(Held::default).open.push(res);
    let start = Instant::now();
    drain_known(true);
    while start.elapsed() < IDLE_WAIT && HELD.lock().unwrap().as_ref().is_some_and(|h| !h.is_empty()) {
        std::hint::spin_loop();
        drain_known(false);
    }
    if HELD.lock().unwrap().as_ref().is_none_or(Held::is_empty) {
        return;
    }
    // A thread of its own lets the held ones go: the title may release a resource and then
    // wait for its memory without another submit (an imported heap pins the pages under its
    // resources – unmapping them stood still while one was held "until the next submit").
    start_thread();
}

/// The thread that lets held resources go. Started by the first `hold`, or on a native host
/// when the device is adopted (`real::class_for`): a Release there starts no thread.
pub(super) fn start_thread() {
    static THREAD: std::sync::Once = std::sync::Once::new();
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-late-release".into()).spawn(|| loop {
            std::thread::sleep(TICK);
            unsafe { drain_known(false) };
            let kept = RING.kept.swap(0, Ordering::Relaxed);
            if kept > 0 {
                crate::xlog_warn!("D3D12.X: {kept} released resource(s) kept for good: the late-release ring was full");
            }
        });
    });
}

/// Resources handed over by a last `Release` that may neither lock nor wait (`hand_over`): a
/// ring of slots written with atomics alone. The indices only grow (a slot is `index % N`);
/// a writer claims an index, then writes its slot; one reader at a time empties a slot before
/// it moves the tail past it.
struct Ring<const N: usize> {
    slots: [AtomicUsize; N],
    /// the next index to write
    head: AtomicUsize,
    /// the next index to read
    tail: AtomicUsize,
    /// one reader at a time (readers are XCL's threads; writers never take it)
    reading: Mutex<()>,
    /// resources kept for good because the ring was full (`hand_over`; logged by the thread here)
    kept: AtomicUsize,
}

/// Room kept free for writers that pass the check in `Ring::push` at the same moment: with
/// fewer than this many at once, a claimed index is never a slot the reader has not emptied.
const RING_WRITERS: usize = 256;

impl<const N: usize> Ring<N> {
    const fn new() -> Self {
        Self { slots: [const { AtomicUsize::new(0) }; N], head: AtomicUsize::new(0), tail: AtomicUsize::new(0), reading: Mutex::new(()), kept: AtomicUsize::new(0) }
    }

    /// Nothing but atomics: no lock, no allocation. False: no room.
    fn push(&self, res: usize) -> bool {
        debug_assert!(res != 0);
        if self.head.load(Ordering::Acquire).wrapping_sub(self.tail.load(Ordering::Acquire)) >= N - RING_WRITERS {
            return false;
        }
        let i = self.head.fetch_add(1, Ordering::AcqRel);
        self.slots[i % N].store(res, Ordering::Release);
        true
    }

    /// What was pushed so far, in order. An index claimed but not written yet ends the look:
    /// it and those after it are taken the next time.
    fn take(&self) -> Vec<usize> {
        // (an empty ring – always, on a host that hands nothing over – is two loads)
        if self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire) {
            return Vec::new();
        }
        let _one = self.reading.lock().unwrap();
        let mut out = Vec::new();
        let mut i = self.tail.load(Ordering::Acquire);
        while i != self.head.load(Ordering::Acquire) {
            let res = self.slots[i % N].swap(0, Ordering::AcqRel);
            if res == 0 {
                break;
            }
            out.push(res);
            i = i.wrapping_add(1);
            self.tail.store(i, Ordering::Release);
        }
        out
    }
}

/// The process's ring (16,384 slots, 128 KiB): a title releases thousands of resources a
/// frame while it loads.
static RING: Ring<{ 1 << 14 }> = Ring::new();

/// `res`, an adopted resource the title released for good with one reference kept
/// (`real::res_release` on a native host), is left to the thread here: what XCL keeps by its
/// address is forgotten there (`real::forget_released`) and the object goes once the queues
/// are past the work in flight. Nothing but atomics on the calling thread. With the ring full
/// the reference is kept for good: the object is never let go, so its address never comes
/// again and nothing XCL keeps by it can be wrong.
pub(super) fn hand_over(res: usize) {
    hand_over_to(&RING, res);
}

fn hand_over_to<const N: usize>(ring: &Ring<N>, res: usize) -> bool {
    if ring.push(res) {
        return true;
    }
    ring.kept.fetch_add(1, Ordering::Relaxed);
    false
}

/// A ring entry with this bit set (an object's address is aligned, so bit 0 is free) is not a
/// resource released for good but a Release still owed: one that came on a thread XCL did not
/// create (`defer_release`).
const OWED: usize = 1;

/// The caller's `Release` of `res` came on a thread XCL did not create (`is_foreign_thread`):
/// it is owed, in the ring, and the next drain runs it through `real::res_release` on a thread
/// of XCL's or the title's. Nothing but atomics on the calling thread. With the ring full the
/// reference is kept for good, as in `hand_over`.
pub(super) fn defer_release(res: usize) {
    hand_over_to(&RING, res | OWED);
}

/// Above this, a thread's stack is a native macOS thread's (D3DMetal's), not a Wine thread's.
///
/// Measured on an M5 Pro (CrossOver 26.3, Rosetta 2):
///   Wine threads (test thread, 5 spawned std::threads): 0x0180ea9f .. 0x0220fb37
///   D3DMetal threads (spy callbacks during ExecuteCommandLists): 0x7ff2063df667 .. 0x7ff2066c3677
///
/// Wine's thread stacks are in the low hundreds of megabytes; macOS native stacks are near the
/// top of the 47-bit user range. 0x7000_0000_0000 (112 TB) leaves tens of TB on each side. The
/// check is only made on a native host (`real::native_host`): on Wine on Linux a Wine thread's
/// stack may sit in that range. The TEB pointer (`gs:[0x30]`) is not read to tell the threads
/// apart: reading it on a D3DMetal thread hangs (measured), and the stack pointer needs no
/// memory access at all.
const NATIVE_STACK_BOUNDARY: usize = 0x7000_0000_0000;

/// Whether the calling thread is one XCL did not create: a native host's own (D3DMetal's), with
/// no Wine TEB. A lock of XCL's that is contended waits in Wine's ntdll there, and never returns.
pub(super) fn is_foreign_thread() -> bool {
    let local = 0u8;
    let foreign = super::real::native_host() && &local as *const u8 as usize >= NATIVE_STACK_BOUNDARY;
    // (a test's own switch, read after the check: a thread of the host's is foreign before it
    // reads this thread-local)
    #[cfg(test)]
    let foreign = foreign || FORCE_FOREIGN.with(|f| f.get());
    foreign
}

#[cfg(test)]
thread_local! {
    /// `is_foreign_thread` answers true on this thread (a test of the foreign thread's Release
    /// on a Wine thread)
    pub(super) static FORCE_FOREIGN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Release what every queue XCL knows of is past: the title's, the tile-mapping one, the
/// upload ones.
pub unsafe fn drain_known(seal_now: bool) {
    // what was handed over by Releases that must not wait: forgotten here, on a thread of
    // XCL's or the title's, then held like any other (the object is still there, so its
    // address cannot come again before it is forgotten)
    let handed = RING.take();
    if !handed.is_empty() {
        // (and the Releases a thread XCL did not create owes (`OWED`): run here, through the
        // whole of `real::res_release`)
        let (owed, handed): (Vec<usize>, Vec<usize>) = handed.into_iter().partition(|&r| r & OWED != 0);
        for &res in &handed {
            super::real::forget_released(res);
        }
        HELD.lock().unwrap().get_or_insert_with(Held::default).open.extend(handed);
        for res in owed {
            super::real::res_release((res & !OWED) as *mut c_void);
        }
    }
    if HELD.lock().unwrap().as_ref().is_none_or(Held::is_empty) {
        return;
    }
    let mut queues: Vec<ID3D12CommandQueue> = Vec::new();
    for raw in super::present::known_queues() {
        let raw = raw as *mut c_void;
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
            queues.push(q.clone());
        }
    }
    queues.extend(super::sparse::queue());
    queues.extend(super::title_textures::upload_queues());
    let gone = match HELD.lock().unwrap().as_mut() {
        Some(h) => h.drain(&queues, seal_now),
        None => return,
    };
    // (outside the lock: a release may end other objects, whose releases come back here)
    for res in gone {
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(super::real::orig_fn(res as *mut c_void, 2));
        f(res as *mut c_void);
    }
}

/// The lock of the held resources, for a test that shows a Release does not wait for it.
#[cfg(test)]
pub(super) fn lock_held_for_test() -> impl Sized {
    HELD.lock().unwrap()
}

/// The ring's reader lock: no drain takes from the ring while a test holds it.
#[cfg(test)]
pub(super) fn lock_ring_for_test() -> impl Sized {
    RING.reading.lock().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

    /// A held resource stays while a queue has not got past the work submitted before it was
    /// released, and goes once the queue has.
    #[test]
    fn a_held_resource_goes_when_the_queue_is_past_its_batch() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            // the queue stands still, as with work in flight: it waits for a fence nobody has signalled
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let mut res: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res).unwrap();
            let raw = res.unwrap().into_raw() as usize;

            let mut held = Held::default();
            held.open.push(raw);
            let queues = [queue.clone()];
            assert!(held.drain(&queues, false).is_empty(), "the queue has not passed the batch's fence value");
            std::thread::sleep(Duration::from_millis(100));
            assert!(held.drain(&queues, false).is_empty(), "still in flight: still held");
            assert!(!held.is_empty());

            gate.Signal(1).unwrap();
            let start = Instant::now();
            let mut gone = Vec::new();
            while gone.is_empty() && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(5));
                gone = held.drain(&queues, false);
            }
            assert_eq!(gone, vec![raw], "released once the queue is past the batch");
            assert!(held.is_empty());
            drop(ID3D12Resource::from_raw(raw as *mut c_void));
        }
    }

    /// With no queue at all nothing can be in flight: the next drain lets the batch go.
    #[test]
    fn without_queues_a_batch_goes_at_once() {
        let mut held = Held::default();
        held.open.push(0x1000);
        assert_eq!(unsafe { held.drain(&[], false) }, vec![0x1000]);
        assert!(held.is_empty());
    }

    /// The ring gives back what was handed over, in order, once, and is empty after.
    #[test]
    fn the_ring_gives_back_what_was_handed_over_in_order() {
        let ring: Ring<1024> = Ring::new();
        assert!(ring.take().is_empty());
        for r in 1..=100 {
            assert!(hand_over_to(&ring, r * 0x1000));
        }
        assert_eq!(ring.take(), (1..=100).map(|r| r * 0x1000).collect::<Vec<_>>());
        assert!(ring.take().is_empty());
        // the indices go on past the end of the slots
        for round in 0..20 {
            for r in 1..=100 {
                assert!(hand_over_to(&ring, round * 0x10_0000 + r));
            }
            assert_eq!(ring.take().len(), 100);
        }
        assert_eq!(ring.kept.load(Ordering::Relaxed), 0);
    }

    /// A full ring refuses at once: the resource is counted as kept (its reference is never
    /// given back, so nothing waits and its address never comes again), and the ring takes
    /// resources again once it has been read.
    #[test]
    fn a_full_ring_keeps_the_resource_and_does_not_wait() {
        let ring: Ring<512> = Ring::new();
        let room = 512 - RING_WRITERS;
        for r in 1..=room {
            assert!(hand_over_to(&ring, r), "room for {room}, refused at {r}");
        }
        let start = Instant::now();
        assert!(!hand_over_to(&ring, 0xdead), "a full ring refuses");
        assert!(start.elapsed() < Duration::from_millis(100));
        assert_eq!(ring.kept.load(Ordering::Relaxed), 1);
        let got = ring.take();
        assert_eq!(got.len(), room);
        assert!(!got.contains(&0xdead), "the kept one is not handed over");
        assert!(hand_over_to(&ring, 0xbeef));
        assert_eq!(ring.take(), vec![0xbeef]);
    }

    /// Writers on several threads against readers on others: every resource comes out once.
    #[test]
    fn writers_and_readers_at_once_lose_nothing_and_take_nothing_twice() {
        const WRITERS: usize = 4;
        const EACH: usize = 50_000;
        let ring: &'static Ring<1024> = Box::leak(Box::new(Ring::new()));
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writers: Vec<_> = (0..WRITERS)
            .map(|w| {
                std::thread::spawn(move || {
                    for k in 0..EACH {
                        let res = (w + 1) << 32 | (k + 1);
                        while !ring.push(res) {
                            std::thread::yield_now();
                        }
                    }
                })
            })
            .collect();
        let readers: Vec<_> = (0..2)
            .map(|_| {
                let done = done.clone();
                std::thread::spawn(move || {
                    let mut got = Vec::new();
                    loop {
                        let last = done.load(Ordering::Acquire);
                        got.extend(ring.take());
                        if last {
                            break got;
                        }
                        std::thread::yield_now();
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        done.store(true, Ordering::Release);
        let mut all: Vec<usize> = readers.into_iter().flat_map(|r| r.join().unwrap()).collect();
        all.extend(ring.take());
        assert_eq!(all.len(), WRITERS * EACH, "every resource once");
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), WRITERS * EACH, "none twice");
    }

    /// A Wine thread (the test thread) is not foreign: off a native host the check is not
    /// made, and on one its stack is below the boundary.
    #[test]
    fn wine_thread_is_not_foreign() {
        assert!(!is_foreign_thread(), "the test thread must not be classified as foreign");
    }

    /// Spawned std::threads are not foreign either (their stacks may be at different
    /// addresses, but they are Wine threads with valid TEBs).
    #[test]
    fn spawned_threads_are_not_foreign() {
        let handles: Vec<_> = (0..8).map(|_| std::thread::spawn(is_foreign_thread)).collect();
        for (i, h) in handles.into_iter().enumerate() {
            assert!(!h.join().unwrap(), "spawned thread {i} classified as foreign");
        }
    }

    /// With FORCE_FOREIGN set, the detection reports the thread as foreign even though
    /// it has a TEB.  This exercises the test hook.
    #[test]
    fn force_foreign_flag_overrides_detection() {
        FORCE_FOREIGN.with(|f| f.set(true));
        assert!(is_foreign_thread(), "FORCE_FOREIGN should make it report foreign");
        FORCE_FOREIGN.with(|f| f.set(false));
        assert!(!is_foreign_thread(), "back to normal");
    }
}
