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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
    // Record whether this is Wine on macOS so is_foreign_thread can check without a Win32 call.
    if !MACOS_HOST.load(Ordering::Relaxed) {
        MACOS_HOST.store(crate::util::mac_host(), Ordering::Relaxed);
    }
}

// --- Foreign-thread detection and deferred release queue ---
//
// A host driver (D3DMetal on macOS) may call Release on adopted resources from a thread
// of its own that has no Wine TEB.  Mutex, env-var reads, and most Win32 calls hang or
// crash there.  When `is_foreign_thread()` is true, the release work is deferred to a
// Wine thread through an atomic ring buffer.

/// Set by `init()` on a Wine thread: Wine on a macOS host, where D3DMetal has threads of its
/// own.  On Windows and on Wine on Linux it stays false, so the foreign-thread path is off and
/// `is_foreign_thread()` returns false unconditionally.
static MACOS_HOST: AtomicBool = AtomicBool::new(false);

/// Boundary between Wine/Rosetta 2 stacks and macOS native stacks.
///
/// Measured on an M5 Pro (CrossOver 26.3, Rosetta 2):
///   Wine threads (test thread, 5 spawned std::threads): 0x0180ea9f .. 0x0220fb37
///   D3DMetal threads (spy callbacks during ExecuteCommandLists): 0x7ff2063df667 .. 0x7ff2066c3677
///
/// Wine under Rosetta 2 runs in the emulated address space; thread stacks are in the low
/// hundreds of megabytes.  macOS native stacks are near the top of the 47-bit user range.
/// The gap is over 100 TB.  A threshold at 0x7000_0000_0000 (112 TB) is well above any
/// Wine thread and well below any observed D3DMetal thread, with tens of TB of margin on
/// each side.  The check only runs on a macOS host (`MACOS_HOST`): on Linux a Wine thread's
/// stack may sit in that range, and every Release from it would take the deferred path.
///
/// The TEB pointer (`gs:[0x30]`) is not used to tell the threads apart: reading it on a
/// D3DMetal thread hangs (measured), and the stack pointer needs no memory access at all.
const NATIVE_STACK_BOUNDARY: usize = 0x7000_0000_0000;

#[cfg(not(test))]
pub(super) fn is_foreign_thread() -> bool {
    if !MACOS_HOST.load(Ordering::Relaxed) {
        return false;
    }
    let local = 0u8;
    let sp = &local as *const u8 as usize;
    sp >= NATIVE_STACK_BOUNDARY
}

/// Test version: a thread-local flag overrides the detection so unit tests can exercise
/// the foreign path on a normal Wine thread.
#[cfg(test)]
pub(super) fn is_foreign_thread() -> bool {
    if FORCE_FOREIGN.load(Ordering::Relaxed) != 0 {
        return true;
    }
    if !MACOS_HOST.load(Ordering::Relaxed) {
        return false;
    }
    let local = 0u8;
    let sp = &local as *const u8 as usize;
    sp >= NATIVE_STACK_BOUNDARY
}

#[cfg(test)]
static FORCE_FOREIGN: AtomicUsize = AtomicUsize::new(0);

/// Ring buffer for releases deferred from a foreign thread.
///
/// Each slot holds a resource pointer whose Release is kept for a Wine thread.  0 means the
/// slot is empty.  Every Release on a foreign thread goes through here, and a full ring leaks
/// the reference, so it is sized for a title's loading screens rather than the 2 Release
/// calls per ExecuteCommandLists the probes show: 16,384 slots, 128 KiB.
const DEFERRED_CAP: usize = 16_384;

struct Ring {
    slots: [AtomicUsize; DEFERRED_CAP],
    /// next slot a producer (foreign thread) claims
    head: AtomicUsize,
    /// next slot the consumer (a Wine thread) takes
    tail: AtomicUsize,
    /// set while one thread takes entries off the ring: there is one consumer at a time
    draining: AtomicBool,
}

impl Ring {
    const fn new() -> Self {
        Ring {
            slots: [const { AtomicUsize::new(0) }; DEFERRED_CAP],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            draining: AtomicBool::new(false),
        }
    }

    /// Push a deferred Release.  Returns false if the ring is full (the caller keeps the
    /// reference).  Only atomics: no locks, no allocation.
    fn push(&self, res: usize) -> bool {
        loop {
            let head = self.head.load(Ordering::Relaxed);
            let next = (head + 1) % DEFERRED_CAP;
            if next == self.tail.load(Ordering::Acquire) {
                return false;
            }
            // Claim this slot.  D3DMetal serialises its submit callbacks (one foreign producer),
            // but CAS is correct if that ever changes.
            if self.head.compare_exchange_weak(head, next, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                self.slots[head].store(res, Ordering::Release);
                return true;
            }
        }
    }

    /// Move every written entry off the ring into `out`, oldest first.  Returns false without
    /// taking anything when another thread is taking entries already (it takes these too, or
    /// the next drain does).
    ///
    /// The slot is emptied before the tail moves past it.  A producer can claim a slot only
    /// once the tail is past it, so it always finds the slot empty and its entry is never
    /// erased.  With one consumer, a slot that is still 0 at the tail is one a producer has
    /// claimed and not written yet: the drain stops there and the next one carries on.
    fn take(&self, out: &mut Vec<usize>) -> bool {
        if self.draining.swap(true, Ordering::Acquire) {
            return false;
        }
        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            if tail == self.head.load(Ordering::Acquire) {
                break; // empty
            }
            let value = self.slots[tail].swap(0, Ordering::Acquire);
            if value == 0 {
                break; // claimed, not written yet
            }
            self.tail.store((tail + 1) % DEFERRED_CAP, Ordering::Release);
            out.push(value);
        }
        self.draining.store(false, Ordering::Release);
        true
    }
}

static DEFERRED: Ring = Ring::new();

/// Releases that could not fit in the ring.  The reference is kept: the object and XCL's
/// bookkeeping entry for it stay until the process ends.
pub(super) static DEFERRED_OVERFLOW: AtomicUsize = AtomicUsize::new(0);

/// Defer a Release from a foreign thread to a Wine thread.  False when the ring is full (the
/// caller keeps the reference and increments `DEFERRED_OVERFLOW`).
pub(super) fn defer_release(res: usize) -> bool {
    DEFERRED.push(res)
}

/// Drain the deferred ring buffer.  Each entry is a Release a foreign thread kept for a Wine
/// thread; it runs here through the full `res_release` path (cleanup, late-release hold).
pub unsafe fn drain_deferred() {
    let mut work = Vec::new();
    DEFERRED.take(&mut work);
    // Released outside the ring: res_release drains again, and these are already off it.
    for res in work {
        super::real::res_release(res as *mut c_void);
    }
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
    static THREAD: std::sync::Once = std::sync::Once::new();
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-late-release".into()).spawn(|| loop {
            std::thread::sleep(TICK);
            unsafe { drain_deferred() };
            unsafe { drain_known(false) };
            let overflow = DEFERRED_OVERFLOW.swap(0, Ordering::Relaxed);
            if overflow > 0 {
                crate::xlog_warn!("D3D12.X: {overflow} deferred release(s) dropped (ring full, references leaked)");
            }
        });
    });
}

/// Release what every queue XCL knows of is past: the title's, the tile-mapping one, the
/// upload ones.
pub unsafe fn drain_known(seal_now: bool) {
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

    /// A Wine thread (the test thread) is not foreign: `under_wine` is false on Windows CI
    /// and the SP is below the boundary on macOS Wine.
    #[test]
    fn wine_thread_is_not_foreign() {
        assert!(!is_foreign_thread(), "the test thread must not be classified as foreign");
    }

    /// Spawned std::threads are not foreign either (their stacks may be at different
    /// addresses, but they are Wine threads with valid TEBs).
    #[test]
    fn spawned_threads_are_not_foreign() {
        let handles: Vec<_> = (0..8).map(|_| {
            std::thread::spawn(|| is_foreign_thread())
        }).collect();
        for (i, h) in handles.into_iter().enumerate() {
            assert!(!h.join().unwrap(), "spawned thread {i} classified as foreign");
        }
    }

    /// With FORCE_FOREIGN set, the detection reports the thread as foreign even though
    /// it has a TEB.  This exercises the test hook.
    #[test]
    fn force_foreign_flag_overrides_detection() {
        FORCE_FOREIGN.store(1, Ordering::Relaxed);
        assert!(is_foreign_thread(), "FORCE_FOREIGN should make it report foreign");
        FORCE_FOREIGN.store(0, Ordering::Relaxed);
        assert!(!is_foreign_thread(), "back to normal");
    }

    /// A ring of the test's own: the process-wide one is drained by every `res_release`, which
    /// would release the made-up pointers these tests push.
    fn empty_ring() -> std::sync::Arc<Ring> {
        std::sync::Arc::new(Ring::new())
    }

    /// Entries come off the ring once each and in order, across the wrap-around: a slot
    /// written again after a drain keeps its new entry.
    #[test]
    fn deferred_ring_round_trips_across_the_wrap() {
        let ring = empty_ring();
        for round in 0..3usize {
            let want: Vec<usize> = (1..DEFERRED_CAP).map(|i| round * DEFERRED_CAP + i).collect();
            for &v in &want {
                assert!(ring.push(v), "round {round}: defer_release failed at {v}");
            }
            let mut got = Vec::new();
            assert!(ring.take(&mut got));
            assert_eq!(got, want, "round {round}");
        }
    }

    /// One producer filling the ring while several threads drain it: every entry comes off
    /// exactly once.  Producers keep reusing slots right behind the consumer (the ring is
    /// small next to the count), which is where an entry written into a slot the consumer
    /// had not finished with would be lost.
    #[test]
    fn deferred_ring_concurrent_producer_and_drains() {
        let ring = empty_ring();
        const N: usize = 200_000;
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let drains: Vec<_> = (0..4).map(|_| {
            let (done, ring) = (done.clone(), ring.clone());
            std::thread::spawn(move || {
                let mut got = Vec::new();
                loop {
                    let finished = done.load(Ordering::Acquire);
                    ring.take(&mut got);
                    if finished && ring.tail.load(Ordering::Acquire) == ring.head.load(Ordering::Acquire) {
                        return got;
                    }
                }
            })
        }).collect();
        for v in 1..=N {
            while !ring.push(v) {
                std::hint::spin_loop();
            }
        }
        done.store(true, Ordering::Release);
        let mut got: Vec<usize> = drains.into_iter().flat_map(|h| h.join().unwrap()).collect();
        let taken = got.len();
        got.sort_unstable();
        got.dedup();
        assert_eq!(taken, N, "entries taken more than once or lost");
        assert_eq!(got, (1..=N).collect::<Vec<_>>(), "each entry must come off exactly once");
    }

    /// While one thread takes entries, another drain returns at once and takes none.
    #[test]
    fn deferred_ring_has_one_consumer_at_a_time() {
        let ring = empty_ring();
        assert!(ring.push(0x1234));
        ring.draining.store(true, Ordering::Relaxed);
        let mut got = Vec::new();
        assert!(!ring.take(&mut got));
        assert!(got.is_empty());
        ring.draining.store(false, Ordering::Relaxed);
        assert!(ring.take(&mut got));
        assert_eq!(got, vec![0x1234]);
    }

    /// A claimed slot that is not written yet stops the drain there, and the next drain
    /// takes it once it is written.
    #[test]
    fn deferred_ring_stops_at_an_unwritten_slot() {
        let ring = empty_ring();
        let head = ring.head.load(Ordering::Relaxed);
        ring.head.store((head + 1) % DEFERRED_CAP, Ordering::Release); // claimed, slot still 0
        assert!(ring.push(0x2222));
        let mut got = Vec::new();
        assert!(ring.take(&mut got));
        assert!(got.is_empty(), "nothing past the unwritten slot");
        ring.slots[head].store(0x1111, Ordering::Release);
        assert!(ring.take(&mut got));
        assert_eq!(got, vec![0x1111, 0x2222]);
    }

    /// When the ring is full, `defer_release` returns false instead of spinning.
    #[test]
    fn deferred_ring_overflow_returns_false() {
        let ring = empty_ring();
        // capacity is DEFERRED_CAP - 1 usable slots
        for i in 0..(DEFERRED_CAP - 1) {
            assert!(ring.push(0x1000 + i), "should succeed at {i}");
        }
        assert!(!ring.push(0xdead), "should return false when full");
        let mut got = Vec::new();
        assert!(ring.take(&mut got));
        assert_eq!(got.len(), DEFERRED_CAP - 1);
    }
}
