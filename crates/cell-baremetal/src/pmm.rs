use core::sync::atomic::{AtomicBool, Ordering};
use limine::memory_map::{Entry, EntryType};

pub const FRAME_SIZE: u64 = 4096;
const MAX_PHYSICAL_ADDRESS: u64 = 4 * 1024 * 1024 * 1024;
const MAX_FRAMES: usize = (MAX_PHYSICAL_ADDRESS / FRAME_SIZE) as usize;
const BITMAP_WORDS: usize = MAX_FRAMES / 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalFrame {
    number: usize,
}

impl PhysicalFrame {
    pub const fn from_address(address: usize) -> Self {
        Self {
            number: address / FRAME_SIZE as usize,
        }
    }

    pub const fn index(self) -> usize {
        self.number
    }

    pub const fn address(self) -> u64 {
        self.number as u64 * FRAME_SIZE
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PmmStats {
    pub usable_frames: usize,
    pub free_frames: usize,
    pub max_frame: usize,
}

pub struct BitmapPmm {
    usable: [u64; BITMAP_WORDS],
    allocated: [u64; BITMAP_WORDS],
    usable_frames: usize,
    free_frames: usize,
    max_frame: usize,
}

impl BitmapPmm {
    pub const fn new() -> Self {
        Self {
            usable: [0; BITMAP_WORDS],
            allocated: [u64::MAX; BITMAP_WORDS],
            usable_frames: 0,
            free_frames: 0,
            max_frame: 0,
        }
    }

    fn initialize(&mut self, entries: &[&Entry]) -> PmmStats {
        self.usable = [0; BITMAP_WORDS];
        self.allocated = [u64::MAX; BITMAP_WORDS];
        self.usable_frames = 0;
        self.free_frames = 0;
        self.max_frame = 0;

        for entry in entries {
            if entry.entry_type != EntryType::USABLE {
                continue;
            }
            let Some(start) = entry.base.checked_add(FRAME_SIZE - 1) else {
                continue;
            };
            let start = start & !(FRAME_SIZE - 1);
            let Some(end) = entry.base.checked_add(entry.length) else {
                continue;
            };
            let end = core::cmp::min(end, MAX_PHYSICAL_ADDRESS);
            if start >= end {
                continue;
            }

            let first_frame = (start / FRAME_SIZE) as usize;
            let last_frame = (end / FRAME_SIZE) as usize;
            for frame in first_frame..last_frame {
                self.mark_usable(frame);
            }
        }

        PmmStats {
            usable_frames: self.usable_frames,
            free_frames: self.free_frames,
            max_frame: self.max_frame,
        }
    }

    fn mark_usable(&mut self, frame: usize) {
        if frame >= MAX_FRAMES || self.is_usable(frame) {
            return;
        }
        self.usable[frame / 64] |= 1 << (frame % 64);
        self.allocated[frame / 64] &= !(1 << (frame % 64));
        self.usable_frames += 1;
        self.free_frames += 1;
        self.max_frame = self.max_frame.max(frame + 1);
    }

    fn is_usable(&self, frame: usize) -> bool {
        frame < MAX_FRAMES && (self.usable[frame / 64] & (1 << (frame % 64))) != 0
    }

    fn is_allocated(&self, frame: usize) -> bool {
        (self.allocated[frame / 64] & (1 << (frame % 64))) != 0
    }

    fn is_free(&self, frame: usize) -> bool {
        self.is_usable(frame) && !self.is_allocated(frame)
    }

    fn allocate(&mut self) -> Option<PhysicalFrame> {
        for word_index in 0..BITMAP_WORDS {
            let available = self.usable[word_index] & !self.allocated[word_index];
            if available == 0 {
                continue;
            }
            let bit = available.trailing_zeros() as usize;
            let frame = word_index * 64 + bit;
            self.allocated[word_index] |= 1 << bit;
            self.free_frames -= 1;
            return Some(PhysicalFrame { number: frame });
        }
        None
    }

    fn free(&mut self, frame: PhysicalFrame) -> bool {
        let number = frame.number;
        if !self.is_usable(number) || !self.is_allocated(number) {
            return false;
        }
        self.allocated[number / 64] &= !(1 << (number % 64));
        self.free_frames += 1;
        true
    }

    fn allocate_contiguous(&mut self, count: usize) -> Option<PhysicalFrame> {
        if count == 0 {
            return None;
        }
        if count == 1 {
            return self.allocate();
        }

        let mut run_start = 0;
        let mut run_length = 0;
        for frame in 0..self.max_frame {
            if self.is_free(frame) {
                if run_length == 0 {
                    run_start = frame;
                }
                run_length += 1;
                if run_length == count {
                    for allocated_frame in run_start..run_start + count {
                        self.allocated[allocated_frame / 64] |= 1 << (allocated_frame % 64);
                    }
                    self.free_frames -= count;
                    return Some(PhysicalFrame { number: run_start });
                }
            } else {
                run_length = 0;
            }
        }
        None
    }

    fn free_contiguous(&mut self, base: PhysicalFrame, count: usize) {
        if count == 0 {
            return;
        }
        let Some(end) = base.number.checked_add(count) else {
            return;
        };
        if end > self.max_frame
            || !(base.number..end).all(|frame| self.is_usable(frame) && self.is_allocated(frame))
        {
            return;
        }
        for frame in base.number..end {
            self.allocated[frame / 64] &= !(1 << (frame % 64));
        }
        self.free_frames += count;
    }

    fn largest_free_run(&self) -> usize {
        let mut largest = 0;
        let mut current = 0;
        for frame in 0..self.max_frame {
            if self.is_free(frame) {
                current += 1;
                largest = largest.max(current);
            } else {
                current = 0;
            }
        }
        largest
    }
}

static mut PMM: BitmapPmm = BitmapPmm::new();
static PMM_LOCK: AtomicBool = AtomicBool::new(false);

fn lock() {
    while PMM_LOCK
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
}

fn unlock() {
    PMM_LOCK.store(false, Ordering::Release);
}

/// SAFETY: Initialization is performed once by the BSP before any frame API is
/// used. The APs do not access PMM in this PoC.
pub unsafe fn initialize(entries: &[&Entry]) -> PmmStats {
    lock();
    let stats = (*core::ptr::addr_of_mut!(PMM)).initialize(entries);
    unlock();
    stats
}

/// SAFETY: Only the BSP calls this allocator in the current kernel design.
pub unsafe fn allocate_frame() -> Option<PhysicalFrame> {
    lock();
    let frame = (*core::ptr::addr_of_mut!(PMM)).allocate();
    unlock();
    frame
}

/// SAFETY: The caller must own the frame exclusively. In the tensor pipeline,
/// ownership is transferred from BSP to AP through TensorChunk before AP calls
/// this function; no other PMM operation runs concurrently.
pub unsafe fn free_frame(frame: PhysicalFrame) -> bool {
    lock();
    let freed = (*core::ptr::addr_of_mut!(PMM)).free(frame);
    unlock();
    freed
}

/// SAFETY: The caller must own the returned contiguous frame range.
pub unsafe fn allocate_contiguous_frames(count: usize) -> Option<PhysicalFrame> {
    lock();
    let frame = (*core::ptr::addr_of_mut!(PMM)).allocate_contiguous(count);
    unlock();
    frame
}

/// SAFETY: The caller must own the complete contiguous frame range.
pub unsafe fn free_contiguous_frames(base: PhysicalFrame, count: usize) {
    lock();
    (*core::ptr::addr_of_mut!(PMM)).free_contiguous(base, count);
    unlock();
}

/// SAFETY: The caller must ensure no concurrent PMM mutation occurs outside
/// this module; this read is protected by the PMM lock.
pub unsafe fn largest_free_frames() -> usize {
    lock();
    let largest = (*core::ptr::addr_of!(PMM)).largest_free_run();
    unlock();
    largest
}
