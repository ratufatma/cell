#![no_std]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueFull;

#[repr(align(64))]
struct CachePadded<T>(T);

pub struct SpscQueue<T, const CAP: usize> {
    buffer: UnsafeCell<[MaybeUninit<T>; CAP]>,
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
}

pub struct Producer<'a, T, const CAP: usize> {
    queue: &'a SpscQueue<T, CAP>,
}

pub struct Consumer<'a, T, const CAP: usize> {
    queue: &'a SpscQueue<T, CAP>,
}

unsafe impl<T: Send, const CAP: usize> Send for SpscQueue<T, CAP> {}
unsafe impl<T: Send, const CAP: usize> Sync for SpscQueue<T, CAP> {}

impl<T, const CAP: usize> SpscQueue<T, CAP> {
    pub const fn new() -> Self {
        Self {
            buffer: UnsafeCell::new([const { MaybeUninit::uninit() }; CAP]),
            head: CachePadded(AtomicUsize::new(0)),
            tail: CachePadded(AtomicUsize::new(0)),
        }
    }

    pub fn split(&self) -> (Producer<'_, T, CAP>, Consumer<'_, T, CAP>) {
        (Producer { queue: self }, Consumer { queue: self })
    }

    pub fn push(&self, item: T) -> Result<(), QueueFull> {
        self.push_inner(item)
    }

    fn push_inner(&self, item: T) -> Result<(), QueueFull> {
        assert!(CAP > 0, "SpscQueue capacity must be greater than zero");
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Acquire);
        if head.wrapping_sub(tail) == CAP {
            return Err(QueueFull);
        }

        let index = head % CAP;
        // SAFETY: Only the producer writes this slot, and the consumer publishes its
        // release of the previous value through tail before this slot is reused.
        unsafe {
            (*self.buffer.get())[index].write(item);
        }
        self.head.0.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub fn pop(&self) -> Option<T> {
        self.pop_inner()
    }

    fn pop_inner(&self) -> Option<T> {
        assert!(CAP > 0, "SpscQueue capacity must be greater than zero");
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);
        if tail == head {
            return None;
        }

        let index = tail % CAP;
        // SAFETY: The acquire load of head observes a fully initialized slot written
        // by the sole producer; only this consumer reads and removes that value.
        let item = unsafe { (*self.buffer.get())[index].assume_init_read() };
        self.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }

    pub fn is_empty(&self) -> bool {
        self.head.0.load(Ordering::Acquire) == self.tail.0.load(Ordering::Acquire)
    }

    pub fn is_full(&self) -> bool {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Acquire);
        head.wrapping_sub(tail) == CAP
    }
}

impl<T, const CAP: usize> Producer<'_, T, CAP> {
    pub fn push(&self, item: T) -> Result<(), QueueFull> {
        self.queue.push_inner(item)
    }
}

impl<T, const CAP: usize> Consumer<'_, T, CAP> {
    pub fn pop(&self) -> Option<T> {
        self.queue.pop_inner()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

impl<T, const CAP: usize> Default for SpscQueue<T, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CAP: usize> Drop for SpscQueue<T, CAP> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}
