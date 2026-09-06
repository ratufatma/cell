use core::arch::asm;
use core::sync::atomic::{AtomicBool, AtomicUsize};

use cell_core::{RawPayload, TaskDescriptor, TraceContext, ValidatedPayload, WorkResult};
use cell_queue::SpscQueue;

pub const PACKET_COUNT: usize = 10;

pub enum PipelineMessage {
    Valid(ValidatedPayload),
    BypassFault {
        context: TraceContext,
        reason_code: u16,
    },
}

pub static QUEUE_INGRESS_TO_W1: SpscQueue<RawPayload, 8> = SpscQueue::new();
pub static QUEUE_W1_TO_W2: SpscQueue<PipelineMessage, 32> = SpscQueue::new();
pub static QUEUE_W2_TO_SUPERVISOR: SpscQueue<WorkResult, 32> = SpscQueue::new();
pub static TENSOR_HOP1: SpscQueue<TaskDescriptor, 8> = SpscQueue::new();
pub static TENSOR_HOP2: SpscQueue<TaskDescriptor, 8> = SpscQueue::new();

pub static AP1_READY: AtomicBool = AtomicBool::new(false);
pub static AP2_READY: AtomicBool = AtomicBool::new(false);
pub static AP3_READY: AtomicBool = AtomicBool::new(false);
pub static AP1_DONE: AtomicBool = AtomicBool::new(false);
pub static AP2_DONE: AtomicBool = AtomicBool::new(false);
pub static AP3_DONE: AtomicBool = AtomicBool::new(false);
pub static TENSOR_DONE: AtomicBool = AtomicBool::new(false);
pub static TENSOR_PHYS_ADDR: AtomicUsize = AtomicUsize::new(0);
pub static TENSOR_SUM_BITS: AtomicUsize = AtomicUsize::new(0);
pub static ATTENTION_DONE: AtomicBool = AtomicBool::new(false);
pub static ATTENTION_PHYS_ADDR: AtomicUsize = AtomicUsize::new(0);
pub static ATTENTION_SUM_BITS: AtomicUsize = AtomicUsize::new(0);
pub static RMSNORM_DONE: AtomicBool = AtomicBool::new(false);
pub static RMSNORM_PHYS_ADDR: AtomicUsize = AtomicUsize::new(0);
pub static RMSNORM_SUM_BITS: AtomicUsize = AtomicUsize::new(0);
pub static TRANSFORMER_BLOCK_DONE: AtomicBool = AtomicBool::new(false);
pub static TRANSFORMER_BLOCK_PHYS_ADDR: AtomicUsize = AtomicUsize::new(0);
pub static TRANSFORMER_BLOCK_SUM_BITS: AtomicUsize = AtomicUsize::new(0);
pub static EXTERNAL_WEIGHTS_LOADED: AtomicBool = AtomicBool::new(false);
pub static mut EXTERNAL_WEIGHTS_PTR: *const f32 = core::ptr::null();
pub static mut EXTERNAL_WEIGHTS_FRAMES: usize = 0;

pub static KV_CACHE_DONE: AtomicBool = AtomicBool::new(false);
pub static KV_CACHE_PHYS_ADDR: AtomicUsize = AtomicUsize::new(0);
pub static KV_CACHE_BLOCK_ID: AtomicUsize = AtomicUsize::new(0);
pub static mut KV_CACHE_VIRT: *mut u8 = core::ptr::null_mut();

const WAIT_TASK_SPIN_LIMIT: usize = 1_000_000;

pub fn wait_task<T>(queue: &'static SpscQueue<T, 8>, queue_name: &'static str) -> Option<T> {
    let mut spins = 0_usize;
    loop {
        if let Some(value) = queue.pop() {
            return Some(value);
        }
        spins += 1;
        if spins >= WAIT_TASK_SPIN_LIMIT {
            serial_println!(
                "[CELL WARN] wait_task timeout on queue '{}' after {} spins",
                queue_name,
                WAIT_TASK_SPIN_LIMIT
            );
            return None;
        }
        core::hint::spin_loop();
    }
}

pub fn halt() -> ! {
    loop {
        // SAFETY: Executing HLT in an infinite loop parks the core until the
        // next interrupt without touching any memory.
        unsafe {
            asm!("hlt", options(nomem, nostack, preserves_flags));
        }
    }
}