#![no_std]
#![no_main]

mod pmm;

use cell_core::{
    DType, MutableState, RawPayload, TensorChunk, TensorShape, TraceContext, ValidatorCapability,
    WorkResult,
};
use cell_queue::SpscQueue;
use cell_supervisor::Supervisor;
use core::arch::asm;
use core::fmt::{self, Write};
use core::mem::size_of;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use limine::mp::Cpu;
use limine::request::{HhdmRequest, MemoryMapRequest, RequestsEndMarker, RequestsStartMarker};
use limine::BaseRevision;

#[used]
#[link_section = ".requests_start_marker"]
static REQUESTS_START_MARKER: RequestsStartMarker = RequestsStartMarker::new();

#[used]
#[link_section = ".limine_requests"]
static BASE_REVISION: BaseRevision = BaseRevision::new();

#[used]
#[link_section = ".limine_requests"]
#[allow(deprecated)]
static SMP_REQUEST: limine::request::SmpRequest = limine::request::SmpRequest::new();

#[used]
#[link_section = ".limine_requests"]
static MEMMAP_REQUEST: MemoryMapRequest = MemoryMapRequest::new();

#[used]
#[link_section = ".limine_requests"]
static HHDM_REQUEST: HhdmRequest = HhdmRequest::new();

#[used]
#[link_section = ".requests_end_marker"]
static REQUESTS_END_MARKER: RequestsEndMarker = RequestsEndMarker::new();

const COM1: u16 = 0x3f8;
const PACKET_COUNT: usize = 10;
const TENSOR_ELEMENTS: usize = 1024;
static INTER_CORE_QUEUE: SpscQueue<RawPayload, 16> = SpscQueue::new();
static COMPLETION_QUEUE: SpscQueue<WorkResult, 32> = SpscQueue::new();
static TENSOR_QUEUE: SpscQueue<TensorChunk<cell_core::Ready>, 2> = SpscQueue::new();
static AP_READY: AtomicBool = AtomicBool::new(false);
static AP_PROCESSED: AtomicUsize = AtomicUsize::new(0);
static TENSOR_COMPLETE: AtomicBool = AtomicBool::new(false);
static TENSOR_SUM_BITS: AtomicUsize = AtomicUsize::new(0);
static SERIAL_LOCK: AtomicBool = AtomicBool::new(false);

struct Serial;

impl Serial {
    const fn new() -> Self {
        Self
    }

    unsafe fn outb(port: u16, value: u8) {
        // SAFETY: COM1 is the conventional x86 primary UART port and this is the
        // only writer during single-core early boot.
        asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack, preserves_flags));
    }

    unsafe fn inb(port: u16) -> u8 {
        let value: u8;
        // SAFETY: Reading the UART line-status register is valid for COM1.
        asm!("in al, dx", out("al") value, in("dx") port, options(nomem, nostack, preserves_flags));
        value
    }

    unsafe fn init(&self) {
        Self::outb(COM1 + 1, 0x00);
        Self::outb(COM1 + 3, 0x80);
        Self::outb(COM1, 0x03);
        Self::outb(COM1 + 1, 0x00);
        Self::outb(COM1 + 3, 0x03);
        Self::outb(COM1 + 2, 0xc7);
        Self::outb(COM1 + 4, 0x0b);
    }

    unsafe fn write_byte(&self, byte: u8) {
        while Self::inb(COM1 + 5) & 0x20 == 0 {
            core::hint::spin_loop();
        }
        Self::outb(COM1, byte);
    }
}

impl Write for Serial {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for byte in text.bytes() {
            unsafe {
                if byte == b'\n' {
                    self.write_byte(b'\r');
                }
                self.write_byte(byte);
            }
        }
        Ok(())
    }
}

static mut SERIAL: Serial = Serial::new();

macro_rules! serial_print {
    ($($arg:tt)*) => {{
        serial_write(format_args!($($arg)*));
    }};
}

macro_rules! serial_println {
    () => (serial_print!("\n"));
    ($fmt:expr) => (serial_print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => (serial_print!(concat!($fmt, "\n"), $($arg)*));
}

fn serial_write(arguments: fmt::Arguments<'_>) {
    while SERIAL_LOCK
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }

    unsafe {
        let serial = &mut *core::ptr::addr_of_mut!(SERIAL);
        let _ = serial.write_fmt(arguments);
    }
    SERIAL_LOCK.store(false, Ordering::Release);
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe { (&*core::ptr::addr_of!(SERIAL)).init() };
    serial_println!("[CELL KERNEL] Dataflow pipeline active");

    let Some(memory_map) = MEMMAP_REQUEST.get_response() else {
        serial_println!("[CELL PMM] no Limine memory map response");
        halt();
    };
    let pmm_stats = unsafe { pmm::initialize(memory_map.entries()) };
    serial_println!(
        "[CELL PMM] usable={} free={} max_frame={}",
        pmm_stats.usable_frames,
        pmm_stats.free_frames,
        pmm_stats.max_frame
    );
    let Some(probe_frame) = (unsafe { pmm::allocate_frame() }) else {
        serial_println!("[CELL PMM] allocation probe failed");
        halt();
    };
    serial_println!(
        "[CELL PMM] allocated frame={} physical=0x{:x}",
        probe_frame.number(),
        probe_frame.address()
    );
    if !unsafe { pmm::free_frame(probe_frame) } {
        serial_println!("[CELL PMM] deallocation probe failed");
        halt();
    }
    serial_println!("[CELL PMM] frame returned to bitmap");

    let Some(hhdm) = HHDM_REQUEST.get_response() else {
        serial_println!("[CELL PMM] no Limine HHDM response");
        halt();
    };
    let tensor_frame = unsafe { pmm::allocate_frame() }.unwrap_or_else(|| halt());
    let tensor_virtual = hhdm
        .offset()
        .checked_add(tensor_frame.address())
        .unwrap_or_else(|| halt()) as usize as *mut u8;
    let mut tensor = TensorChunk::<MutableState>::new(
        TraceContext::new(100, 100, 0),
        tensor_frame.address() as usize,
        tensor_virtual,
        TensorShape::new_1d(TENSOR_ELEMENTS),
        DType::F32,
    );
    let one = 1.0_f32.to_ne_bytes();
    for bytes in tensor.as_mut_slice().chunks_exact_mut(size_of::<f32>()) {
        bytes.copy_from_slice(&one);
    }
    let tensor = tensor.freeze();
    serial_println!(
        "[CELL TENSOR] BSP prepared {} F32 elements phys=0x{:x}",
        TENSOR_ELEMENTS,
        tensor_frame.address()
    );

    let Some(response) = SMP_REQUEST.get_response() else {
        serial_println!("[CELL SMP] no Limine SMP response");
        halt();
    };

    let mut ap_started = false;
    for cpu in response.cpus() {
        if cpu.lapic_id != response.bsp_lapic_id() {
            cpu.goto_address.write(ap_entry);
            ap_started = true;
            break;
        }
    }

    if !ap_started {
        serial_println!("[CELL SMP] no application processor found");
        halt();
    }

    while !AP_READY.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    serial_println!("[CELL SMP] BSP -> AP pipeline online");
    serial_println!("[CELL SMP] Bidirectional feedback loop active");

    for trace_id in 1..=PACKET_COUNT {
        let context = TraceContext::new(trace_id as u64, trace_id as u64, 0);
        let data = if trace_id == 4 { &[][..] } else { b"CELL data" };
        let raw = RawPayload::new(context, data).unwrap_or_else(|_| halt());
        while INTER_CORE_QUEUE.push(raw).is_err() {
            core::hint::spin_loop();
        }
    }

    if TENSOR_QUEUE.push(tensor).is_err() {
        serial_println!("[CELL TENSOR] tensor queue unexpectedly full");
        halt();
    }
    while !TENSOR_COMPLETE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let tensor_sum = f32::from_bits(TENSOR_SUM_BITS.load(Ordering::Acquire) as u32);
    serial_println!(
        "[CELL TENSOR] AP1 zero-copy reduction sum={} expected={}",
        tensor_sum,
        TENSOR_ELEMENTS
    );

    let mut supervisor = Supervisor::<PACKET_COUNT>::new();
    let mut reports = 0;
    let mut succeeded = 0;
    let mut failed = 0;
    while reports < PACKET_COUNT {
        let Some(result) = COMPLETION_QUEUE.pop() else {
            core::hint::spin_loop();
            continue;
        };
        match result {
            WorkResult::Completed(context) => {
                succeeded += 1;
                serial_println!("[SUPERVISOR] Trace {} OK", context.trace_id);
            }
            WorkResult::Failed {
                context,
                node_id,
                reason,
            } => {
                failed += 1;
                let _ = supervisor.observe(result);
                serial_println!(
                    "[SUPERVISOR WARN] Fault captured on Node {}! Trace {} failed: {}",
                    node_id,
                    context.trace_id,
                    reason
                );
            }
        }
        reports += 1;
    }

    if succeeded == 9 && failed == 1 && supervisor.failure_count() == 1 {
        serial_println!(
            "[CELL KERNEL] Batch completed: 9 succeeded, 1 isolated failure. Zero crash."
        );
    } else {
        serial_println!(
            "[CELL KERNEL] Batch invariant failed: ok={} failed={} recorded={}",
            succeeded,
            failed,
            supervisor.failure_count()
        );
    }
    halt()
}

unsafe extern "C" fn ap_entry(_cpu: &Cpu) -> ! {
    AP_READY.store(true, Ordering::Release);
    serial_println!("[CELL SMP] AP1 online");

    let verifier = ValidatorCapability::new(0xCE11_2021);
    while AP_PROCESSED.load(Ordering::Relaxed) < PACKET_COUNT {
        if let Some(raw) = INTER_CORE_QUEUE.pop() {
            let result = if raw.is_empty() {
                WorkResult::Failed {
                    context: raw.context(),
                    node_id: 1,
                    reason: "Corrupted payload in AP1",
                }
            } else {
                let validated = raw.validate(&verifier);
                WorkResult::Completed(validated.context())
            };
            while COMPLETION_QUEUE.push(result).is_err() {
                core::hint::spin_loop();
            }
            AP_PROCESSED.fetch_add(1, Ordering::Release);
        } else {
            core::hint::spin_loop();
        }
    }

    loop {
        let Some(tensor) = TENSOR_QUEUE.pop() else {
            core::hint::spin_loop();
            continue;
        };
        let (sum, element_count) = {
            let values = tensor.as_slice::<f32>();
            let mut sum = 0.0_f32;
            for value in values {
                sum += *value;
            }
            (sum, values.len())
        };
        let (context, phys_addr) = tensor.deconstruct();
        let sum_bits = sum.to_bits() as usize;
        TENSOR_SUM_BITS.store(sum_bits, Ordering::Release);
        // SAFETY: BSP initialized PMM before starting the AP and transfers this
        // frame exclusively to AP1 through TensorChunk ownership.
        let returned = pmm::free_frame(pmm::PhysicalFrame::from_address(phys_addr));
        if returned && sum == TENSOR_ELEMENTS as f32 {
            serial_println!(
                "[CELL TENSOR] AP1 trace={} reduced {} elements and freed phys=0x{:x}",
                context.trace_id,
                element_count,
                phys_addr
            );
        } else {
            serial_println!("[CELL TENSOR] reduction or frame release failed");
        }
        TENSOR_COMPLETE.store(true, Ordering::Release);
        break;
    }
    halt()
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    serial_println!("[CELL KERNEL] panic: {}", info);
    halt()
}

fn halt() -> ! {
    loop {
        unsafe {
            asm!("hlt", options(nomem, nostack, preserves_flags));
        }
    }
}
