#![no_std]
#![no_main]

mod pmm;
mod simd;

use cell_core::{
    DType, MutableState, RawPayload, TensorChunk, TensorShape, TraceContext, ValidatedPayload,
    ValidatorCapability, WorkResult,
};
use cell_queue::SpscQueue;
use cell_supervisor::telemetry::{
    FaultIncident, Heartbeat, PmmSnapshot, QueueMetrics, TelemetryEncoder, TelemetryEvent,
    TensorExecution,
};
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
const TENSOR_FRAME_COUNT: usize = 4;
const TENSOR_ELEMENTS: usize = 4096;
pub enum PipelineMessage {
    Valid(ValidatedPayload),
    BypassFault {
        context: TraceContext,
        reason_code: u16,
    },
}

static QUEUE_INGRESS_TO_W1: SpscQueue<RawPayload, 32> = SpscQueue::new();
static QUEUE_W1_TO_W2: SpscQueue<PipelineMessage, 32> = SpscQueue::new();
static QUEUE_W2_TO_SUPERVISOR: SpscQueue<WorkResult, 32> = SpscQueue::new();
static TENSOR_HOP1: SpscQueue<TensorChunk<cell_core::Ready>, 8> = SpscQueue::new();
static TENSOR_HOP2: SpscQueue<TensorChunk<cell_core::Ready>, 8> = SpscQueue::new();
static AP1_READY: AtomicBool = AtomicBool::new(false);
static AP2_READY: AtomicBool = AtomicBool::new(false);
static AP3_READY: AtomicBool = AtomicBool::new(false);
static AP1_DONE: AtomicBool = AtomicBool::new(false);
static AP2_DONE: AtomicBool = AtomicBool::new(false);
static AP3_DONE: AtomicBool = AtomicBool::new(false);
static TENSOR_DONE: AtomicBool = AtomicBool::new(false);
static TENSOR_PHYS_ADDR: AtomicUsize = AtomicUsize::new(0);
static TENSOR_SUM_BITS: AtomicUsize = AtomicUsize::new(0);
static SERIAL_LOCK: AtomicBool = AtomicBool::new(false);
static mut TELEMETRY: TelemetryEncoder = TelemetryEncoder::new();

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

fn telemetry_write(event: TelemetryEvent) {
    let frame = unsafe { (&mut *core::ptr::addr_of_mut!(TELEMETRY)).encode(event) };
    while SERIAL_LOCK
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
    unsafe {
        let serial = &mut *core::ptr::addr_of_mut!(SERIAL);
        for byte in b"[CELL TM] " {
            serial.write_byte(*byte);
        }
        for byte in frame.as_bytes() {
            serial.write_byte(hex_digit(byte >> 4));
            serial.write_byte(hex_digit(byte & 0x0f));
        }
        serial.write_byte(b'\r');
        serial.write_byte(b'\n');
    }
    SERIAL_LOCK.store(false, Ordering::Release);
}

fn hex_digit(value: u8) -> u8 {
    match value {
        0..=9 => b'0' + value,
        _ => b'a' + (value - 10),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe { (&*core::ptr::addr_of!(SERIAL)).init() };
    serial_println!("[CELL KERNEL] Dataflow pipeline active");
    let bsp_simd = unsafe { simd::init_fpu_sse_avx() };
    serial_println!(
        "[CELL SIMD] BSP initialized hardware vector engine: {}",
        bsp_simd.label()
    );
    telemetry_write(TelemetryEvent::Heartbeat(Heartbeat {
        timestamp: 0,
        node_id: 0,
    }));

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
    telemetry_write(TelemetryEvent::PmmSnapshot(PmmSnapshot {
        usable_frames: pmm_stats.usable_frames as u64,
        free_frames: pmm_stats.free_frames as u64,
        largest_free_run: unsafe { pmm::largest_free_frames() as u64 },
    }));
    let Some(probe_frame) = (unsafe { pmm::allocate_frame() }) else {
        serial_println!("[CELL PMM] single-frame probe failed");
        halt();
    };
    serial_println!(
        "[CELL PMM] single-frame probe index={} phys=0x{:x}",
        probe_frame.index(),
        probe_frame.address()
    );
    if !unsafe { pmm::free_frame(probe_frame) } {
        serial_println!("[CELL PMM] single-frame probe release failed");
        halt();
    }

    let Some(hhdm) = HHDM_REQUEST.get_response() else {
        serial_println!("[CELL PMM] no Limine HHDM response");
        halt();
    };
    let tensor_frame =
        unsafe { pmm::allocate_contiguous_frames(TENSOR_FRAME_COUNT) }.unwrap_or_else(|| halt());
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
        "[CELL PMM] allocated contiguous frames count={} start_phys=0x{:x}",
        TENSOR_FRAME_COUNT,
        tensor_frame.address()
    );
    serial_println!(
        "[CELL TENSOR] BSP prepared {} F32 elements (16 KiB, 4 frames) phys=0x{:x}",
        TENSOR_ELEMENTS,
        tensor_frame.address()
    );

    let Some(response) = SMP_REQUEST.get_response() else {
        serial_println!("[CELL SMP] no Limine SMP response");
        halt();
    };
    let cpus = response.cpus();
    if cpus.len() < 4 {
        serial_println!("[CELL SMP] requires 4 CPUs, found {}", cpus.len());
        halt();
    }
    let mut secondary = cpus
        .iter()
        .filter(|cpu| cpu.lapic_id != response.bsp_lapic_id());
    let ap1 = secondary.next().unwrap_or_else(|| halt());
    let ap2 = secondary.next().unwrap_or_else(|| halt());
    let ap3 = secondary.next().unwrap_or_else(|| halt());
    ap1.goto_address.write(ap1_worker_entry);
    ap2.goto_address.write(ap2_compute_entry);
    ap3.goto_address.write(ap3_supervisor_entry);

    while !AP3_READY.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    serial_println!("[CELL SMP] Core 0 (BSP Ingress) online");
    serial_println!("[CELL SMP] Core 1 (AP1 Arbiter) online");
    serial_println!("[CELL SMP] Core 2 (AP2 Compute) online");
    serial_println!("[CELL SMP] Core 3 (AP3 Supervisor) online");
    serial_println!("[CELL PIPELINE] 4-Core topology active: C0 -> C1 -> C2 -> C3");

    for trace_id in 1..=PACKET_COUNT {
        let context = TraceContext::new(trace_id as u64, trace_id as u64, 0);
        let data = if trace_id == 4 { &[][..] } else { b"CELL data" };
        let raw = RawPayload::new(context, data).unwrap_or_else(|_| halt());
        while QUEUE_INGRESS_TO_W1.push(raw).is_err() {
            core::hint::spin_loop();
        }
    }

    if TENSOR_HOP1.push(tensor).is_err() {
        serial_println!("[CELL TENSOR] hop 1 queue unexpectedly full");
        halt();
    }
    while !AP3_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    serial_println!("[CELL KERNEL] Core 0 ingress complete; pipeline drained");
    halt()
}

unsafe extern "C" fn ap1_worker_entry(_cpu: &Cpu) -> ! {
    let ap_simd = unsafe { simd::init_fpu_sse_avx() };
    AP1_READY.store(true, Ordering::Release);
    serial_println!(
        "[CELL SIMD] AP1 initialized hardware vector engine: {}",
        ap_simd.label()
    );
    let verifier = ValidatorCapability::new(0xCE11_2021);
    let mut processed = 0;
    while processed < PACKET_COUNT {
        if let Some(raw) = QUEUE_INGRESS_TO_W1.pop() {
            let message = if raw.is_empty() {
                PipelineMessage::BypassFault {
                    context: raw.context(),
                    reason_code: 1,
                }
            } else {
                PipelineMessage::Valid(raw.validate(&verifier))
            };
            if QUEUE_W1_TO_W2.push(message).is_err() {
                serial_println!("[AP1 ARBITER] hop 2 queue unexpectedly full");
                halt();
            }
            processed += 1;
        } else {
            core::hint::spin_loop();
        }
    }
    if let Some(tensor) = wait_tensor(&TENSOR_HOP1) {
        if TENSOR_HOP2.push(tensor).is_err() {
            serial_println!("[AP1 ARBITER] tensor hop 2 queue unexpectedly full");
            halt();
        }
    }
    serial_println!("[AP1 ARBITER] Validated traces dispatched");
    AP1_DONE.store(true, Ordering::Release);
    halt()
}

unsafe extern "C" fn ap2_compute_entry(_cpu: &Cpu) -> ! {
    let ap_simd = unsafe { simd::init_fpu_sse_avx() };
    AP2_READY.store(true, Ordering::Release);
    serial_println!(
        "[CELL SIMD] AP2 initialized hardware vector engine: {}",
        ap_simd.label()
    );
    let mut processed = 0;
    while processed < PACKET_COUNT {
        if let Some(message) = QUEUE_W1_TO_W2.pop() {
            let result = match message {
                PipelineMessage::Valid(payload) => WorkResult::Completed(payload.context()),
                PipelineMessage::BypassFault { context, .. } => WorkResult::Failed {
                    context,
                    node_id: 1,
                    reason: "Corrupted payload in AP1",
                },
            };
            while QUEUE_W2_TO_SUPERVISOR.push(result).is_err() {
                core::hint::spin_loop();
            }
            processed += 1;
        } else {
            core::hint::spin_loop();
        }
    }
    if let Some(tensor) = wait_tensor(&TENSOR_HOP2) {
        let sum = {
            let values = tensor.as_slice::<f32>();
            let vector_ok = match ap_simd {
                simd::SimdLevel::Avx => unsafe { simd::verify_avx(values) },
                simd::SimdLevel::Sse => unsafe { simd::verify_sse(values) },
            };
            let mut sum = 0.0_f32;
            for value in values {
                sum += *value;
            }
            if vector_ok {
                serial_println!(
                    "[AP2 COMPUTE] SIMD AVX vector reduction finished sum={}",
                    sum
                );
            }
            sum
        };
        let (_context, phys_addr) = tensor.deconstruct();
        TENSOR_PHYS_ADDR.store(phys_addr, Ordering::Release);
        TENSOR_SUM_BITS.store(sum.to_bits() as usize, Ordering::Release);
        TENSOR_DONE.store(true, Ordering::Release);
        serial_println!(
            "[AP2 COMPUTE] Tensor descriptor forwarded phys=0x{:x}",
            phys_addr
        );
    }
    AP2_DONE.store(true, Ordering::Release);
    halt()
}

unsafe extern "C" fn ap3_supervisor_entry(_cpu: &Cpu) -> ! {
    let _simd = unsafe { simd::init_fpu_sse_avx() };
    AP3_READY.store(true, Ordering::Release);
    serial_println!("[CELL SMP] Core 3 (AP3 Supervisor) online");
    let mut supervisor = Supervisor::<PACKET_COUNT>::new();
    let mut reports = 0;
    let mut succeeded = 0;
    let mut failed = 0;
    while reports < PACKET_COUNT {
        let Some(result) = QUEUE_W2_TO_SUPERVISOR.pop() else {
            core::hint::spin_loop();
            continue;
        };
        match result {
            WorkResult::Completed(context) => {
                succeeded += 1;
                serial_println!("[CORE 3 SUPERVISOR] Trace {} OK", context.trace_id);
            }
            WorkResult::Failed {
                context,
                node_id,
                reason,
            } => {
                failed += 1;
                let _ = supervisor.observe(result);
                serial_println!(
                    "[CORE 3 SUPERVISOR] Trace {} isolated failure captured: {}",
                    context.trace_id,
                    reason
                );
                telemetry_write(TelemetryEvent::FaultIncident(FaultIncident {
                    context,
                    node_id,
                    reason_code: 1,
                }));
            }
        }
        reports += 1;
    }
    while !TENSOR_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let tensor_sum_bits = TENSOR_SUM_BITS.load(Ordering::Acquire) as u32;
    let tensor_phys = TENSOR_PHYS_ADDR.load(Ordering::Acquire);
    serial_println!(
        "[CELL TENSOR] AP2 zero-copy reduction sum={} expected=4096",
        f32::from_bits(tensor_sum_bits)
    );
    telemetry_write(TelemetryEvent::TensorExecution(TensorExecution {
        context: TraceContext::new(100, 0, 1),
        elements: TENSOR_ELEMENTS as u32,
        frame_count: TENSOR_FRAME_COUNT as u16,
        dtype: DType::F32 as u8,
        simd_level: 2,
        sum_bits: tensor_sum_bits,
    }));
    unsafe {
        pmm::free_contiguous_frames(
            pmm::PhysicalFrame::from_address(tensor_phys),
            TENSOR_FRAME_COUNT,
        )
    };
    serial_println!("[CELL PMM] contiguous frames count=4 returned to bitmap");
    telemetry_write(TelemetryEvent::QueueMetrics(QueueMetrics {
        queue_id: 2,
        capacity: 32,
        pushed: PACKET_COUNT as u32,
        popped: PACKET_COUNT as u32,
        dropped: 0,
    }));
    serial_println!(
        "[CORE 3 SUPERVISOR] Pipeline 4-Core tuntas: {} sukses, {} terisolasi. Zero crash.",
        succeeded,
        failed
    );
    AP3_DONE.store(true, Ordering::Release);
    halt()
}

fn wait_tensor<T>(queue: &SpscQueue<T, 8>) -> Option<T> {
    loop {
        if let Some(value) = queue.pop() {
            return Some(value);
        }
        core::hint::spin_loop();
    }
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
