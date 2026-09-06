#![no_std]
#![no_main]

#[macro_use]
mod serial;
mod ap_worker;
mod bsp;
mod ipc;
mod pmm;
mod simd;
mod tensor_init;

use limine::request::{
    HhdmRequest, MemoryMapRequest, ModuleRequest, RequestsEndMarker, RequestsStartMarker,
};
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
#[link_section = ".limine_requests"]
static MODULE_REQUEST: ModuleRequest = ModuleRequest::new();

#[used]
#[link_section = ".requests_end_marker"]
static REQUESTS_END_MARKER: RequestsEndMarker = RequestsEndMarker::new();

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    assert!(BASE_REVISION.is_supported(), "Limine base revision unsupported");

    serial::init();
    serial_println!("[CELL KERNEL] Dataflow pipeline active");
    let bsp_simd = unsafe { simd::init_fpu_sse_avx() };
    serial_println!(
        "[CELL SIMD] BSP initialized hardware vector engine: {}",
        bsp_simd.label()
    );

    let Some(memory_map) = MEMMAP_REQUEST.get_response() else {
        serial_println!("[CELL PMM] no Limine memory map response");
        ipc::halt()
    };
    let pmm_stats = unsafe { pmm::initialize(memory_map.entries()) };
    serial_println!(
        "[CELL PMM] usable={} free={} max_frame={}",
        pmm_stats.usable_frames,
        pmm_stats.free_frames,
        pmm_stats.max_frame
    );
    let Some(probe_frame) = (unsafe { pmm::allocate_frame() }) else {
        serial_println!("[CELL PMM] single-frame probe failed");
        ipc::halt()
    };
    serial_println!(
        "[CELL PMM] single-frame probe index={} phys=0x{:x}",
        probe_frame.index(),
        probe_frame.address()
    );
    if !unsafe { pmm::free_frame(probe_frame) } {
        serial_println!("[CELL PMM] single-frame probe release failed");
        ipc::halt();
    }
    crate::serial::telemetry(cell_supervisor::telemetry::TelemetryEvent::Heartbeat(
        cell_supervisor::telemetry::Heartbeat {
            timestamp: 0,
            node_id: 0,
        },
    ));
    crate::serial::telemetry(cell_supervisor::telemetry::TelemetryEvent::PmmSnapshot(
        cell_supervisor::telemetry::PmmSnapshot {
            usable_frames: pmm_stats.usable_frames as u64,
            free_frames: pmm_stats.free_frames as u64,
            largest_free_run: unsafe { pmm::largest_free_frames() as u64 },
        },
    ));

    let hhdm_offset = HHDM_REQUEST
        .get_response()
        .unwrap_or_else(|| {
            serial_println!("[CELL PMM] no Limine HHDM response");
            ipc::halt()
        })
        .offset();
    let smp_response = SMP_REQUEST
        .get_response()
        .unwrap_or_else(|| {
            serial_println!("[CELL SMP] no Limine SMP response");
            ipc::halt()
        });
    let module_response = MODULE_REQUEST.get_response();

    unsafe { bsp::bsp_main(hhdm_offset, smp_response, module_response) }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    serial_println!("[CELL KERNEL] panic: {}", info);
    ipc::halt()
}