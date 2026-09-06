use core::sync::atomic::Ordering;

use cell_core::{RawPayload, TensorOp, TraceContext};
use cell_supervisor::telemetry::{QueueMetrics, TelemetryEvent};

use crate::ap_worker;
use crate::{ipc, serial, tensor_init};

pub unsafe fn bsp_main(
    hhdm_offset: u64,
    smp_response: &limine::response::MpResponse,
    module_response: Option<&limine::response::ModuleResponse>,
) -> ! {
    let matmul_task = unsafe {
        tensor_init::create_tensor_task(
            hhdm_offset,
            100,
            TensorOp::MatMul,
            0,
            1024,
            2048,
            1024,
            |s: &mut [f32]| {
                s[0..1024].fill(1.0);
                s[1024..2048].fill(2.0);
                s[2048..3072].fill(0.0);
                s[3072..4096].fill(-14.0);
            },
        )
    };
    serial_println!(
        "[CELL PMM] allocated contiguous frames count={} start_phys=0x{:x}",
        tensor_init::TENSOR_FRAME_COUNT,
        matmul_task.tensor.phys_addr
    );
    serial_println!(
        "[BSP TENSOR] Prepared MatMul 32x32 layout (A=1.0, B=2.0, Bias=-14.0) at phys: 0x{:x}",
        matmul_task.tensor.phys_addr
    );

    let cpus = smp_response.cpus();
    if cpus.len() < 4 {
        serial_println!("[CELL SMP] requires 4 CPUs, found {}", cpus.len());
        ipc::halt();
    }
    let mut secondary = cpus
        .iter()
        .filter(|cpu| cpu.lapic_id != smp_response.bsp_lapic_id());
    let ap1 = secondary.next().unwrap_or_else(|| ipc::halt());
    let ap2 = secondary.next().unwrap_or_else(|| ipc::halt());
    let ap3 = secondary.next().unwrap_or_else(|| ipc::halt());
    ap1.goto_address.write(ap_worker::ap1_entry);
    ap2.goto_address.write(ap_worker::ap2_entry);
    ap3.goto_address.write(ap_worker::ap3_entry);

    while !ipc::AP3_READY.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    serial_println!("[CELL SMP] Core 0 (BSP Ingress) online");
    serial_println!("[CELL SMP] Core 1 (AP1 Arbiter) online");
    serial_println!("[CELL SMP] Core 2 (AP2 Compute) online");
    serial_println!("[CELL SMP] Core 3 (AP3 Supervisor) online");
    serial_println!("[CELL PIPELINE] 4-Core topology active: C0 -> C1 -> C2 -> C3");

    let mut pushed = 0_u32;
    let dropped = 0_u32;

    for trace_id in 1..=ipc::PACKET_COUNT {
        if ipc::QUEUE_INGRESS_TO_W1.is_congested() {
            let pct = ipc::QUEUE_INGRESS_TO_W1.occupancy_pct();
            serial_println!(
                "[CELL FLOW] High watermark reached ({}%) -> Backpressure active",
                pct
            );
            serial::telemetry(TelemetryEvent::QueueMetrics(QueueMetrics {
                queue_id: 0,
                capacity: 8,
                pushed,
                popped: 0,
                dropped,
                watermark_state: 1,
                _padding: [0; 3],
            }));
            while !ipc::QUEUE_INGRESS_TO_W1.is_drained() {
                core::hint::spin_loop();
            }
            let pct = ipc::QUEUE_INGRESS_TO_W1.occupancy_pct();
            serial_println!(
                "[CELL FLOW] Low watermark reached ({}%) -> Backpressure released",
                pct
            );
            serial::telemetry(TelemetryEvent::QueueMetrics(QueueMetrics {
                queue_id: 0,
                capacity: 8,
                pushed,
                popped: 0,
                dropped,
                watermark_state: 2,
                _padding: [0; 3],
            }));
        }

        let context = TraceContext::new(trace_id as u64, trace_id as u64, 0);
        let data = if trace_id == 4 { &[][..] } else { b"CELL data" };
        let raw = RawPayload::new(context, data).unwrap_or_else(|_| ipc::halt());
        while ipc::QUEUE_INGRESS_TO_W1.push(raw).is_err() {
            core::hint::spin_loop();
        }
        pushed += 1;
    }

    if ipc::TENSOR_HOP1.push(matmul_task).is_err() {
        serial_println!("[CELL TENSOR] hop 1 queue unexpectedly full");
        ipc::halt();
    }

    let attention_task = unsafe {
        tensor_init::create_tensor_task(
            hhdm_offset,
            101,
            TensorOp::Attention,
            0,
            64,
            320,
            64,
            |s: &mut [f32]| {
                s[0..64].fill(0.25);
                s[64..128].fill(1.0);
                s[128..192].fill(0.5);
                s[192..256].fill(2.0);
                s[256..320].fill(4.0);
                s[320..384].fill(0.0);
            },
        )
    };
    serial_println!(
        "[CELL PMM] allocated attention buffer count={} phys=0x{:x}",
        tensor_init::TENSOR_FRAME_COUNT,
        attention_task.tensor.phys_addr
    );
    serial_println!(
        "[BSP TENSOR] Prepared Attention layout (Q=0.25 K0=1.0 K1=0.5 V0=2.0 V1=4.0) at phys: 0x{:x}",
        attention_task.tensor.phys_addr
    );
    if ipc::TENSOR_HOP1.push(attention_task).is_err() {
        serial_println!("[CELL TENSOR] hop 1 queue full for attention task");
        ipc::halt();
    }

    let rmsnorm_task = unsafe {
        tensor_init::create_tensor_task(
            hhdm_offset,
            102,
            TensorOp::RMSNorm,
            0,
            64,
            128,
            64,
            |s: &mut [f32]| {
                s[0..64].fill(2.0);
                s[64..128].fill(0.5);
                s[128..192].fill(0.0);
            },
        )
    };
    serial_println!(
        "[CELL PMM] allocated rmsnorm buffer count={} phys=0x{:x}",
        tensor_init::TENSOR_FRAME_COUNT,
        rmsnorm_task.tensor.phys_addr
    );
    serial_println!(
        "[BSP TENSOR] Prepared RMSNorm layout (x=2.0 gamma=0.5) at phys: 0x{:x}",
        rmsnorm_task.tensor.phys_addr
    );
    if ipc::TENSOR_HOP1.push(rmsnorm_task).is_err() {
        serial_println!("[CELL TENSOR] hop 1 queue full for rmsnorm task");
        ipc::halt();
    }

    let Some(module_response) = module_response else {
        serial_println!("[MODULE LOADER] failed to load weights.bin");
        ipc::halt()
    };
    let (weights_ptr, _weight_frames) =
        unsafe { tensor_init::load_external_weights(module_response, hhdm_offset) }
            .unwrap_or_else(|_| {
                serial_println!("[MODULE LOADER] failed to load weights.bin");
                ipc::halt()
            });

    let transformer_task = unsafe {
        tensor_init::create_tensor_task(
            hhdm_offset,
            103,
            TensorOp::TransformerBlock,
            0,
            0,
            6656,
            512,
            |s: &mut [f32]| {
                s[0..512].fill(1.0);
                s[1024..1536].fill(0.0);
                s[3584..4096].fill(0.0);
                s[4096..4608].fill(0.0);
                s[5120..5632].fill(0.0);
                s[5632..6144].fill(0.0);
                s[6656..7168].fill(0.0);
                let wv =
                    core::slice::from_raw_parts(weights_ptr, tensor_init::WT_TOTAL_FLOATS);
                s[512..1024].copy_from_slice(&wv[tensor_init::WT_GAMMA1_OFF..tensor_init::WT_GAMMA1_OFF + 512]);
                s[1536..2048].copy_from_slice(&wv[tensor_init::WT_K0_OFF..tensor_init::WT_K0_OFF + 512]);
                s[2048..2560].copy_from_slice(&wv[tensor_init::WT_K1_OFF..tensor_init::WT_K1_OFF + 512]);
                s[2560..3072].copy_from_slice(&wv[tensor_init::WT_V0_OFF..tensor_init::WT_V0_OFF + 512]);
                s[3072..3584].copy_from_slice(&wv[tensor_init::WT_V1_OFF..tensor_init::WT_V1_OFF + 512]);
                s[4608..5120].copy_from_slice(&wv[tensor_init::WT_GAMMA2_OFF..tensor_init::WT_GAMMA2_OFF + 512]);
                s[6144..6656].copy_from_slice(&wv[tensor_init::WT_BIAS_OFF..tensor_init::WT_BIAS_OFF + 512]);
            },
        )
    };
    serial_println!(
        "[CELL PMM] allocated transformer block buffer count={} phys=0x{:x}",
        tensor_init::TB_FRAME_COUNT,
        transformer_task.tensor.phys_addr
    );
    serial_println!(
        "[BSP TENSOR] Prepared TransformerBlock layout at phys: 0x{:x}",
        transformer_task.tensor.phys_addr
    );
    if ipc::TENSOR_HOP1.push(transformer_task).is_err() {
        serial_println!("[CELL TENSOR] hop 1 queue full for transformer task");
        ipc::halt();
    }

    while !ipc::AP3_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    serial_println!("[CELL KERNEL] Core 0 ingress complete; pipeline drained");
    ipc::halt()
}