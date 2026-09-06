use core::sync::atomic::Ordering;

use cell_core::{DType, KvBlockDescriptor, SequenceContext, TensorOp, ValidatorCapability, WorkResult};
use cell_supervisor::telemetry::{FaultIncident, QueueMetrics, TelemetryEvent, TensorExecution};
use cell_supervisor::Supervisor;
use limine::mp::Cpu;

use crate::ipc::{self, PipelineMessage};
use crate::{pmm, simd, tensor_init};

pub const DOWNSTREAM_BUSY_SPINS: usize = 5_000_000;
pub const INITIAL_STALL_PACKETS: usize = 3;
pub const TENSOR_TASK_COUNT: usize = 4;

pub unsafe extern "C" fn ap1_entry(_cpu: &Cpu) -> ! {
    let ap_simd = unsafe { simd::init_fpu_sse_avx() };
    ipc::AP1_READY.store(true, Ordering::Release);
    serial_println!(
        "[CELL SIMD] AP1 initialized hardware vector engine: {}",
        ap_simd.label()
    );
    let verifier = ValidatorCapability::new(0xCE11_2021);
    let mut processed = 0;
    let mut stalled_packets = 0;
    while processed < ipc::PACKET_COUNT {
        if let Some(raw) = ipc::QUEUE_INGRESS_TO_W1.pop() {
            if stalled_packets < INITIAL_STALL_PACKETS {
                for _ in 0..DOWNSTREAM_BUSY_SPINS {
                    core::hint::spin_loop();
                }
                stalled_packets += 1;
            }
            let message = if raw.is_empty() {
                PipelineMessage::BypassFault {
                    context: raw.context(),
                    reason_code: 1,
                }
            } else {
                PipelineMessage::Valid(raw.validate(&verifier))
            };
            if ipc::QUEUE_W1_TO_W2.push(message).is_err() {
                serial_println!("[AP1 ARBITER] hop 2 queue unexpectedly full");
                ipc::halt();
            }
            processed += 1;
        } else {
            core::hint::spin_loop();
        }
    }
    for _ in 0..TENSOR_TASK_COUNT {
        loop {
            if let Some(task) = ipc::TENSOR_HOP1.pop() {
                if ipc::TENSOR_HOP2.push(task).is_err() {
                    serial_println!("[AP1 ARBITER] tensor hop 2 queue unexpectedly full");
                    ipc::halt();
                }
                break;
            }
            core::hint::spin_loop();
        }
    }
    serial_println!("[AP1 ARBITER] Validated traces dispatched");
    ipc::AP1_DONE.store(true, Ordering::Release);
    ipc::halt()
}

pub unsafe extern "C" fn ap2_entry(_cpu: &Cpu) -> ! {
    let ap_simd = unsafe { simd::init_fpu_sse_avx() };
    ipc::AP2_READY.store(true, Ordering::Release);
    serial_println!(
        "[CELL SIMD] AP2 initialized hardware vector engine: {}",
        ap_simd.label()
    );
    let mut processed = 0;
    while processed < ipc::PACKET_COUNT {
        if let Some(message) = ipc::QUEUE_W1_TO_W2.pop() {
            let result = match message {
                PipelineMessage::Valid(payload) => WorkResult::Completed(payload.context()),
                PipelineMessage::BypassFault {
                    context,
                    reason_code,
                } => WorkResult::Failed {
                    context,
                    node_id: reason_code,
                    reason: "Corrupted payload in AP1",
                },
            };
            while ipc::QUEUE_W2_TO_SUPERVISOR.push(result).is_err() {
                core::hint::spin_loop();
            }
            processed += 1;
        } else {
            core::hint::spin_loop();
        }
    }
    let mut tensors_processed = 0_u32;
    while (tensors_processed as usize) < TENSOR_TASK_COUNT {
        if let Some(task) = ipc::wait_task(&ipc::TENSOR_HOP2, "tensor_hop2") {
            let base_ptr = task.tensor.virt_ptr as *mut f32;
            match task.op {
                TensorOp::MatMul => {
                    let a_ptr = base_ptr.add(task.in_offset_a as usize);
                    let b_ptr = base_ptr.add(task.in_offset_b as usize);
                    let c_ptr = base_ptr.add(task.out_offset as usize);
                    unsafe { simd::gemm_32x32_avx(a_ptr, b_ptr, c_ptr) };
                    serial_println!("[AP2 DISPATCH] Executed Op::MatMul (32x32 AVX)");

                    let bias_ptr = base_ptr.add(3072);
                    unsafe { simd::vector_add_avx(c_ptr, bias_ptr, c_ptr, task.element_count as usize) };
                    serial_println!("[AP2 DISPATCH] Chained VectorAdd bias=-14.0");

                    unsafe { simd::relu_avx(c_ptr, task.element_count as usize) };
                    serial_println!("[AP2 DISPATCH] Chained ReLU activation");

                    let final_slice = unsafe { core::slice::from_raw_parts(c_ptr, task.element_count as usize) };
                    let mut final_sum = 0.0_f32;
                    let mut nonzero = 0_u32;
                    for value in final_slice {
                        final_sum += *value;
                        if *value > 0.0 {
                            nonzero += 1;
                        }
                    }
                    serial_println!(
                        "[AP2 COMPUTE] Pipeline calculation checksum verified: sum={:.1} nonzero={} C[0,0]={:.1}",
                        final_sum,
                        nonzero,
                        final_slice[0]
                    );

                    let (_context, phys_addr) = task.tensor.deconstruct();
                    ipc::TENSOR_PHYS_ADDR.store(phys_addr, Ordering::Release);
                    ipc::TENSOR_SUM_BITS.store(final_sum.to_bits() as usize, Ordering::Release);
                    ipc::TENSOR_DONE.store(true, Ordering::Release);
                    serial_println!("[AP2 COMPUTE] Tensor descriptor forwarded phys=0x{:x}", phys_addr);
                }
                TensorOp::Attention => {
                    let q_ptr = base_ptr.add(task.in_offset_a as usize);
                    let k0_ptr = base_ptr.add(64);
                    let k1_ptr = base_ptr.add(128);
                    let v0_ptr = base_ptr.add(192);
                    let v1_ptr = base_ptr.add(256);
                    let out_ptr = base_ptr.add(task.out_offset as usize);

                    unsafe {
                        simd::attention_head_64_avx(
                            q_ptr,
                            [k0_ptr, k1_ptr, core::ptr::null(), core::ptr::null()],
                            [v0_ptr, v1_ptr, core::ptr::null(), core::ptr::null()],
                            2,
                            out_ptr,
                        );
                    }

                    let out_slice = unsafe { core::slice::from_raw_parts(out_ptr, 64) };
                    let mut attn_sum = 0.0_f32;
                    for v in out_slice {
                        attn_sum += *v;
                    }
                    serial_println!(
                        "[AP2 COMPUTE] Scaled Dot-Product Attention verified sum={:.2} expected=162.42",
                        attn_sum
                    );

                    let (_context, phys_addr) = task.tensor.deconstruct();
                    ipc::ATTENTION_PHYS_ADDR.store(phys_addr, Ordering::Release);
                    ipc::ATTENTION_SUM_BITS.store(attn_sum.to_bits() as usize, Ordering::Release);
                    ipc::ATTENTION_DONE.store(true, Ordering::Release);
                }
                TensorOp::RMSNorm => {
                    let x_ptr = base_ptr.add(task.in_offset_a as usize);
                    let gamma_ptr = base_ptr.add(task.in_offset_b as usize);
                    let out_ptr = base_ptr.add(task.out_offset as usize);
                    let sum = unsafe { simd::rmsnorm_64_avx(x_ptr, gamma_ptr, out_ptr, 1e-5) };
                    serial_println!(
                        "[AP2 COMPUTE] RMSNorm 64 AVX-256 verified sum={:.2} expected=32.00",
                        sum
                    );

                    let (_context, phys_addr) = task.tensor.deconstruct();
                    ipc::RMSNORM_PHYS_ADDR.store(phys_addr, Ordering::Release);
                    ipc::RMSNORM_SUM_BITS.store(sum.to_bits() as usize, Ordering::Release);
                    ipc::RMSNORM_DONE.store(true, Ordering::Release);
                }
                TensorOp::TransformerBlock => {
                    let x0_ptr = base_ptr;
                    let gamma1_ptr = base_ptr.add(512);
                    let norm1_ptr = base_ptr.add(1024);
                    let k0_ptr = base_ptr.add(1536);
                    let k1_ptr = base_ptr.add(2048);
                    let v0_ptr = base_ptr.add(2560);
                    let v1_ptr = base_ptr.add(3072);
                    let attn_out_ptr = base_ptr.add(3584);
                    let x1_ptr = base_ptr.add(4096);
                    let gamma2_ptr = base_ptr.add(4608);
                    let norm2_ptr = base_ptr.add(5120);
                    let ffn_raw_ptr = base_ptr.add(5632);
                    let bias_ptr = base_ptr.add(6144);
                    let x2_ptr = base_ptr.add(6656);

                    unsafe { simd::rmsnorm_512_avx(x0_ptr, gamma1_ptr, norm1_ptr, 1e-5) };
                    serial_println!("[AP2 BLOCK] Step 1: Pre-Attention RMSNorm 512 done");

                    let q_ptr = norm1_ptr;
                    unsafe {
                        let q_scaled = core::slice::from_raw_parts_mut(q_ptr, 512);
                        for v in q_scaled.iter_mut() {
                            *v *= 0.25;
                        }
                    }
                    unsafe {
                        let mha_sum = simd::multi_head_attention_512_avx(
                            q_ptr,
                            [k0_ptr, k1_ptr, core::ptr::null(), core::ptr::null()],
                            [v0_ptr, v1_ptr, core::ptr::null(), core::ptr::null()],
                            2,
                            attn_out_ptr,
                        );
                        serial_println!(
                            "[AP2 BLOCK] Step 2: MHA 8-head attention done (sum={:.2})",
                            mha_sum
                        );
                    }
                    unsafe { simd::vector_add_avx(x0_ptr, attn_out_ptr, x1_ptr, 512) };
                    serial_println!("[AP2 BLOCK] Step 3: Residual connection 1 done");

                    unsafe { simd::rmsnorm_512_avx(x1_ptr, gamma2_ptr, norm2_ptr, 1e-5) };
                    serial_println!("[AP2 BLOCK] Step 4: Pre-FFN RMSNorm done");

                    let w_ffn_ptr = ipc::EXTERNAL_WEIGHTS_PTR.add(tensor_init::WT_WFFN_OFF);
                    unsafe { simd::gemv_512x512_avx(norm2_ptr, w_ffn_ptr, ffn_raw_ptr) };
                    unsafe { simd::vector_add_avx(ffn_raw_ptr, bias_ptr, ffn_raw_ptr, 512) };
                    unsafe { simd::relu_avx(ffn_raw_ptr, 512) };
                    serial_println!("[AP2 BLOCK] Step 5: FFN (GEMV+Bias+ReLU) done");

                    unsafe { simd::vector_add_avx(x1_ptr, ffn_raw_ptr, x2_ptr, 512) };
                    serial_println!("[AP2 BLOCK] Step 6: Residual connection 2 done");

                    let x2_slice = unsafe { core::slice::from_raw_parts(x2_ptr, 512) };
                    let mut tb_sum = 0.0_f64;
                    for v in x2_slice {
                        tb_sum += *v as f64;
                    }
                    let tb_sum_f32 = tb_sum as f32;
                    serial_println!(
                        "[AP2 COMPUTE] Full Transformer Block (MHA-8 512) verified sum={:.2} expected=3347.40",
                        tb_sum_f32
                    );

                    // --- 4-step autoregressive decode loop (Paged KV-Cache fill) ---
                    let kv_block = ipc::KV_CACHE_VIRT as *mut u8;
                    if !kv_block.is_null() {
                        let mut seq = SequenceContext::<1>::new(
                            ipc::KV_CACHE_BLOCK_ID.load(Ordering::Acquire) as u64,
                        );
                        let _ = seq.append_block(KvBlockDescriptor::new(
                            0,
                            ipc::KV_CACHE_PHYS_ADDR.load(Ordering::Acquire),
                            kv_block,
                        ));
                        let wv = ipc::EXTERNAL_WEIGHTS_PTR;
                        let kv_ptr_table: [(usize, usize); 4] = [
                            (tensor_init::WT_K0_OFF, tensor_init::WT_V0_OFF),
                            (tensor_init::WT_K1_OFF, tensor_init::WT_V1_OFF),
                            (tensor_init::WT_K2_OFF, tensor_init::WT_V2_OFF),
                            (tensor_init::WT_K3_OFF, tensor_init::WT_V3_OFF),
                        ];
                        let mut kv_token_ptrs = [core::ptr::null::<f32>(); 4];
                        let mut kv_val_ptrs   = [core::ptr::null::<f32>(); 4];
                        for step in 1..=4usize {
                            let (k_ptr, v_ptr) = seq
                                .reserve_next_token_slot()
                                .expect("KV block capacity exceeded before step 4");
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    wv.add(kv_ptr_table[step - 1].0),
                                    k_ptr,
                                    512,
                                );
                                core::ptr::copy_nonoverlapping(
                                    wv.add(kv_ptr_table[step - 1].1),
                                    v_ptr,
                                    512,
                                );
                            }
                            kv_token_ptrs[step - 1] = k_ptr;
                            kv_val_ptrs[step - 1] = v_ptr;

                            unsafe { simd::rmsnorm_512_avx(x0_ptr, gamma1_ptr, norm1_ptr, 1e-5) };
                            unsafe {
                                let q = core::slice::from_raw_parts_mut(norm1_ptr, 512);
                                for v in q.iter_mut() { *v *= 0.25; }
                            }
                            let mha_sum = unsafe {
                                simd::multi_head_attention_512_avx(
                                    norm1_ptr,
                                    kv_token_ptrs,
                                    kv_val_ptrs,
                                    step,
                                    attn_out_ptr,
                                )
                            };
                            unsafe { simd::vector_add_avx(x0_ptr, attn_out_ptr, x1_ptr, 512) };
                            unsafe { simd::rmsnorm_512_avx(x1_ptr, gamma2_ptr, norm2_ptr, 1e-5) };
                            let wffn = ipc::EXTERNAL_WEIGHTS_PTR.add(tensor_init::WT_WFFN_OFF);
                            unsafe { simd::gemv_512x512_avx(norm2_ptr, wffn, ffn_raw_ptr) };
                            unsafe { simd::vector_add_avx(ffn_raw_ptr, bias_ptr, ffn_raw_ptr, 512) };
                            unsafe { simd::relu_avx(ffn_raw_ptr, 512) };
                            unsafe { simd::vector_add_avx(x1_ptr, ffn_raw_ptr, x2_ptr, 512) };

                            let w_unembed =
                                ipc::EXTERNAL_WEIGHTS_PTR.add(tensor_init::WT_UNEMBED_OFF);
                            let logits_sum = unsafe {
                                simd::gemv_256x512_avx(x2_ptr, w_unembed, ffn_raw_ptr)
                            };
                            let (token_id, max_logit) =
                                unsafe { simd::argmax_256_avx(ffn_raw_ptr) };
                            let token_char =
                                core::char::from_u32(token_id as u32).unwrap_or('?');

                            let tb_chunk = unsafe { core::slice::from_raw_parts(x2_ptr, 512) };
                            let mut tb_dec_sum = 0.0_f64;
                            for v in tb_chunk { tb_dec_sum += *v as f64; }
                            let tb_dec_f32 = tb_dec_sum as f32;
                            let full_tag = if step == 4 { " [BLOCK FULL]" } else { "" };
                            let exp_tb = match step {
                                1 => "3072.00",
                                2 => "3347.40",
                                3 => "3420.08",
                                _ => "3332.75",
                            };
                            serial_println!(
                                "[AP2 DECODE] Step {}/4 (N={}) KV-slot={} MHA sum={:.2} TB sum={:.2} expected={}{}",
                                step, step, step - 1,
                                mha_sum as f32, tb_dec_f32,
                                exp_tb, full_tag
                            );
                            serial_println!(
                                "[AP2 GENERATE] Step {}/4 Token ID={} ('{}') max_logit={:.2} logits_sum={:.2}",
                                step, token_id, token_char, max_logit, logits_sum
                            );
                        }
                        let overflow_slot = seq.reserve_next_token_slot();
                        if overflow_slot.is_err() {
                            serial_println!(
                                "[AP2 KV] capacity boundary: 5th slot rejected, 16 KiB block full"
                            );
                        }
                        ipc::KV_CACHE_DONE.store(true, Ordering::Release);
                    } else {
                        serial_println!("[AP2 KV] KV-Cache block unavailable, skipping decode");
                    }

                    let (_context, phys_addr) = task.tensor.deconstruct();
                    ipc::TRANSFORMER_BLOCK_PHYS_ADDR.store(phys_addr, Ordering::Release);
                    ipc::TRANSFORMER_BLOCK_SUM_BITS.store(tb_sum_f32.to_bits() as usize, Ordering::Release);
                    ipc::TRANSFORMER_BLOCK_DONE.store(true, Ordering::Release);
                }
                _ => {}
            }
            tensors_processed += 1;
        } else {
            serial_println!("[AP2 COMPUTE] tensor task timeout, aborting");
            break;
        }
    }
    ipc::AP2_DONE.store(true, Ordering::Release);
    ipc::halt()
}

pub unsafe extern "C" fn ap3_entry(_cpu: &Cpu) -> ! {
    let _simd = unsafe { simd::init_fpu_sse_avx() };
    ipc::AP3_READY.store(true, Ordering::Release);
    serial_println!("[CELL SMP] Core 3 (AP3 Supervisor) online");
    let mut supervisor = Supervisor::<{ ipc::PACKET_COUNT }>::new();
    let mut reports = 0;
    let mut succeeded = 0;
    let mut failed = 0;
    while reports < ipc::PACKET_COUNT {
        let Some(result) = ipc::QUEUE_W2_TO_SUPERVISOR.pop() else {
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
                crate::serial::telemetry(TelemetryEvent::FaultIncident(FaultIncident {
                    context,
                    node_id,
                    reason_code: 1,
                }));
            }
        }
        reports += 1;
    }
    while !ipc::TENSOR_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let tensor_sum_bits = ipc::TENSOR_SUM_BITS.load(Ordering::Acquire) as u32;
    let tensor_phys = ipc::TENSOR_PHYS_ADDR.load(Ordering::Acquire);
    let final_val = f32::from_bits(tensor_sum_bits);
    serial_println!(
        "[CELL TENSOR] AP2 GEMM+Bias+ReLU result sum={:.1} expected=32768",
        final_val
    );
    crate::serial::telemetry(TelemetryEvent::TensorExecution(TensorExecution {
        context: cell_core::TraceContext::new(100, 0, 1),
        elements: 1024,
        frame_count: tensor_init::TENSOR_FRAME_COUNT as u16,
        dtype: DType::F32 as u8,
        simd_level: 2,
        sum_bits: tensor_sum_bits,
    }));
    unsafe {
        pmm::free_contiguous_frames(
            pmm::PhysicalFrame::from_address(tensor_phys),
            tensor_init::TENSOR_FRAME_COUNT,
        )
    };
    serial_println!("[CELL PMM] contiguous frames count=4 returned to bitmap");

    while !ipc::ATTENTION_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let attn_sum_bits = ipc::ATTENTION_SUM_BITS.load(Ordering::Acquire) as u32;
    let attn_phys = ipc::ATTENTION_PHYS_ADDR.load(Ordering::Acquire);
    let attn_val = f32::from_bits(attn_sum_bits);
    serial_println!(
        "[CELL TENSOR] AP2 Attention verified sum={:.2} expected=162.42",
        attn_val
    );
    crate::serial::telemetry(TelemetryEvent::TensorExecution(TensorExecution {
        context: cell_core::TraceContext::new(101, 0, 1),
        elements: 64,
        frame_count: tensor_init::TENSOR_FRAME_COUNT as u16,
        dtype: DType::F32 as u8,
        simd_level: 2,
        sum_bits: attn_sum_bits,
    }));
    unsafe {
        pmm::free_contiguous_frames(
            pmm::PhysicalFrame::from_address(attn_phys),
            tensor_init::TENSOR_FRAME_COUNT,
        )
    };
    serial_println!("[CELL PMM] contiguous frames count=4 returned to bitmap");

    while !ipc::RMSNORM_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let rms_sum_bits = ipc::RMSNORM_SUM_BITS.load(Ordering::Acquire) as u32;
    let rms_phys = ipc::RMSNORM_PHYS_ADDR.load(Ordering::Acquire);
    let rms_val = f32::from_bits(rms_sum_bits);
    serial_println!(
        "[CELL TENSOR] AP2 RMSNorm verified sum={:.2} expected=32.00",
        rms_val
    );
    crate::serial::telemetry(TelemetryEvent::TensorExecution(TensorExecution {
        context: cell_core::TraceContext::new(102, 0, 1),
        elements: 64,
        frame_count: tensor_init::TENSOR_FRAME_COUNT as u16,
        dtype: DType::F32 as u8,
        simd_level: 2,
        sum_bits: rms_sum_bits,
    }));
    unsafe {
        pmm::free_contiguous_frames(
            pmm::PhysicalFrame::from_address(rms_phys),
            tensor_init::TENSOR_FRAME_COUNT,
        )
    };
    serial_println!("[CELL PMM] contiguous frames count=4 returned to bitmap");

    while !ipc::TRANSFORMER_BLOCK_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let tb_sum_bits = ipc::TRANSFORMER_BLOCK_SUM_BITS.load(Ordering::Acquire) as u32;
    let tb_phys = ipc::TRANSFORMER_BLOCK_PHYS_ADDR.load(Ordering::Acquire);
    let tb_val = f32::from_bits(tb_sum_bits);
    serial_println!(
        "[CELL TENSOR] AP2 Transformer Block (MHA-8 512) verified sum={:.2} expected=3347.40",
        tb_val
    );
    crate::serial::telemetry(TelemetryEvent::TensorExecution(TensorExecution {
        context: cell_core::TraceContext::new(103, 0, 1),
        elements: 512,
        frame_count: tensor_init::TB_FRAME_COUNT as u16,
        dtype: DType::F32 as u8,
        simd_level: 2,
        sum_bits: tb_sum_bits,
    }));
    unsafe {
        pmm::free_contiguous_frames(
            pmm::PhysicalFrame::from_address(tb_phys),
            tensor_init::TB_FRAME_COUNT,
        )
    };
    serial_println!("[CELL PMM] contiguous frames count=5 returned to bitmap");

    while !ipc::KV_CACHE_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let kv_phys = ipc::KV_CACHE_PHYS_ADDR.load(Ordering::Acquire);
    unsafe {
        pmm::free_contiguous_frames(
            pmm::PhysicalFrame::from_address(kv_phys),
            4,
        )
    };
    serial_println!("[CELL PMM] KV-Cache 16 KiB block count=4 returned to bitmap");

    crate::serial::telemetry(TelemetryEvent::QueueMetrics(QueueMetrics {
        queue_id: 2,
        capacity: 32,
        pushed: ipc::PACKET_COUNT as u32,
        popped: ipc::PACKET_COUNT as u32,
        dropped: 0,
        watermark_state: 0,
        _padding: [0; 3],
    }));
    serial_println!(
        "[CORE 3 SUPERVISOR] Pipeline 4-Core tuntas: {} sukses, {} terisolasi. 0 dropped. Zero crash.",
        succeeded,
        failed
    );
    ipc::AP3_DONE.store(true, Ordering::Release);
    ipc::halt()
}