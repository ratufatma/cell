use core::mem::size_of;
use core::sync::atomic::Ordering;

use cell_core::{DType, MutableState, TaskDescriptor, TensorChunk, TensorOp, TensorShape, TraceContext};

use crate::ipc;

pub const WEIGHT_MAGIC: &[u8; 8] = b"CELLWGHT";
pub const WEIGHT_VERSION: u16 = 1;
pub const WEIGHT_HEADER_SIZE: usize = 64;

pub const HIDDEN_DIM: usize = 512;
pub const NUM_HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;

pub const WT_GAMMA1_OFF: usize = 0;
pub const WT_K0_OFF: usize = 512;
pub const WT_K1_OFF: usize = 1024;
pub const WT_V0_OFF: usize = 1536;
pub const WT_V1_OFF: usize = 2048;
pub const WT_GAMMA2_OFF: usize = 2560;
pub const WT_BIAS_OFF: usize = 3072;
pub const WT_WFFN_OFF: usize = 3584;
pub const WT_K2_OFF: usize = WT_WFFN_OFF + (HIDDEN_DIM * HIDDEN_DIM);
pub const WT_V2_OFF: usize = WT_K2_OFF + HIDDEN_DIM;
pub const WT_K3_OFF: usize = WT_V2_OFF + HIDDEN_DIM;
pub const WT_V3_OFF: usize = WT_K3_OFF + HIDDEN_DIM;
pub const WT_TOTAL_FLOATS: usize = WT_V3_OFF + HIDDEN_DIM;

pub const TENSOR_FRAME_COUNT: usize = 4;
pub const TB_FRAME_COUNT: usize = 7;

pub unsafe fn load_external_weights(
    module_response: &limine::response::ModuleResponse,
    hhdm_offset: u64,
) -> Result<(*const f32, usize), &'static str> {
    let Some(module) = module_response
        .modules()
        .iter()
        .find(|module| module.path().to_bytes().ends_with(b"weights.bin"))
    else {
        serial_println!("[MODULE LOADER] weights.bin module not found");
        return Err("weights.bin module not found");
    };
    let module_size = module.size() as usize;
    let module_virt = module.addr() as *const u8;
    // SAFETY: `module_virt` is the higher-half virtual mapping of the loaded
    // module, sized by the bootloader response; reading the header region is
    // in-bounds as long as the file is at least the header size.
    let header = unsafe { core::slice::from_raw_parts(module_virt, WEIGHT_HEADER_SIZE) };
    if &header[0..8] != WEIGHT_MAGIC.as_slice() {
        serial_println!("[MODULE LOADER] invalid magic in weights.bin");
        return Err("invalid magic in weights.bin");
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    if version != WEIGHT_VERSION {
        serial_println!(
            "[MODULE LOADER] unsupported weights version: {}",
            version
        );
        return Err("unsupported weights version");
    }
    if module_size < WEIGHT_HEADER_SIZE + WT_TOTAL_FLOATS * size_of::<f32>() {
        serial_println!("[MODULE LOADER] weights.bin too small: {}", module_size);
        return Err("weights.bin too small");
    }
    let frames = (module_size + 4095) / 4096;
    // SAFETY: The PMM tracks free frames; allocation of the contiguous block is
    // only performed once from the BSP before any tensor access.
    let frame = unsafe { crate::pmm::allocate_contiguous_frames(frames) }
        .ok_or("weights frame allocation failed")?;
    let block_virt = (hhdm_offset as usize + frame.address() as usize) as *mut u8;
    // SAFETY: block_virt covers `frames` fresh PMM frames and module_virt spans
    // module_size bytes; both regions are disjoint and writable/readable.
    unsafe { core::ptr::copy_nonoverlapping(module_virt, block_virt, module_size) };
    // SAFETY: Single-writer during BSP bring-up; consumers publish their reads
    // behind EXTERNAL_WEIGHTS_LOADED with acquire ordering.
    unsafe {
        ipc::EXTERNAL_WEIGHTS_PTR = block_virt.add(WEIGHT_HEADER_SIZE).cast::<f32>();
        ipc::EXTERNAL_WEIGHTS_FRAMES = frames;
    }
    ipc::EXTERNAL_WEIGHTS_LOADED.store(true, Ordering::Release);
    serial_println!(
        "[MODULE LOADER] Weights loaded into PMM: frames={} phys=0x{:x} magic=CELLWGHT verified",
        frames,
        frame.address()
    );
    Ok((unsafe { block_virt.add(WEIGHT_HEADER_SIZE).cast::<f32>() }, frames))
}

pub unsafe fn create_tensor_task<F>(
    hhdm_offset: u64,
    trace_id: u64,
    op: TensorOp,
    in_offset_a: u32,
    in_offset_b: u32,
    out_offset: u32,
    element_count: u32,
    init_fn: F,
) -> TaskDescriptor
where
    F: FnOnce(&mut [f32]),
{
    let (frame_count, shape_len) = match op {
        TensorOp::TransformerBlock => (TB_FRAME_COUNT, 7168),
        _ => (TENSOR_FRAME_COUNT, 4096),
    };
    // SAFETY: allocate_contiguous_frames returns exclusive ownership of the
    // requested frame block, never handed out twice by the bitmap allocator.
    let frame =
        unsafe { crate::pmm::allocate_contiguous_frames(frame_count) }.unwrap_or_else(|| ipc::halt());
    let virt_ptr = (hhdm_offset as usize + frame.address() as usize) as *mut u8;
    let mut chunk = TensorChunk::<MutableState>::new(
        TraceContext::new(trace_id, trace_id, 0),
        frame.address() as usize,
        virt_ptr,
        TensorShape::new_1d(shape_len),
        DType::F32,
    );
    {
        // SAFETY: The fresh 4/5-frame allocation already covers shape_len f32
        // elements, and the caller's closure is the sole writer before freeze.
        let floats = unsafe {
            core::slice::from_raw_parts_mut(
                chunk.as_mut_slice().as_mut_ptr().cast::<f32>(),
                shape_len,
            )
        };
        init_fn(floats);
    }
    TaskDescriptor {
        context: TraceContext::new(trace_id, trace_id, 0),
        op,
        tensor: chunk.freeze(),
        in_offset_a,
        in_offset_b,
        out_offset,
        element_count,
        _reserved: [0; 15],
    }
}