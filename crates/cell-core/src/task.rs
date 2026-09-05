use crate::tensor::{Ready, TensorChunk};
use crate::TraceContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorOp {
    MatMul = 0x01,
    VectorAdd = 0x02,
    ReLU = 0x03,
    Attention = 0x04,
    RMSNorm = 0x05,
    TransformerBlock = 0x06,
}

#[repr(C, align(64))]
pub struct TaskDescriptor {
    pub context: TraceContext,
    pub op: TensorOp,
    pub tensor: TensorChunk<Ready>,
    pub in_offset_a: u32,
    pub in_offset_b: u32,
    pub out_offset: u32,
    pub element_count: u32,
    pub _reserved: [u8; 15],
}

unsafe impl Send for TaskDescriptor {}
unsafe impl Sync for TaskDescriptor {}
