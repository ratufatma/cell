use core::marker::PhantomData;
use core::mem::size_of;
use core::slice;

use crate::TraceContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    U8 = 4,
}

impl DType {
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: [usize; 4],
    pub rank: u8,
}

impl TensorShape {
    pub const fn new_1d(len: usize) -> Self {
        Self {
            dims: [len, 0, 0, 0],
            rank: 1,
        }
    }

    pub const fn new_2d(rows: usize, cols: usize) -> Self {
        Self {
            dims: [rows, cols, 0, 0],
            rank: 2,
        }
    }

    pub const fn new_3d(batch: usize, sequence: usize, depth: usize) -> Self {
        Self {
            dims: [batch, sequence, depth, 0],
            rank: 3,
        }
    }

    pub const fn new_4d(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        Self {
            dims: [batch, channels, height, width],
            rank: 4,
        }
    }

    pub fn total_elements(&self) -> usize {
        let mut total: usize = 1;
        let mut index = 0;
        while index < self.rank as usize {
            total = total.saturating_mul(self.dims[index]);
            index += 1;
        }
        total
    }
}

pub struct MutableState;
pub struct Ready;

#[repr(C, align(64))]
pub struct TensorChunk<State> {
    pub context: TraceContext,
    pub shape: TensorShape,
    pub dtype: DType,
    pub phys_addr: usize,
    pub virt_ptr: *mut u8,
    pub byte_len: usize,
    marker: PhantomData<State>,
}

// SAFETY: TensorChunk transfers exclusive ownership of its physical/virtual
// buffer with the descriptor. The owner must not alias that buffer elsewhere.
unsafe impl<State> Send for TensorChunk<State> {}

// SAFETY: Shared access only exposes immutable metadata and Ready slices; the
// buffer ownership contract prevents concurrent mutation through another owner.
unsafe impl<State> Sync for TensorChunk<State> {}

impl TensorChunk<MutableState> {
    pub fn new(
        context: TraceContext,
        phys_addr: usize,
        virt_ptr: *mut u8,
        shape: TensorShape,
        dtype: DType,
    ) -> Self {
        let byte_len = shape.total_elements() * dtype.size_in_bytes();
        Self {
            context,
            shape,
            dtype,
            phys_addr,
            virt_ptr,
            byte_len,
            marker: PhantomData,
        }
    }

    /// Returns the owned tensor storage as mutable bytes.
    ///
    /// The constructor's caller must provide a writable allocation of at least
    /// `byte_len` bytes and retain exclusive ownership for this borrow.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: TensorChunk owns the buffer described by virt_ptr and byte_len.
        unsafe { slice::from_raw_parts_mut(self.virt_ptr, self.byte_len) }
    }

    pub fn freeze(self) -> TensorChunk<Ready> {
        TensorChunk {
            context: self.context,
            shape: self.shape,
            dtype: self.dtype,
            phys_addr: self.phys_addr,
            virt_ptr: self.virt_ptr,
            byte_len: self.byte_len,
            marker: PhantomData,
        }
    }
}

impl TensorChunk<Ready> {
    /// Returns the frozen tensor storage as elements of the matching Rust type.
    ///
    /// # Panics
    /// Panics if `T` does not match the tensor dtype, the byte length is not a
    /// multiple of `size_of::<T>()`, or the pointer is not correctly aligned.
    ///
    /// The frozen backing allocation must contain initialized values of type `T`.
    pub fn as_slice<T>(&self) -> &[T] {
        assert_eq!(size_of::<T>(), self.dtype.size_in_bytes());
        assert_eq!(self.byte_len % size_of::<T>(), 0);
        assert_eq!(self.virt_ptr as usize % core::mem::align_of::<T>(), 0);
        // SAFETY: Ready owns the initialized buffer, and the checks above
        // establish the element count and alignment invariants.
        unsafe { slice::from_raw_parts(self.virt_ptr.cast(), self.byte_len / size_of::<T>()) }
    }

    pub fn deconstruct(self) -> (TraceContext, usize) {
        (self.context, self.phys_addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_shape_elements_and_byte_lengths() {
        assert_eq!(TensorShape::new_1d(7).total_elements(), 7);
        assert_eq!(TensorShape::new_2d(3, 4).total_elements(), 12);
        assert_eq!(TensorShape::new_3d(2, 3, 4).total_elements(), 24);
        assert_eq!(TensorShape::new_4d(2, 3, 4, 5).total_elements(), 120);

        let mut buffer = [0_u8; 64];
        let chunk = TensorChunk::<MutableState>::new(
            TraceContext::new(1, 2, 3),
            0x4000,
            buffer.as_mut_ptr(),
            TensorShape::new_2d(4, 4),
            DType::U8,
        );
        assert_eq!(chunk.byte_len, 16);
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
        assert_eq!(DType::I8.size_in_bytes(), 1);
    }

    #[test]
    fn freezes_and_reads_owned_buffer_without_copying() {
        let mut buffer = [0_u8; 64];
        let context = TraceContext::new(42, 100, 7);
        let mut chunk = TensorChunk::<MutableState>::new(
            context,
            0x8000,
            buffer.as_mut_ptr(),
            TensorShape::new_1d(8),
            DType::U8,
        );

        chunk
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ready = chunk.freeze();
        let values = ready.as_slice::<u8>();
        assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(values.as_ptr(), buffer.as_ptr());

        let (returned_context, phys_addr) = ready.deconstruct();
        assert_eq!(returned_context, context);
        assert_eq!(phys_addr, 0x8000);
    }
}
