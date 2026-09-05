pub const TOKENS_PER_BLOCK: usize = 4;
pub const NUM_HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const ELEM_SIZE: usize = 4;

pub const BYTES_PER_TOKEN: usize = 2 * NUM_HEADS * HEAD_DIM * ELEM_SIZE;
pub const BLOCK_SIZE_BYTES: usize = TOKENS_PER_BLOCK * BYTES_PER_TOKEN;

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct KvBlockDescriptor {
    pub block_id: u32,
    pub phys_addr: usize,
    pub virt_ptr: *mut u8,
    pub token_capacity: u16,
    pub tokens_written: u16,
}

unsafe impl Send for KvBlockDescriptor {}
unsafe impl Sync for KvBlockDescriptor {}

impl KvBlockDescriptor {
    pub fn new(block_id: u32, phys_addr: usize, virt_ptr: *mut u8) -> Self {
        Self {
            block_id,
            phys_addr,
            virt_ptr,
            token_capacity: TOKENS_PER_BLOCK as u16,
            tokens_written: 0,
        }
    }

    pub const fn remaining(&self) -> u16 {
        self.token_capacity - self.tokens_written
    }

    pub const fn is_full(&self) -> bool {
        self.tokens_written >= self.token_capacity
    }
}

pub struct SequenceContext<const MAX_BLOCKS: usize> {
    pub sequence_id: u64,
    pub total_tokens: usize,
    pub block_count: usize,
    pub block_table: [Option<KvBlockDescriptor>; MAX_BLOCKS],
}

impl<const MAX_BLOCKS: usize> SequenceContext<MAX_BLOCKS> {
    pub const fn new(sequence_id: u64) -> Self {
        const NONE_BLOCK: Option<KvBlockDescriptor> = None;
        Self {
            sequence_id,
            total_tokens: 0,
            block_count: 0,
            block_table: [NONE_BLOCK; MAX_BLOCKS],
        }
    }

    pub fn append_block(&mut self, block: KvBlockDescriptor) -> Result<(), &'static str> {
        if self.block_count >= MAX_BLOCKS {
            return Err("MAX_BLOCKS capacity reached");
        }
        self.block_table[self.block_count] = Some(block);
        self.block_count += 1;
        Ok(())
    }

    pub fn reserve_next_token_slot(
        &mut self,
    ) -> Result<(*mut f32, *mut f32), &'static str> {
        if self.block_count == 0 {
            return Err("No physical blocks allocated");
        }

        let current_idx = self.block_count - 1;
        let block = self.block_table[current_idx]
            .as_mut()
            .ok_or("Invalid block slot")?;

        if block.is_full() {
            return Err("Current block full, allocate next block");
        }

        let token_idx = block.tokens_written as usize;
        let token_base = unsafe { block.virt_ptr.add(token_idx * BYTES_PER_TOKEN) };

        let k_ptr = token_base as *mut f32;
        let v_ptr = unsafe { token_base.add(NUM_HEADS * HEAD_DIM) } as *mut f32;

        block.tokens_written += 1;
        self.total_tokens += 1;

        Ok((k_ptr, v_ptr))
    }

    pub const fn total_blocks_bytes(&self) -> usize {
        self.block_count * BLOCK_SIZE_BYTES
    }

    pub fn last_block(&self) -> Result<&KvBlockDescriptor, &'static str> {
        if self.block_count == 0 {
            return Err("No blocks in sequence");
        }
        self.block_table[self.block_count - 1]
            .as_ref()
            .ok_or("Invalid block slot")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_empty_sequence_context() {
        let ctx = SequenceContext::<8>::new(42);
        assert_eq!(ctx.sequence_id, 42);
        assert_eq!(ctx.total_tokens, 0);
        assert_eq!(ctx.block_count, 0);
    }

    #[test]
    fn appends_blocks_and_fills_tokens_without_heap() {
        let mut buffer = [0u8; BLOCK_SIZE_BYTES];
        let ptr = buffer.as_mut_ptr();

        let mut ctx = SequenceContext::<8>::new(1);
        let block = KvBlockDescriptor::new(0, 0x1000, ptr);
        ctx.append_block(block).unwrap();

        for i in 0..TOKENS_PER_BLOCK {
            let (k_ptr, v_ptr) = ctx.reserve_next_token_slot().unwrap();
            unsafe {
                *k_ptr = (i as f32) * 1.0;
                *v_ptr = (i as f32) * 2.0;
            }
        }

        assert_eq!(ctx.total_tokens, TOKENS_PER_BLOCK);
        assert_eq!(ctx.block_count, 1);

        let block = ctx.last_block().unwrap();
        assert!(block.is_full());
        assert_eq!(block.remaining(), 0);
    }

    #[test]
    fn demands_new_block_when_capacity_exceeded() {
        let mut buffer = [0u8; BLOCK_SIZE_BYTES];
        let ptr = buffer.as_mut_ptr();

        let mut ctx = SequenceContext::<8>::new(2);
        let block = KvBlockDescriptor::new(0, 0x2000, ptr);
        ctx.append_block(block).unwrap();

        for _ in 0..TOKENS_PER_BLOCK {
            let _ = ctx.reserve_next_token_slot().unwrap();
        }

        let err = ctx.reserve_next_token_slot().unwrap_err();
        assert_eq!(err, "Current block full, allocate next block");
    }

    #[test]
    fn multi_block_sequence_manages_tokens_across_blocks() {
        let mut buf_a = [0u8; BLOCK_SIZE_BYTES];
        let mut buf_b = [0u8; BLOCK_SIZE_BYTES];
        let ptr_a = buf_a.as_mut_ptr();
        let ptr_b = buf_b.as_mut_ptr();

        let mut ctx = SequenceContext::<8>::new(3);
        ctx.append_block(KvBlockDescriptor::new(0, 0x3000, ptr_a))
            .unwrap();

        for _ in 0..TOKENS_PER_BLOCK {
            let _ = ctx.reserve_next_token_slot().unwrap();
        }

        ctx.append_block(KvBlockDescriptor::new(1, 0x4000, ptr_b))
            .unwrap();

        for _ in 0..TOKENS_PER_BLOCK {
            let _ = ctx.reserve_next_token_slot().unwrap();
        }

        assert_eq!(ctx.total_tokens, TOKENS_PER_BLOCK * 2);
        assert_eq!(ctx.block_count, 2);
        assert_eq!(ctx.total_blocks_bytes(), BLOCK_SIZE_BYTES * 2);
    }

    #[test]
    fn rejects_append_when_table_full() {
        let mut ctx = SequenceContext::<2>::new(5);
        let dummy = [0u8; BLOCK_SIZE_BYTES];
        let ptr = dummy.as_ptr() as *mut u8;

        ctx.append_block(KvBlockDescriptor::new(0, 0x5000, ptr))
            .unwrap();
        ctx.append_block(KvBlockDescriptor::new(1, 0x6000, ptr))
            .unwrap();

        let err = ctx
            .append_block(KvBlockDescriptor::new(2, 0x7000, ptr))
            .unwrap_err();
        assert_eq!(err, "MAX_BLOCKS capacity reached");
    }

    #[test]
    fn slot_writes_do_not_overlap_in_memory() {
        let mut buffer = [0u8; BLOCK_SIZE_BYTES];
        let ptr = buffer.as_mut_ptr();

        let mut ctx = SequenceContext::<8>::new(6);
        ctx.append_block(KvBlockDescriptor::new(0, 0x8000, ptr))
            .unwrap();

        let (k0, v0) = ctx.reserve_next_token_slot().unwrap();
        let (k1, v1) = ctx.reserve_next_token_slot().unwrap();

        let k0_addr = k0 as usize;
        let k1_addr = k1 as usize;
        let v0_addr = v0 as usize;
        let v1_addr = v1 as usize;

        assert!(k1_addr > k0_addr + BYTES_PER_TOKEN / 2);
        assert!(v1_addr > v0_addr + BYTES_PER_TOKEN / 2);
        assert_ne!(k0_addr, v0_addr);
        assert_ne!(k1_addr, v1_addr);
    }
}
