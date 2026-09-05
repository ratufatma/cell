pub const TOKENS_PER_BLOCK: usize = 4;
pub const NUM_HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const ELEM_SIZE: usize = 4;

pub const BYTES_PER_TOKEN: usize = 2 * NUM_HEADS * HEAD_DIM * ELEM_SIZE;
pub const BLOCK_SIZE_BYTES: usize = TOKENS_PER_BLOCK * BYTES_PER_TOKEN;

pub const ATTENTION_SCALE: f32 = 0.125;

pub fn softmax_4_stable(scores: &[f32; 4], n: usize) -> [f32; 4] {
    let mut weights = [0.0f32; 4];
    if n == 0 {
        return weights;
    }

    let mut max_val = scores[0];
    for i in 1..n {
        if scores[i] > max_val {
            max_val = scores[i];
        }
    }

    let mut sum_exp = 0.0f32;
    for i in 0..n {
        let x = scores[i] - max_val;
        let e = if x < -8.0 {
            0.0
        } else {
            1.0 + x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x / 24.0)))
        };
        weights[i] = e;
        sum_exp += e;
    }

    if sum_exp > 0.0 {
        for i in 0..n {
            weights[i] /= sum_exp;
        }
    }
    weights
}

pub fn dot_product_64_scalar(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..64 {
        sum += a[i] * b[i];
    }
    sum
}

pub const RMSNORM_EPS: f32 = 1e-5;

pub fn rmsnorm_scalar(x: &[f32; 64], gamma: &[f32; 64], out: &mut [f32; 64]) -> f32 {
    let mut sum_sq = 0.0f32;
    for i in 0..64 {
        sum_sq += x[i] * x[i];
    }
    let mean_sq = sum_sq / 64.0;
    let val = mean_sq + RMSNORM_EPS;
    let mut guess = val * 0.5;
    for _ in 0..8 {
        guess = (guess + val / guess) * 0.5;
    }
    let rms = guess;
    let inv_rms = 1.0 / rms;
    let mut total = 0.0f32;
    for i in 0..64 {
        out[i] = (x[i] * inv_rms) * gamma[i];
        total += out[i];
    }
    total
}

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

    #[test]
    fn softmax_single_element_is_one() {
        let scores = [3.0, 0.0, 0.0, 0.0];
        let w = softmax_4_stable(&scores, 1);
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert_eq!(w[1], 0.0);
    }

    #[test]
    fn softmax_two_equal_scores_are_half() {
        let scores = [1.0, 1.0, 0.0, 0.0];
        let w = softmax_4_stable(&scores, 2);
        assert!((w[0] - 0.5).abs() < 1e-5);
        assert!((w[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn softmax_weights_sum_to_one() {
        let scores = [2.0, 1.0, 0.5, 0.0];
        let w = softmax_4_stable(&scores, 4);
        let total: f32 = w.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_zeroes_out_large_negative() {
        let scores = [0.0, -20.0, -100.0, 0.0];
        let w = softmax_4_stable(&scores, 4);
        assert_eq!(w[2], 0.0);
        let total: f32 = w.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dot_product_64_scalar_matches_hand() {
        let a = [0.25f32; 64];
        let b = [1.0f32; 64];
        let result = dot_product_64_scalar(&a, &b);
        assert!((result - 16.0).abs() < 1e-6);
    }

    #[test]
    fn attention_two_token_checksum() {
        let q = [0.25f32; 64];
        let k0 = [1.0f32; 64];
        let k1 = [0.5f32; 64];
        let v0 = [2.0f32; 64];
        let v1 = [4.0f32; 64];

        let s0 = dot_product_64_scalar(&q, &k0) * ATTENTION_SCALE;
        let s1 = dot_product_64_scalar(&q, &k1) * ATTENTION_SCALE;
        assert!((s0 - 2.0).abs() < 1e-5);
        assert!((s1 - 1.0).abs() < 1e-5);

        let scores = [s0, s1, 0.0, 0.0];
        let w = softmax_4_stable(&scores, 2);
        let total_w: f32 = w[0] + w[1];
        assert!((total_w - 1.0).abs() < 1e-5);

        let mut out = [0.0f32; 64];
        for j in 0..64 {
            out[j] = w[0] * v0[j] + w[1] * v1[j];
        }

        let checksum: f32 = out.iter().sum();
        assert!(
            (checksum - 162.909).abs() < 0.01,
            "checksum = {}, expected ~162.909 (Taylor exp)",
            checksum
        );
    }

    #[test]
    fn rmsnorm_uniform_input_checksum() {
        let x = [2.0f32; 64];
        let gamma = [0.5f32; 64];
        let mut out = [0.0f32; 64];
        let sum = rmsnorm_scalar(&x, &gamma, &mut out);
        assert!(
            (sum - 32.0).abs() < 0.001,
            "rmsnorm sum = {}, expected ~32.0",
            sum
        );
        for i in 0..64 {
            assert!(
                (out[i] - 0.5).abs() < 0.001,
                "out[{}] = {}, expected ~0.5",
                i,
                out[i]
            );
        }
    }
}
