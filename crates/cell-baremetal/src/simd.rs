use core::arch::asm;
use cell_core::softmax_4_stable;

use crate::tensor_init::{HEAD_DIM, HIDDEN_DIM, NUM_HEADS};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdLevel {
    Sse,
    Avx,
}

impl SimdLevel {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sse => "SSE (128-bit)",
            Self::Avx => "AVX (256-bit)",
        }
    }
}

pub unsafe fn init_fpu_sse_avx() -> SimdLevel {
    let features = core::arch::x86_64::__cpuid(1);
    let has_xsave = features.ecx & (1 << 26) != 0;
    let has_avx = features.ecx & (1 << 28) != 0;
    let avx_enabled = has_xsave && has_avx;

    let mut cr0: u64;
    unsafe {
        asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
    }
    cr0 &= !(1 << 2);
    cr0 |= 1 << 1;
    cr0 &= !(1 << 3);
    unsafe {
        asm!("mov cr0, {}", in(reg) cr0, options(nomem, nostack, preserves_flags));
    }

    let mut cr4: u64;
    unsafe {
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
    }
    cr4 |= (1 << 9) | (1 << 10);
    if avx_enabled {
        cr4 |= 1 << 18;
    }
    unsafe {
        asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack, preserves_flags));
    }

    if avx_enabled {
        unsafe {
            asm!(
                "xsetbv",
                in("ecx") 0_u32,
                in("eax") 0x7_u32,
                in("edx") 0x0_u32,
                options(nomem, nostack, preserves_flags)
            );
        }
        SimdLevel::Avx
    } else {
        SimdLevel::Sse
    }
}

pub unsafe fn gemm_32x32_avx(a: *const f32, b: *const f32, c: *mut f32) {
    const N: usize = 32;
    const K: usize = 32;
    const M: usize = 32;

    for i in 0..(M * N) {
        *c.add(i) = 0.0;
    }

    for i in 0..M {
        for k in 0..K {
            let a_val = a.add(i * K + k);
            for j in (0..N).step_by(8) {
                let b_idx = k * N + j;
                let c_idx = i * N + j;
                unsafe {
                    asm!(
                        "vmovss xmm4, [{a_ptr}]",
                        "vbroadcastss ymm0, xmm4",
                        "vmovups ymm1, [{b}]",
                        "vmovups ymm2, [{c}]",
                        "vmulps ymm3, ymm0, ymm1",
                        "vaddps ymm2, ymm2, ymm3",
                        "vmovups [{c}], ymm2",
                        "vzeroupper",
                        a_ptr = in(reg) a_val,
                        b = in(reg) b.add(b_idx),
                        c = in(reg) c.add(c_idx),
                        out("xmm4") _,
                        out("ymm0") _,
                        out("ymm1") _,
                        out("ymm2") _,
                        out("ymm3") _,
                        options(nostack),
                    );
                }
            }
        }
    }
}

pub unsafe fn vector_add_avx(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    assert_eq!(count % 8, 0, "vector_add_avx requires count divisible by 8");
    for i in (0..count).step_by(8) {
        unsafe {
            asm!(
                "vmovups ymm0, [{a}]",
                "vmovups ymm1, [{b}]",
                "vaddps ymm2, ymm0, ymm1",
                "vmovups [{out}], ymm2",
                "vzeroupper",
                a = in(reg) a.add(i),
                b = in(reg) b.add(i),
                out = in(reg) out.add(i),
                out("ymm0") _,
                out("ymm1") _,
                out("ymm2") _,
                options(nostack),
            );
        }
    }
}

pub unsafe fn relu_avx(data: *mut f32, count: usize) {
    assert_eq!(count % 8, 0, "relu_avx requires count divisible by 8");
    for i in (0..count).step_by(8) {
        unsafe {
            asm!(
                "vxorps ymm1, ymm1, ymm1",
                "vmovups ymm0, [{data}]",
                "vmaxps ymm2, ymm0, ymm1",
                "vmovups [{data}], ymm2",
                "vzeroupper",
                data = in(reg) data.add(i),
                out("ymm0") _,
                out("ymm1") _,
                out("ymm2") _,
                options(nostack),
            );
        }
    }
}

#[inline(always)]
pub unsafe fn dot_product_64_avx(a: *const f32, b: *const f32) -> f32 {
    let sum: f32 = 0.0;
    unsafe {
        asm!(
            "vxorps ymm0, ymm0, ymm0",
            "xor {idx}, {idx}",
            "2:",
            "vmovups ymm1, [{a} + {idx}]",
            "vmovups ymm2, [{b} + {idx}]",
            "vmulps ymm1, ymm1, ymm2",
            "vaddps ymm0, ymm0, ymm1",
            "add {idx}, 32",
            "cmp {idx}, 256",
            "jl 2b",
            "vextractf128 xmm1, ymm0, 1",
            "vaddps xmm0, xmm0, xmm1",
            "vhaddps xmm0, xmm0, xmm0",
            "vhaddps xmm0, xmm0, xmm0",
            "vmovss [{sum_ptr}], xmm0",
            "vzeroupper",
            a = in(reg) a,
            b = in(reg) b,
            idx = out(reg) _,
            sum_ptr = in(reg) &sum as *const f32,
            out("ymm0") _,
            out("ymm1") _,
            out("ymm2") _,
            options(nostack),
        );
    }
    sum
}

pub unsafe fn attention_head_64_avx(
    q: *const f32,
    k_ptrs: [*const f32; 4],
    v_ptrs: [*const f32; 4],
    num_tokens: usize,
    out: *mut f32,
) {
    const SCALE: f32 = 0.125;
    let n = num_tokens.min(4);

    for i in 0..64 {
        *out.add(i) = 0.0;
    }

    let mut scores = [0.0f32; 4];
    for i in 0..n {
        let dot = dot_product_64_avx(q, k_ptrs[i]);
        scores[i] = dot * SCALE;
    }

    let weights = softmax_4_stable(scores, n);

    for i in 0..n {
        let w = weights[i];
        if w == 0.0 {
            continue;
        }
        for j in (0..64).step_by(8) {
            unsafe {
                asm!(
                    "vbroadcastss ymm0, [{w_ptr}]",
                    "vmovups ymm1, [{v}]",
                    "vmulps ymm0, ymm0, ymm1",
                    "vmovups ymm2, [{out}]",
                    "vaddps ymm0, ymm0, ymm2",
                    "vmovups [{out}], ymm0",
                    "vzeroupper",
                    w_ptr = in(reg) &w as *const f32,
                    v = in(reg) v_ptrs[i].add(j),
                    out = in(reg) out.add(j),
                    out("ymm0") _,
                    out("ymm1") _,
                    out("ymm2") _,
                    options(nostack),
                );
            }
        }
    }
}

pub unsafe fn rmsnorm_64_avx(
    x: *const f32,
    gamma: *const f32,
    out: *mut f32,
    eps: f32,
) -> f32 {
    let sum_sq: f32 = 0.0;
    unsafe {
        asm!(
            "vxorps ymm0, ymm0, ymm0",
            "xor {idx}, {idx}",
            "3:",
            "vmovups ymm1, [{x} + {idx}]",
            "vmulps ymm1, ymm1, ymm1",
            "vaddps ymm0, ymm0, ymm1",
            "add {idx}, 32",
            "cmp {idx}, 256",
            "jl 3b",
            "vextractf128 xmm1, ymm0, 1",
            "vaddps xmm0, xmm0, xmm1",
            "vhaddps xmm0, xmm0, xmm0",
            "vhaddps xmm0, xmm0, xmm0",
            "vmovss [{out_sum}], xmm0",
            "vzeroupper",
            x = in(reg) x,
            idx = out(reg) _,
            out_sum = in(reg) &sum_sq as *const f32,
            out("ymm0") _,
            out("ymm1") _,
            options(nostack),
        );
    }

    let mean_sq = sum_sq / 64.0;
    let val = mean_sq + eps;
    let mut rms: f32 = 0.0;
    unsafe {
        asm!(
            "vmovss xmm0, [{val_ptr}]",
            "vsqrtss xmm0, xmm0, xmm0",
            "vmovss [{rms_ptr}], xmm0",
            val_ptr = in(reg) &val as *const f32,
            rms_ptr = in(reg) &mut rms as *mut f32,
            out("xmm0") _,
            options(nostack),
        );
    }
    let inv_rms = 1.0 / rms;

    unsafe {
        asm!(
            "vbroadcastss ymm0, [{inv_ptr}]",
            "xor {idx}, {idx}",
            "2:",
            "vmovups ymm1, [{x} + {idx}]",
            "vmulps ymm1, ymm1, ymm0",
            "vmovups ymm2, [{gamma} + {idx}]",
            "vmulps ymm1, ymm1, ymm2",
            "vmovups [{out} + {idx}], ymm1",
            "add {idx}, 32",
            "cmp {idx}, 256",
            "jl 2b",
            "vzeroupper",
            inv_ptr = in(reg) &inv_rms as *const f32,
            x = in(reg) x,
            gamma = in(reg) gamma,
            out = in(reg) out,
            idx = out(reg) _,
            out("ymm0") _,
            out("ymm1") _,
            out("ymm2") _,
            options(nostack),
        );
    }

    let out_slice = core::slice::from_raw_parts(out, 64);
    let mut checksum = 0.0f32;
    for v in out_slice {
        checksum += *v;
    }
    checksum
}

pub unsafe fn multi_head_attention_512_avx(
    q: *const f32,
    k_tokens: [*const f32; 4],
    v_tokens: [*const f32; 4],
    num_tokens: usize,
    out: *mut f32,
) -> f32 {
    let mut total_sum: f64 = 0.0;
    for h in 0..NUM_HEADS {
        let head_offset = h * HEAD_DIM;
        let q_h = q.add(head_offset);
        let out_h = out.add(head_offset);

        let k_h = [
            if !k_tokens[0].is_null() {
                k_tokens[0].add(head_offset)
            } else {
                core::ptr::null()
            },
            if !k_tokens[1].is_null() {
                k_tokens[1].add(head_offset)
            } else {
                core::ptr::null()
            },
            core::ptr::null(),
            core::ptr::null(),
        ];
        let v_h = [
            if !v_tokens[0].is_null() {
                v_tokens[0].add(head_offset)
            } else {
                core::ptr::null()
            },
            if !v_tokens[1].is_null() {
                v_tokens[1].add(head_offset)
            } else {
                core::ptr::null()
            },
            core::ptr::null(),
            core::ptr::null(),
        ];

        attention_head_64_avx(q_h, k_h, v_h, num_tokens, out_h);

        for i in 0..HEAD_DIM {
            total_sum += *out_h.add(i) as f64;
        }
    }
    total_sum as f32
}

pub unsafe fn rmsnorm_512_avx(
    x: *const f32,
    gamma: *const f32,
    out: *mut f32,
    eps: f32,
) -> f32 {
    let mut sum_sq: f32 = 0.0;
    for c in 0..(HIDDEN_DIM / HEAD_DIM) {
        let ptr = x.add(c * HEAD_DIM);
        sum_sq += dot_product_64_avx(ptr, ptr);
    }
    let mean_sq = sum_sq / HIDDEN_DIM as f32;
    let val = mean_sq + eps;
    let mut rms: f32 = 0.0;
    unsafe {
        asm!(
            "vmovss xmm0, [{val_ptr}]",
            "vsqrtss xmm0, xmm0, xmm0",
            "vmovss [{rms_ptr}], xmm0",
            val_ptr = in(reg) &val as *const f32,
            rms_ptr = in(reg) &mut rms as *mut f32,
            out("xmm0") _,
            options(nostack),
        );
    }
    let inv_rms = 1.0 / rms;

    unsafe {
        asm!(
            "vbroadcastss ymm0, [{inv_ptr}]",
            "xor {idx}, {idx}",
            "2:",
            "vmovups ymm1, [{x} + {idx}]",
            "vmulps ymm1, ymm1, ymm0",
            "vmovups ymm2, [{gamma} + {idx}]",
            "vmulps ymm1, ymm1, ymm2",
            "vmovups [{out} + {idx}], ymm1",
            "add {idx}, 32",
            "cmp {idx}, 2048",
            "jl 2b",
            "vzeroupper",
            inv_ptr = in(reg) &inv_rms as *const f32,
            x = in(reg) x,
            gamma = in(reg) gamma,
            out = in(reg) out,
            idx = out(reg) _,
            out("ymm0") _,
            out("ymm1") _,
            out("ymm2") _,
            options(nostack),
        );
    }

    let out_slice = core::slice::from_raw_parts(out, HIDDEN_DIM);
    let mut checksum = 0.0f32;
    for v in out_slice {
        checksum += *v;
    }
    checksum
}

pub unsafe fn gemv_512x512_avx(x: *const f32, w: *const f32, out: *mut f32) {
    for row in 0..HIDDEN_DIM {
        let w_row = w.add(row * HIDDEN_DIM);
        let mut row_sum: f32 = 0.0;
        for c in 0..(HIDDEN_DIM / HEAD_DIM) {
            row_sum += dot_product_64_avx(x.add(c * HEAD_DIM), w_row.add(c * HEAD_DIM));
        }
        *out.add(row) = row_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_64_avx_matches_scalar() {
        let a = [0.25f32; 64];
        let b = [1.0f32; 64];
        let result = unsafe { dot_product_64_avx(a.as_ptr(), b.as_ptr()) };
        let expected = cell_core::dot_product_64_scalar(&a, &b);
        assert!(
            (result - expected).abs() < 1e-4,
            "avx={}, scalar={}",
            result,
            expected
        );
    }

    #[test]
    fn attention_two_token_checksum() {
        let q = [0.25f32; 64];
        let k0 = [1.0f32; 64];
        let k1 = [0.5f32; 64];
        let v0 = [2.0f32; 64];
        let v1 = [4.0f32; 64];
        let mut out = [0.0f32; 64];

        let k_ptrs = [k0.as_ptr(), k1.as_ptr(), k0.as_ptr(), k0.as_ptr()];
        let v_ptrs = [v0.as_ptr(), v1.as_ptr(), v0.as_ptr(), v0.as_ptr()];

        unsafe {
            attention_head_64_avx(q.as_ptr(), k_ptrs, v_ptrs, 2, out.as_mut_ptr());
        }

        let checksum: f32 = out.iter().sum();
        let diff = (checksum - 162.4245).abs();
        assert!(
            diff < 1e-3,
            "Attention checksum drifted: checksum = {}, expected ~162.4245 (exact exp)",
            checksum
        );
    }

    #[test]
    fn rmsnorm_uniform_input_checksum() {
        let x = [2.0f32; 64];
        let gamma = [0.5f32; 64];
        let mut out = [0.0f32; 64];
        let sum = unsafe { rmsnorm_64_avx(x.as_ptr(), gamma.as_ptr(), out.as_mut_ptr(), 1e-5) };
        let expected = cell_core::rmsnorm_scalar(&x, &gamma, &mut [0.0f32; 64]);
        assert!(
            (sum - expected).abs() < 0.01,
            "avx_sum={}, scalar_sum={}",
            sum,
            expected
        );
    }

    #[test]
    fn multi_head_attention_512_checksum() {
        let q = [0.25f32; 512];
        let k0 = [1.0f32; 512];
        let k1 = [0.5f32; 512];
        let v0 = [2.0f32; 512];
        let v1 = [4.0f32; 512];
        let mut out = [0.0f32; 512];

        let k_ptrs = [k0.as_ptr(), k1.as_ptr(), core::ptr::null(), core::ptr::null()];
        let v_ptrs = [v0.as_ptr(), v1.as_ptr(), core::ptr::null(), core::ptr::null()];

        let total = unsafe {
            multi_head_attention_512_avx(q.as_ptr(), k_ptrs, v_ptrs, 2, out.as_mut_ptr())
        };

        let diff = (total - 1299.4).abs();
        assert!(
            diff < 0.1,
            "MHA-8 checksum drifted: total = {}, expected ~1299.4",
            total
        );
    }

    #[test]
    fn rmsnorm_512_uniform_ones_checksum() {
        let x = [1.0f32; 512];
        let gamma = [1.0f32; 512];
        let mut out = [0.0f32; 512];
        let sum = unsafe { rmsnorm_512_avx(x.as_ptr(), gamma.as_ptr(), out.as_mut_ptr(), 1e-5) };
        let diff = (sum - 512.0).abs();
        assert!(
            diff < 0.5,
            "rmsnorm_512 sum = {}, expected ~512.0",
            sum
        );
    }

    #[test]
    fn gemv_512x512_uniform_row_sum() {
        let x = [1.0f32; 512];
        let w = [0.0078125f32; 512 * 512];
        let mut out = [0.0f32; 512];
        unsafe { gemv_512x512_avx(x.as_ptr(), w.as_ptr(), out.as_mut_ptr()) };
        let sum: f32 = out.iter().sum();
        let diff = (sum - 2048.0).abs();
        assert!(
            diff < 0.1,
            "gemv_512x512 row sum = {}, expected ~2048.0",
            sum
        );
    }
}
