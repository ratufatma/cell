use core::arch::asm;
use cell_core::softmax_4_stable;

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

/// Enables the floating-point and SIMD state for the current logical CPU.
///
/// # Safety
/// Must run during early boot on the current CPU before executing floating-point
/// or SIMD instructions. The caller must not invoke it concurrently on the same
/// CPU.
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

/// Computes C = A x B for 32x32 F32 matrices using AVX-256 inline assembly.
///
/// # Safety
/// Pointers `a`, `b`, and `c` must be valid for reading/writing 1024 F32 elements.
/// The CPU must have AVX enabled via XCR0/CR4.
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
        assert!(
            (checksum - 162.909).abs() < 0.01,
            "checksum = {}, expected ~162.909 (Taylor exp)",
            checksum
        );
    }
}
        }
    }
}

/// Out[i] = A[i] + B[i] for N F32 elements (N must be a multiple of 8).
///
/// # Safety
/// Pointers `a`, `b`, and `out` must be valid for `count` F32 elements.
pub unsafe fn vector_add_avx(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
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

/// Data[i] = max(0.0, Data[i]) for N F32 elements (N must be a multiple of 8).
///
/// # Safety
/// Pointer `data` must be valid for `count` F32 elements.
pub unsafe fn relu_avx(data: *mut f32, count: usize) {
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

/// Dot product of two 64-element f32 slices via AVX-256 inline assembly.
///
/// # Safety
/// `a` and `b` must point to at least 64 valid f32 elements (256 bytes).
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

/// Scaled Dot-Product Attention for 1 head (dim 64) against N KV tokens.
///
/// Computes: out = softmax(Q · K_i / sqrt(64)) · V_i  for i in 0..N
///
/// # Safety
/// - `q` must point to 64 valid f32 elements.
/// - `k_ptrs[i]` and `v_ptrs[i]` must point to 64 valid f32 elements each.
/// - `out` must point to 64 writable f32 elements.
/// - `num_tokens` must be <= 4.
/// - All pointers must be aligned to at least 4 bytes.
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

    let weights = softmax_4_stable(&scores, n);

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
