use core::arch::asm;
use core::arch::x86_64::*;

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

/// Computes C = A x B for 32x32 F32 matrices using AVX-256 intrinsics.
///
/// # Safety
/// Pointers `a`, `b`, and `c` must be valid for reading/writing 1024 F32 elements.
/// The CPU must have AVX enabled via XCR0/CR4.
#[target_feature(enable = "avx")]
pub unsafe fn gemm_32x32_avx(a: *const f32, b: *const f32, c: *mut f32) {
    const N: usize = 32;
    const K: usize = 32;
    const M: usize = 32;

    for i in 0..(M * N) {
        *c.add(i) = 0.0;
    }

    for i in 0..M {
        for k in 0..K {
            let a_val = *a.add(i * K + k);
            let a_vec = _mm256_set1_ps(a_val);

            for j in (0..N).step_by(8) {
                let b_idx = k * N + j;
                let c_idx = i * N + j;

                let b_vec = _mm256_loadu_ps(b.add(b_idx));
                let mut c_vec = _mm256_loadu_ps(c.add(c_idx));

                let prod = _mm256_mul_ps(a_vec, b_vec);
                c_vec = _mm256_add_ps(c_vec, prod);

                _mm256_storeu_ps(c.add(c_idx), c_vec);
            }
        }
    }
}

/// Out[i] = A[i] + B[i] for N F32 elements (N must be a multiple of 8).
///
/// # Safety
/// Pointers `a`, `b`, and `out` must be valid for `count` F32 elements.
#[target_feature(enable = "avx")]
pub unsafe fn vector_add_avx(a: *const f32, b: *const f32, out: *mut f32, count: usize) {
    for i in (0..count).step_by(8) {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let vres = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out.add(i), vres);
    }
}

/// Data[i] = max(0.0, Data[i]) for N F32 elements (N must be a multiple of 8).
///
/// # Safety
/// Pointer `data` must be valid for `count` F32 elements.
#[target_feature(enable = "avx")]
pub unsafe fn relu_avx(data: *mut f32, count: usize) {
    let zero = _mm256_setzero_ps();
    for i in (0..count).step_by(8) {
        let v = _mm256_loadu_ps(data.add(i));
        let activated = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(data.add(i), activated);
    }
}
