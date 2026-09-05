use core::arch::asm;

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
    // CPUID.OSXSAVE reports whether the OS has already enabled CR4.OSXSAVE;
    // before this function runs it is expected to be clear. XSAVE + AVX are
    // the hardware prerequisites, after which this function enables OSXSAVE.
    let avx_enabled = has_xsave && has_avx;

    let mut cr0: u64;
    // SAFETY: Reading CR0 is valid in ring 0 during kernel initialization.
    unsafe {
        asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
    }
    cr0 &= !(1 << 2); // EM: do not emulate the FPU.
    cr0 |= 1 << 1; // MP: monitor the coprocessor.
    cr0 &= !(1 << 3); // TS: do not report a task-switch trap.
                      // SAFETY: The updated CR0 enables native x87/SSE execution on this CPU.
    unsafe {
        asm!("mov cr0, {}", in(reg) cr0, options(nomem, nostack, preserves_flags));
    }

    let mut cr4: u64;
    // SAFETY: Reading CR4 is valid in ring 0 during kernel initialization.
    unsafe {
        asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
    }
    cr4 |= (1 << 9) | (1 << 10); // OSFXSR and OSXMMEXCPT.
    if avx_enabled {
        cr4 |= 1 << 18; // OSXSAVE.
    }
    // SAFETY: The CR4 feature bits are enabled only after CPUID capability checks.
    unsafe {
        asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack, preserves_flags));
    }

    if avx_enabled {
        // SAFETY: OSXSAVE is enabled above and CPUID reports XSAVE and AVX.
        unsafe {
            asm!(
                "xsetbv",
                in("ecx") 0_u32,
                in("eax") 0x7_u32,
                in("edx") 0_u32,
                options(nomem, nostack, preserves_flags)
            );
        }
        SimdLevel::Avx
    } else {
        SimdLevel::Sse
    }
}

/// Adds the first eight F32 values to themselves using AVX `vaddps`.
///
/// # Safety
/// `values` must contain at least eight initialized, readable `f32` values and
/// the current CPU must have been initialized with `SimdLevel::Avx`.
pub unsafe fn verify_avx(values: &[f32]) -> bool {
    if values.len() < 8 {
        return false;
    }
    let mut output = [0.0_f32; 8];
    // SAFETY: The caller guarantees eight readable f32 values; output is a
    // writable eight-element f32 array and both pointers are properly aligned.
    unsafe {
        asm!(
            "vmovups ymm0, [{input}]",
            "vaddps ymm0, ymm0, ymm0",
            "vmovups [{output}], ymm0",
            "vzeroupper",
            input = in(reg) values.as_ptr(),
            output = in(reg) output.as_mut_ptr(),
            options(nostack, preserves_flags)
        );
    }
    output.iter().all(|value| *value == 2.0)
}

/// Adds the first four F32 values to themselves using SSE `addps`.
///
/// # Safety
/// `values` must contain at least four initialized, readable `f32` values and
/// the current CPU must have been initialized with SSE support.
pub unsafe fn verify_sse(values: &[f32]) -> bool {
    if values.len() < 4 {
        return false;
    }
    let mut output = [0.0_f32; 4];
    // SAFETY: The caller guarantees four readable f32 values and output is a
    // writable four-element f32 array.
    unsafe {
        asm!(
            "movups xmm0, [{input}]",
            "addps xmm0, xmm0",
            "movups [{output}], xmm0",
            input = in(reg) values.as_ptr(),
            output = in(reg) output.as_mut_ptr(),
            options(nostack, preserves_flags)
        );
    }
    output.iter().all(|value| *value == 2.0)
}
