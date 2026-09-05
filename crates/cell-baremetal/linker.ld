ENTRY(_start)

KERNEL_VMA = 0xffffffff80000000;

SECTIONS
{
    . = KERNEL_VMA;
    .text ALIGN(4K) : AT(ADDR(.text) - KERNEL_VMA) {
        KEEP(*(.text._start))
        *(.text*)
    }
    .rodata ALIGN(4K) : AT(ADDR(.rodata) - KERNEL_VMA) { *(.rodata*) }
    .got ALIGN(4K) : AT(ADDR(.got) - KERNEL_VMA) { *(.got*) }
    .data ALIGN(4K) : AT(ADDR(.data) - KERNEL_VMA) { *(.data*) }
    .bss ALIGN(4K) : AT(ADDR(.bss) - KERNEL_VMA) { *(COMMON) *(.bss*) }
    .requests ALIGN(8) : AT(ADDR(.requests) - KERNEL_VMA) {
        KEEP(*(.requests_start_marker))
        KEEP(*(.limine_requests))
        KEEP(*(.requests_end_marker))
    }
}
