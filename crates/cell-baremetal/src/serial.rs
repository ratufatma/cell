use core::arch::asm;
use core::fmt::{self, Write};
use core::sync::atomic::{AtomicBool, Ordering};

use cell_supervisor::telemetry::{TelemetryEncoder, TelemetryEvent};

const COM1: u16 = 0x3f8;

pub static SERIAL_LOCK: AtomicBool = AtomicBool::new(false);

pub struct Serial;

impl Serial {
    pub const fn new() -> Self {
        Self
    }

    unsafe fn outb(port: u16, value: u8) {
        // SAFETY: COM1 is the conventional x86 primary UART port and this is the
        // only writer during single-core early boot.
        asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack, preserves_flags));
    }

    unsafe fn inb(port: u16) -> u8 {
        let value: u8;
        // SAFETY: Reading the UART line-status register is valid for COM1.
        asm!("in al, dx", out("al") value, in("dx") port, options(nomem, nostack, preserves_flags));
        value
    }

    unsafe fn init(&self) {
        Self::outb(COM1 + 1, 0x00);
        Self::outb(COM1 + 3, 0x80);
        Self::outb(COM1, 0x03);
        Self::outb(COM1 + 1, 0x00);
        Self::outb(COM1 + 3, 0x03);
        Self::outb(COM1 + 2, 0xc7);
        Self::outb(COM1 + 4, 0x0b);
    }

    unsafe fn write_byte(&self, byte: u8) {
        while Self::inb(COM1 + 5) & 0x20 == 0 {
            core::hint::spin_loop();
        }
        Self::outb(COM1, byte);
    }
}

impl Write for Serial {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        for byte in text.bytes() {
            unsafe {
                if byte == b'\n' {
                    self.write_byte(b'\r');
                }
                self.write_byte(byte);
            }
        }
        Ok(())
    }
}

pub static mut SERIAL: Serial = Serial::new();

static mut TELEMETRY: TelemetryEncoder = TelemetryEncoder::new();

pub fn init() {
    // SAFETY: Single-threaded during early boot; the constructor's invariants
    // are only established here, before any concurrent serial users exist.
    unsafe { (&*core::ptr::addr_of!(SERIAL)).init() }
}

macro_rules! serial_print {
    ($($arg:tt)*) => {{
        crate::serial::_print(format_args!($($arg)*));
    }};
}

macro_rules! serial_println {
    () => {serial_print!("\n");};
    ($($arg:tt)*) => {serial_print!("{}\n", format_args!($($arg)*));};
}

pub fn _print(arguments: fmt::Arguments<'_>) {
    while SERIAL_LOCK
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }

    // SAFETY: Writers serialize on SERIAL_LOCK above, so the mut access is
    // guaranteed unique for the duration of the write.
    unsafe {
        let serial = &mut *core::ptr::addr_of_mut!(SERIAL);
        let _ = serial.write_fmt(arguments);
    }
    SERIAL_LOCK.store(false, Ordering::Release);
}

pub fn telemetry(event: TelemetryEvent) {
    while SERIAL_LOCK
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
    // SAFETY: Serialized by SERIAL_LOCK; TELEMETRY is only mutated here.
    let frame = unsafe { (&mut *core::ptr::addr_of_mut!(TELEMETRY)).encode(event) };
    // SAFETY: Serialized by SERIAL_LOCK; SERIAL is only mutated here.
    unsafe {
        let serial = &mut *core::ptr::addr_of_mut!(SERIAL);
        for byte in b"[CELL TM] " {
            serial.write_byte(*byte);
        }
        for byte in frame.as_bytes() {
            serial.write_byte(hex_digit(byte >> 4));
            serial.write_byte(hex_digit(byte & 0x0f));
        }
        serial.write_byte(b'\r');
        serial.write_byte(b'\n');
    }
    SERIAL_LOCK.store(false, Ordering::Release);
}

fn hex_digit(value: u8) -> u8 {
    match value {
        0..=9 => b'0' + value,
        _ => b'a' + (value - 10),
    }
}