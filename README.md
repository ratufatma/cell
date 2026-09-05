# CELL

CELL adalah fondasi kernel dataflow eksperimental berbasis Rust 2021. Modul berkomunikasi dengan nilai bertipe, ownership transfer, dan queue SPSC lock-free tanpa alokasi heap pada jalur panas.

## Struktur

- `cell-core`: crate `no_std` untuk `TraceContext`, typestate `RawPayload`/`ValidatedPayload`, `WorkResult`, dan `KernelModule`.
- `cell-queue`: crate `no_std` untuk ring buffer SPSC berbasis atomics, slot `MaybeUninit`, cache-line isolation, serta handle `Producer`/`Consumer`.
- `cell-supervisor`: pencatat lineage berkapasitas tetap dan reset hook node.
- `cell-baremetal`: kernel `no_std`/`no_main` x86_64 dengan boot protocol Limine, UART 16550 COM1, bitmap PMM 4 KiB/frame, pipeline SMP BSP-to-AP, dan completion loop AP-to-BSP.
- `tests/simulation_harness`: simulasi ingress, validator, worker, dan fault injection.

## Menjalankan

```sh
cargo test --all
```

Test harness mengalirkan sepuluh paket dari Node A ke Node B dan Node C. Satu paket rusak ditolak Node C sebagai `WorkResult::Failed`, dicatat supervisor berdasarkan `trace_id`, lalu node di-reset dan sembilan paket lain tetap selesai.

Stress test multi-thread menjalankan satu juta pesan melalui producer dan consumer SPSC secara paralel:

```sh
cargo test -p simulation-harness --test concurrency_stress -- --exact concurrency_stress_test
```

## Boot QEMU

Pasang target Rust sekali, lalu jalankan runner. Script akan mengunduh dan membangun tool Limine di `.tools/limine`, membuat ISO, dan meneruskan UART ke terminal:

```sh
rustup target add x86_64-unknown-none
./scripts/run_qemu.sh
```

Output awal yang diharapkan:

```text
[CELL KERNEL] Dataflow pipeline active
[CELL KERNEL] trace=1 origin=1 len=9
```

Dengan QEMU dua vCPU, BSP mengirim delapan `RawPayload` melalui SPSC ring buffer
ke AP pertama. AP memvalidasi setiap payload dan mencetak trace secara paralel:

```text
[CELL SMP] AP1 online
[CELL SMP] BSP -> AP pipeline online
[CELL SMP] AP1 validated trace=0 len=9
...
[CELL SMP] completed 8 packets
```

Runner menggunakan `qemu-system-x86_64 -smp 2`; lock hanya melindungi output UART
agar karakter BSP/AP tidak bercampur. Jalur data antar-core tetap menggunakan
queue SPSC tanpa global lock.

### Feedback loop dan fault injection

BSP dan AP berkomunikasi dua arah melalui dua queue SPSC statis:

- `INTER_CORE_QUEUE`: BSP producer -> AP consumer untuk `RawPayload`.
- `COMPLETION_QUEUE`: AP producer -> BSP consumer untuk `WorkResult`.

Batch bare-metal berisi sepuluh trace. Trace 4 dikirim sebagai payload kosong,
ditolak AP sebagai `WorkResult::Failed`, dan dicatat `Supervisor` di BSP tanpa
menghentikan pemrosesan trace lain:

```text
[CELL SMP] Bidirectional feedback loop active
[SUPERVISOR] Trace 1 OK
[SUPERVISOR WARN] Fault captured on Node 1! Trace 4 failed: Corrupted payload in AP1
[SUPERVISOR] Trace 10 OK
[CELL KERNEL] Batch completed: 9 succeeded, 1 isolated failure. Zero crash.
```

### Physical Memory Manager

Kernel meminta `MemoryMapRequest` dari Limine dan membangun bitmap allocator
untuk frame 4 KiB pada region `USABLE` saja. Semua region lain tetap dianggap
teralokasi atau reserved. Implementasi PoC membatasi bitmap pada physical
address di bawah 4 GiB, tidak memakai heap, dan menyediakan operasi deterministic
allocate/free yang hanya dipanggil BSP pada tahap ini. Saat boot, kernel mencetak
statistik frame usable/free dan melakukan probe allocate lalu free untuk
memverifikasi bahwa frame kembali ke bitmap.

Validasi compile kernel tanpa boot QEMU:

```sh
cargo check --target x86_64-unknown-none -p cell-baremetal
```
