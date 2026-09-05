# CELL

CELL adalah fondasi kernel dataflow eksperimental berbasis Rust 2021. Modul berkomunikasi dengan nilai bertipe, ownership transfer, dan queue SPSC lock-free tanpa alokasi heap pada jalur panas.

## Struktur

- `cell-core`: crate `no_std` untuk `TraceContext`, typestate `RawPayload`/`ValidatedPayload`, `WorkResult`, `KernelModule`, serta primitive tensor zero-copy `TensorChunk`, `DType`, dan `TensorShape`.
- `cell-queue`: crate `no_std` untuk ring buffer SPSC berbasis atomics, slot `MaybeUninit`, cache-line isolation, serta handle `Producer`/`Consumer`.
- `cell-supervisor`: pencatat lineage berkapasitas tetap dan reset hook node.
- `cell-baremetal`: kernel `no_std`/`no_main` x86_64 dengan boot protocol Limine, UART 16550 COM1, bitmap PMM 4 KiB/frame, pipeline DAG SMP 4-core, dan telemetry CELLTM.
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

Dengan QEMU empat vCPU, BSP mengirim batch melalui tiga hop SPSC ke AP1, AP2,
dan AP3. Masing-masing core menjadi operator dataflow terpisah:

```text
[CELL SMP] Core 0 (BSP Ingress) online
[CELL SMP] Core 1 (AP1 Arbiter) online
[CELL SMP] Core 2 (AP2 Compute) online
[CELL SMP] Core 3 (AP3 Supervisor) online
[CELL PIPELINE] 4-Core topology active: C0 -> C1 -> C2 -> C3
```

Runner menggunakan `qemu-system-x86_64 -cpu max -smp 4 -m 512M`; lock hanya
melindungi output UART agar karakter antar-core tidak bercampur. Jalur data
antar-core tetap menggunakan queue SPSC tanpa global lock.

### Topologi DAG 4-core

```text
C0 BSP Ingress/Loader
	| QUEUE_INGRESS_TO_W1 / TENSOR_HOP1
	v
C1 AP1 Arbiter/Preprocessor
	| QUEUE_W1_TO_W2 / TENSOR_HOP2
	v
C2 AP2 AVX Compute
	| QUEUE_W2_TO_SUPERVISOR + tensor completion token
	| (GEMM 32x32 AVX-256: C = A x B, verifikasi C[0,0]=64, sum=65536)
	v
C3 AP3 Supervisor/Telemetry/Egress
```

AP1 mengubah payload menjadi `PipelineMessage::Valid` atau
`BypassFault`. AP2 menjalankan AVX reduction dan meneruskan hasil. AP3
mencatat lineage, memancarkan CELLTM, dan mengembalikan frame tensor ke PMM.
Trace 4 diisolasi di Core 3 tanpa menghentikan trace lainnya.

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
allocate/free dengan spin lock PMM agar dapat dipanggil pada handoff BSP/AP.
API `allocate_contiguous_frames(count)` mencari run bit bebas yang melintasi
batas word bitmap dengan aman, sedangkan `free_contiguous_frames(base, count)`
mengembalikan seluruh run secara atomik terhadap lock PMM. Saat boot, kernel
mencetak statistik frame usable/free dan largest free run.

Validasi compile kernel tanpa boot QEMU:

```sh
cargo check --target x86_64-unknown-none -p cell-baremetal
```

### TensorChunk zero-copy antar-core

BSP meminta HHDM offset Limine, mengalokasikan empat frame fisik kontigu dari
PMM, lalu menata layout tensor 2D: matriks A (32x32, 1024 elemen F32 = 4 KiB)
diisi `1.0`, matriks B (32x32) diisi `2.0`, matriks C (32x32) diisi `0.0`,
dan 1024 elemen sisa sebagai padding. Descriptor `TensorChunk<Ready>` dikirim
melalui `TENSOR_HOP1` ke AP1 lalu `TENSOR_HOP2` ke AP2 tanpa menyalin buffer.

AP2 menjalankan kernel GEMM 32x32 AVX-256 (`C = A x B`), memverifikasi bahwa
setiap elemen C[i,j] = 64.0 dan total reduksi = 65536.0, lalu meneruskan
checksum ke AP3. AP3 memancarkan paket `TensorExecution` ke telemetri CELLTM
dan mengembalikan empat frame ke PMM:

```text
[BSP TENSOR] Prepared MatMul 32x32 layout (A=1.0, B=2.0) at phys: 0x53000
[AP2 COMPUTE] GEMM 32x32 AVX-256 complete: C[0,0]=64 sum=65536 expected=65536
[CELL PMM] contiguous frames count=4 returned to bitmap
[CORE 3 SUPERVISOR] Pipeline 4-Core tuntas: 9 sukses, 1 terisolasi. Zero crash.
```

### Hardware SIMD per-core

Modul `simd` mengaktifkan FPU/SSE/AVX secara independen pada BSP dan AP.
Deteksi CPUID memeriksa XSAVE dan AVX, kemudian konfigurasi ring-0 menghapus
CR0.EM/TS, mengaktifkan CR0.MP, CR4.OSFXSR/OSXMMEXCPT/OSXSAVE, dan XCR0
bits x87+SSE+AVX (`0x7`). Kernel GEMM 32x32 AVX-256 menggunakan intrinsik
`_mm256_set1_ps`, `_mm256_loadu_ps`, `_mm256_mul_ps`, `_mm256_add_ps`, dan
`_mm256_storeu_ps` untuk menghitung perkalian matriks pada 8 float per register
`ymm` secara paralel. Runner QEMU menggunakan `-cpu max -smp 4 -m 512M` agar
capability AVX terlihat di semua core.

```text
[CELL SIMD] BSP initialized hardware vector engine: AVX (256-bit)
[CELL SIMD] AP2 initialized hardware vector engine: AVX (256-bit)
[AP2 COMPUTE] GEMM 32x32 AVX-256 complete: C[0,0]=64 sum=65536 expected=65536
```

### Binary telemetry

`cell-supervisor::telemetry` menyediakan fixed-layout, little-endian frames
dengan magic `[ce 11 54 4d]`, versi `1`, sequence number, opcode, payload
length, dan payload deterministik untuk `Heartbeat`, `PmmSnapshot`,
`QueueMetrics`, `TensorExecution` (termasuk GEMM 32x32 dengan
`elements=1024`, `simd_level=2` untuk AVX-256), serta `FaultIncident`. Encoder memakai buffer
tetap tanpa `alloc`; `decode()` memvalidasi magic, versi, opcode, dan panjang
sebelum mengembalikan event terstruktur.

Kernel bare-metal mengirim frame sebagai hex di baris berawalan `[CELL TM]` agar
UART tetap dapat diamati manusia sekaligus diproses parser machine-readable.
Format hex hanya transport display; byte wire frame adalah isi setelah prefix
dan dapat didekode deterministik dengan `cell_supervisor::telemetry::decode`.

Parser Python real-time tersedia di `scripts/parse_telemetry.py`:

```sh
chmod +x scripts/parse_telemetry.py
./scripts/run_qemu.sh | python3 scripts/parse_telemetry.py
./scripts/run_qemu.sh | python3 scripts/parse_telemetry.py --pretty --verbose
./scripts/run_qemu.sh | python3 scripts/parse_telemetry.py --output telemetry_stream.jsonl
```

Output default adalah NDJSON satu event per baris. Log kernel non-telemetry
hanya diteruskan ke stderr saat `--verbose` digunakan.
