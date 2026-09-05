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
C2 AP2 Dynamic Dispatch Engine
	| match task.op { MatMul | VectorAdd | ReLU }
	| Chained execution: ReLU((A x B) + Bias)
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

### Flow Control & Watermark Monitoring

Untuk mencegah saturasi antrean antar-core, `SpscQueue` menyediakan inspeksi
kapasitas O(1) berbasis atomik tanpa lock:

- `len()` — jumlah elemen saat ini (head - tail dengan wrapping)
- `occupancy_pct()` — rasio keterisian dalam persentase (0..=100)
- `is_congested()` — `true` jika occupancy >= 75% (High Watermark)
- `is_drained()` — `true` jika occupancy <= 25% (Low Watermark)

Pembacaan `tail` oleh produsen menggunakan `Ordering::Acquire` untuk menjaga
konsistensi memori tanpa menimbulkan cache invalidation storm antar-core (64-byte
cache-line isolation).

Core 0 (BSP Ingress) menerapkan **backpressure adaptif**: sebelum setiap `push`,
core memeriksa `is_congested()`. Jika terdeteksi kongesti, core menahan
pengiriman (throttle) dan mencatat transisi watermark ke telemetri `QueueMetrics`
(`watermark_state: 1` = Congested). Saat konsumen mengosongkan antrean hingga
`is_drained()`, core melanjutkan dispatch (`watermark_state: 2` = Recovered).

```text
[CELL FLOW] Backpressure engaged on Q0->W1 (occupancy: 75%)
[CELL FLOW] Backpressure released (occupancy: 12%), resuming dispatch
[CORE 3 SUPERVISOR] Pipeline 4-Core tuntas: 9 sukses, 1 terisolasi. 0 dropped. Zero crash.
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

### TaskDescriptor & Dynamic Dispatch

BSP mengalokasikan 4 frame kontigu dari PMM (16 KiB = 4096 elemen F32) dan
menata layout: `[0..1024]` Matriks A (1.0), `[1024..2048]` Matriks B (2.0),
`[2048..3072]` Akumulator C (0.0), `[3072..4096]` Bias (-14.0). Buffer ini
dibungkus dalam `TaskDescriptor` (64-byte cache-aligned) yang berisi opcode
`TensorOp`, pointer tensor, dan offset elemen untuk input/output.

`cell-core` mendefinisikan empat opcode via enum `TensorOp`:
- `MatMul` (0x01): C = A x B (32x32 F32, AVX-256)
- `VectorAdd` (0x02): Out[i] = A[i] + B[i] (element-wise, AVX-256)
- `ReLU` (0x03): X[i] = max(0.0, X[i]) (in-place, AVX-256)
- `Attention` (0x04): softmax(Q·K_i^T / sqrt(64)) · V_i (AVX-256 dot product + Taylor softmax)

Antrean `TENSOR_HOP1` dan `TENSOR_HOP2` mengalirkan `TaskDescriptor` (bukan
`TensorChunk`). Core 2 (AP2 Compute) mengevaluasi opcode secara dinamis via
`match task.op`, sehingga satu compute engine dapat menjalankan berbagai jenis
operasi tanpa perubahan kode statis.

**Chained Layer Execution** mendemonstrasikan inferensi AI nyata:
```
Layer Output = ReLU((A x B) + Bias)
```
AP2 mengeksekusi berantai: MatMul (C = A x B, C[i,j] = 64.0) -> VectorAdd
(C += -14.0, C[i,j] = 50.0) -> ReLU (tidak ada perubahan karena 50.0 > 0).

```text
[BSP TENSOR] Prepared MatMul 32x32 layout (A=1.0, B=2.0, Bias=-14.0) at phys: 0x53000
[AP2 DISPATCH] Executed Op::MatMul (32x32 AVX)
[AP2 DISPATCH] Chained VectorAdd bias=-14.0
[AP2 DISPATCH] Chained ReLU activation
[AP2 COMPUTE] Pipeline calculation checksum verified: sum=51200.0 nonzero=1024 C[0,0]=50.0
[CELL PMM] contiguous frames count=4 returned to bitmap
[CORE 3 SUPERVISOR] Pipeline 4-Core tuntas: 9 sukses, 1 terisolasi. Zero crash.
```

### Hardware SIMD per-core

Modul `simd` mengaktifkan FPU/SSE/AVX secara independen pada BSP dan AP.
Deteksi CPUID memeriksa XSAVE dan AVX, kemudian konfigurasi ring-0 menghapus
CR0.EM/TS, mengaktifkan CR0.MP, CR4.OSFXSR/OSXMMEXCPT/OSXSAVE, dan XCR0
bits x87+SSE+AVX (`0x7`). Pustaka intrinsik AVX-256 menyediakan empat kernel:
- `gemm_32x32_avx`: perkalian matriks 32x32 F32 via `_mm256_set1_ps/mul_ps/add_ps`
- `vector_add_avx`: element-wise addition via `_mm256_add_ps`
- `relu_avx`: in-place ReLU activation via `_mm256_max_ps` dengan zero vector
- `verify_avx`/`verify_sse`: probe vektor untuk validasi hardware SIMD

Runner QEMU menggunakan `-cpu max -smp 4 -m 512M` agar capability AVX terlihat
di semua core.

### Binary telemetry

`cell-supervisor::telemetry` menyediakan fixed-layout, little-endian frames
dengan magic `[ce 11 54 4d]`, versi `1`, sequence number, opcode, payload
length, dan payload deterministik untuk `Heartbeat`, `PmmSnapshot`,
`QueueMetrics` (termasuk `watermark_state`: 0=Normal, 1=Congested, 2=Recovered), `TensorExecution` (termasuk GEMM 32x32 dengan
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

### Paged KV-Cache

`cell-core` mengimplementasikan engine KV-Cache berbasis paged memory untuk
inferensi LLM autoregresif. Setiap blok memori fisik PMM (16 KiB = 4 frame)
menyimpan Key dan Value untuk sejumlah token tetap tanpa heap allocation.

**Konstanta model:**
- `TOKENS_PER_BLOCK = 4` (untuk f32: 4 token × 4096 byte = 16 KiB)
- `NUM_HEADS = 8`, `HEAD_DIM = 64`
- `BYTES_PER_TOKEN = 2 × 8 × 64 × 4 = 4096 byte` (K + V)

**Struktur utama:**
- `KvBlockDescriptor` — deskriptor blok fisik 64-byte aligned dengan `block_id`,
  `phys_addr`, `virt_ptr`, `token_capacity`, dan `tokens_written`
- `SequenceContext<MAX_BLOCKS>` — tabel blok per-sequence dengan `append_block()`
  dan `reserve_next_token_slot()` yang mengembalikan pointer `(k_ptr, v_ptr)`

**Lifecycle:**
```
PMM alloc → KvBlockDescriptor::new() → SequenceContext::append_block()
→ reserve_next_token_slot() → tulis K/V via pointer → PMM free
```

Operasi slot bersifat deterministik O(1) tanpa fragmentasi. Ketika blok penuh,
panggil `append_block()` dengan blok PMM baru sebelum reserve berikutnya.

```sh
cargo test -p cell-core  # 14 unit tests termasuk KV-Cache + attention primitives
```

### Scaled Dot-Product Attention (AVX-256)

Kernel attention diimplementasikan dalam tiga tahap:

1. **Dot Product (AVX-256):** `dot_product_64_avx` menghitung Q · K_i untuk 64 elemen
   menggunakan 8 iterasi `vmovups` + `vmulps` + `vaddps`, diikuti reduksi horizontal
   via `vextractf128` + `vhaddps`.
2. **Softmax (Taylor):** `softmax_4_stable` mencari max skor untuk stabilitas numerik,
   menghitung exp(x) via Taylor series orde 4 ($1 + x + x^2/2 + x^3/6 + x^4/24$),
   dan menormalisasi bobot. Untuk $x < -8.0$, exp langsung diset 0.0.
3. **Weighted Value (AVX-256):** Bobot attention di-broadcast via `vbroadcastss`,
   dikalikan dengan V_i menggunakan `vmulps`, dan diakumulasikan ke buffer output
   menggunakan `vaddps`.

```sh
cargo test -p cell-core  # attention_two_token_checksum: verifikasi checksum ~162.909
```
