# CELL: AI-Native Operating Substrate Architecture

CELL adalah sistem operasi *bare-metal* berbasis Rust (`#![no_std]`) yang dirancang khusus untuk mengeksekusi beban kerja inferensi AI dan pemrosesan aliran data (*dataflow*) secara deterministik. Sistem ini mengeliminasi abstraksi monolitik POSIX konvensional, menggantikan mekanisme *preemptive time-sharing* dengan pipeline *Directed Acyclic Graph (DAG)* multi-core, menyediakan komunikasi *zero-copy lock-free*, dan mengekspos telemetri biner deterministik (`CELLTM`) untuk supervisi agen AI otonom.

---

## 1. Spesifikasi Perangkat Keras & Siklus Bootstrap

### A. Target Kompilasi & Lingkungan Mesin
* **Target Triplet:** `x86_64-unknown-none`
* **Bootloader:** Limine Boot Protocol (Higher-Half Direct Map / HHDM & Limine SMP Request)
* **Runner Virtualisasi:** QEMU emulator dengan deteksi otomatis KVM (`-enable-kvm -cpu host` jika tersedia) atau fallback TCG (`-cpu max`). Set `CELL_FORCE_TCG=1` untuk memaksa mode TCG.
* **Saluran I/O:** Serial UART 16550 COM1 (`0x3F8`) beroperasi pada baudrate 115200 8N1 dengan serial spinlock minimal.

### B. Inisialisasi Register Kontrol CPU (FPU, SSE, AVX)
Untuk mencegah exception `#UD` (Invalid Opcode) dan `#NM` (Device Not Available) saat instruksi floating-point atau vektor dieksekusi di Ring 0, BSP dan seluruh AP mengonfigurasi register kontrol hardware sebelum memasuki loop tugas:

1. **Control Register 0 (`CR0`):**
   * Hapus bit **EM** (`bit 2`): Menonaktifkan emulasi software x87 FPU.
   * Setel bit **MP** (`bit 1`): Mengaktifkan pemantauan *coprocessor* untuk instruksi `WAIT`/`FWAIT`.
   * Hapus bit **TS** (`bit 3`): Mencegah *Task Switched trap* yang tidak diperlukan pada model eksekusi *run-to-completion*.
2. **Control Register 4 (`CR4`):**
   * Setel bit **OSFXSR** (`bit 9`): Mengaktifkan dukungan instruksi SSE/SSE2 dan struktur simpan-pulih `FXSAVE`/`FXRSTOR`.
   * Setel bit **OSXMMEXCPT** (`bit 10`): Mengarahkan *unmasked SIMD floating-point exceptions* ke handler `#XM`.
   * Setel bit **OSXSAVE** (`bit 18`): Mengaktifkan instruksi manajemen *extended processor state* (`xgetbv`/`xsetbv`).
3. **Extended Control Register 0 (`XCR0`):**
   * Ditulis melalui instruksi assembly `xsetbv` dengan nilai `0x7`:
     * `Bit 0`: State x87 FPU.
     * `Bit 1`: State SSE (register `xmm0`..`xmm15`).
     * `Bit 2`: State AVX (register `ymm0`..`ymm15` *upper 128-bit*).

---

## 2. Physical Memory Manager (PMM) & Zero-Copy Fabric

### A. Bitmap Allocator
PMM memetakan seluruh memori fisik yang dapat digunakan (*usable RAM*) ke dalam bitmap statis. Setiap bit merepresentasikan satu frame memori fisik berukuran 4 KiB ($4.096$ byte).

### B. Alokasi Frame Kontigu Multi-Frame
Inferensi tensor menuntut alokasi penyangga data berurutan tanpa fragmentasi. Fungsi `allocate_contiguous_frames(count)` mencari rentang bit bernilai `0` berurutan lintas batas kata (*word boundary*) pada array bitmap:
* **Blok Standar Pipeline:** 4 frame fisik kontigu = 16 KiB ($16.384$ byte = $4.096$ elemen `f32`).
* **Translasi Alamat HHDM:** Alamat fisik instan dipetakan ke alamat virtual kernel higher-half tanpa modifikasi *page table* runtime:
  $$\text{virt\_ptr} = \text{phys\_addr} + \text{hhdm\_offset}$$
* **Dealokasi Deterministik:** `free_contiguous_frames(base_phys, count)` mereset bit secara atomik dan mengembalikan kepemilikan blok ke bitmap pool.

### C. Siklus Hidup Memori Berbasis Typestate
Untuk menjamin integritas memori pada waktu kompilasi (*compile-time thread safety*), kepemilikan buffer tensor dimodelkan dengan *Typestate Pattern*:
$$\text{Unallocated} \xrightarrow{\text{PMM Alloc}} \text{TensorChunk}\langle\text{MutableState}\rangle \xrightarrow{\text{.freeze()}} \text{TensorChunk}\langle\text{Ready}\rangle \xrightarrow{\text{.deconstruct()}} \text{PMM Free}$$
* Hanya `TensorChunk<Ready>` yang diizinkan masuk ke saluran transfer inter-core. Buffer berstatus *Ready* bersifat *read-only* (atau *in-place compute boundary*) sehingga bebas dari *data race*.

---

## 3. Topologi DAG 4-Core SMP & Inter-Core Fabric

CELL membagi komputasi menjadi topologi *Directed Acyclic Graph* (DAG) 4-core di mana setiap inti CPU mengeksekusi peran terisolasi:

```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Core 0 (BSP)   │       │  Core 1 (AP1)   │       │  Core 2 (AP2)   │       │  Core 3 (AP3)   │
│ Ingress & Loader│       │ Arbiter & Filter│       │ Compute Engine  │       │Supervisor/Egress│
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │                         │
         │───[QUEUE_INGRESS_W1]───>│                         │                         │
         │───[TENSOR_HOP1]────────>│                         │                         │
         │                         │───[QUEUE_W1_TO_W2]─────>│                         │
         │                         │───[TENSOR_HOP2]────────>│                         │
         │                         │                         │───[QUEUE_W2_TO_SUP]────>│
         │                         │                         │                         │
         │                         │                         │                         │── PMM Reclaim
         │                         │                         │                         │── CELLTM Stream

```

### A. Pembagian Peran per-Core

1. **Core 0 (BSP - Ingress & Loader):**
* Menginisialisasi perangkat keras dasar (PMM, UART, SIMD AVX).
* Membangunkan AP1, AP2, dan AP3 melalui Limine SMP Request.
* Mengalokasikan 4 frame fisik kontigu PMM, menginisialisasi bobot/tensor, dan menyusun `TaskDescriptor`.
* Memantau *watermark* antrean dan memoderasi laju transmisi (*throttling*).


2. **Core 1 (AP1 - Arbiter & Preprocessor):**
* Mengonsumsi paket dari `QUEUE_INGRESS_TO_W1`.
* Menerapkan validasi tipestate: `RawPayload` $\to$ `ValidatedPayload`.
* *Fault Containment:* Jika anomali terdeteksi (Trace 4 corrupt), paket dibungkus menjadi `PipelineMessage::BypassFault` untuk mencegah kegagalan pipeline.
* Meneruskan `TaskDescriptor` dari `TENSOR_HOP1` ke `TENSOR_HOP2`.


3. **Core 2 (AP2 - Compute Engine):**
* Dynamic Task Dispatcher: Mengevaluasi opcode `task.op` (`MatMul`, `VectorAdd`, `ReLU`).
* Menjalankan eksekusi vektor berantai (*chained execution*) menggunakan instruksi inline AVX-256 murni.
* Menghitung nilai reduksi checksum dan meneruskan status kerja ke `QUEUE_W2_TO_SUPERVISOR`.


4. **Core 3 (AP3 - Supervisor & Egress):**
* Mengonsumsi laporan kerja dari `QUEUE_W2_TO_SUPERVISOR`.
* Mencatat silsilah kegagalan (*fault lineage*) Trace 4 secara terisolasi tanpa *kernel panic*.
* Mengambil alih kepemilikan blok frame fisik tensor dan mengembalikannya ke PMM via `pmm::free_contiguous_frames`.
* Memancarkan paket telemetri biner `CELLTM` ke UART serial COM1.



### B. Primitif Komunikasi: SPSC Lock-Free Ring Buffer

* **Bebas False Sharing:** Struktur data `SpscQueue<T, N>` menggunakan atribut `#[repr(align(64))]` untuk memisahkan indeks atomik `head` dan `tail` ke dalam jalur *cache line* fisik yang independen.
* **Semantik Memori:**
* Konsumsi/Penulisan `head` dan `tail` lokal menggunakan `Ordering::Relaxed`.
* Publikasi data slot menggunakan `Ordering::Release`.
* Inspeksi indeks oleh inti lawan menggunakan `Ordering::Acquire`.


* **Zero-Copy Overhead:** Transfer antrean hanya memindahkan deskriptor tugas berukuran 64 byte. Data fisik tensor sebesar 16 KiB tidak pernah disalin (*zero-copy*).

---

## 4. Mekanisme Flow Control & Backpressure

Untuk mencegah Core 0 (Ingress) membanjiri antrean ketika Core 2 sedang memproses komputasi matriks yang intensif, `cell-queue` mengimplementasikan pemantauan kapasitas berbasis *watermark*:

### A. Algoritma Okupansi $O(1)$

Keterisian antrean dihitung secara instan melalui operasi selisih modulo atomik:


$$\text{Occupancy} = \text{head.load(Relaxed)} - \text{tail.load(Acquire)}$$

$$\text{Occupancy Pct} = \left( \frac{\text{Occupancy} \times 100}{N} \right)$$

### B. Ambang Batas Watermark & Mitigasi

* **High Watermark (HWM $\ge 75\%$):** Menandakan antrean mulai jenuh.
* Ketika `queue.is_congested()` bernilai `true`, Core 0 mengaktifkan *backpressure*.
* Core 0 menghentikan `push` baru dan mengeksekusi loop jeda adaptif via `core::hint::spin_loop()`.
* Status ditransmisikan ke telemetri (`watermark_state = 1`).


* **Low Watermark (LWM $\le 25\%$):** Menandakan antrean telah kembali longgar.
* Ketika `queue.is_drained()` bernilai `true`, *backpressure* dilepas (`watermark_state = 2`).
* Core 0 melanjutkan pemompaan sisa batch.


* **Invarian Sistem:** Seluruh paket data terkirim tanpa ada yang terbuang (**0 dropped packets**).

---

## 5. Dynamic Task Dispatcher & Komputasi Vektor AVX-256

Core 2 mengoperasikan *Compute Engine* fleksibel berbasis *Actor Token* yang mendukung inferensi lapisan jaringan saraf tiruan berantai:

$$\mathbf{y} = \text{ReLU}(\mathbf{W}\mathbf{x} + \mathbf{b})$$

### A. Struktur `TaskDescriptor` (64-byte Cache-Aligned)

```rust
#[repr(C, align(64))]
pub struct TaskDescriptor {
    pub context: TraceContext,
    pub op: TensorOp,
    pub tensor: TensorChunk<Ready>,
    pub in_offset_a: u32,
    pub in_offset_b: u32,
    pub out_offset: u32,
    pub element_count: u32,
    pub _reserved: [u8; 15],
}

```

### B. Katalog Opcode & Eksekusi Berantai

1. **`TensorOp::MatMul` (0x01):** Perkalian matriks $32 \times 32$ ($A \times B \to C$) menggunakan layout baris-utama (*row-major*). Setiap elemen $C_{i,j} = \sum_{k=0}^{31} (1.0 \times 2.0) = 64.0$.
2. **`TensorOp::VectorAdd` (0x02):** Penjumlahan elemen vektor (injeksi bias). Bias sebesar $-14.0$ ditambahkan ke tiap elemen matriks $C$, menghasilkan nilai $50.0$.
3. **`TensorOp::ReLU` (0x03):** Fungsi aktivasi non-linear *in-place*. Memotong nilai negatif $\max(0.0, x)$. Elemen $50.0$ tetap bernilai $50.0$.
4. **`TensorOp::Attention` (0x04):** Scaled Dot-Product Attention untuk satu attention head ($d_k = 64$) terhadap $N \le 4$ token KV-cache. Formula: $\text{softmax}\left(\frac{Q \cdot K_i^T}{\sqrt{64}}\right) \cdot V_i$.
4. **Verifikasi Reduksi Akhir:**

$$\sum_{i=0}^{31} \sum_{j=0}^{31} C_{i,j} = 1.024 \times 50.0 = \mathbf{51.200,0}$$



---

## 6. Mitigasi Bug Kritis Kompiler & Assembly (Hard-Won Engineering Lessons)

Selama proses implementasi komputasi vektor di bare-metal x86_64, empat anomali teknis diselesaikan:

### Mitigasi 1: Konflik Operand pada Instruksi `vbroadcastss` & `vmovss`

* **Masalah:** Percobaan menggunakan instruksi broadcast langsung dari dereferensi memori dengan pola register fleksibel menyebabkan assembler LLVM menolak instruksi atau menghasilkan pengalamatan memori yang tidak valid.
* **Solusi:** Memisahkan pembacaan skalar eksplisit dan broadcast ke register vektor:
1. Baca nilai float skalar ke register XMM: `vmovss {xmm_tmp}, dword ptr [{ptr}]`
2. Broadcast dari jalur bawah XMM ke seluruh 8 lajur YMM: `vbroadcastss {ymm_dst}, {xmm_tmp}`



### Mitigasi 2: Resolusi Offset Elemen vs Offset Byte pada Buffer Tensor

* **Masalah:** Inisialisasi slice memori di BSP sempat menggunakan pengindeksan elemen float langsung pada slice byte (`slice[1024..2048]`), yang menyebabkan matriks $B$ dan $C$ saling tumpang tindih (*overlap*) karena offset byte seharusnya $4 \times$ offset elemen ($1.024 \times 4 = 4.096$ byte per matriks).
* **Solusi:** Menyamakan kontrak tata letak memori:
* Matriks $A$ ($1.024$ elemen): byte `0..4096`
* Matriks $B$ ($1.024$ elemen): byte `4096..8192`
* Matriks $C$ / Output ($1.024$ elemen): byte `8192..12288`
* Vektor Bias ($1.024$ elemen): byte `12288..16384`



### Mitigasi 3: LLVM Codegen Crash pada Atribut `#[target_feature(enable = "avx")]`

* **Masalah:** Pada target `x86_64-unknown-none`, penggunaan intrinsik bawaan Rust/LLVM (`_mm256_*`) yang dihiasi atribut `#[target_feature(enable = "avx")]` memicu *compiler backend crash* (`rustc-LLVM ERROR: LLVM ERROR: Do not know how to split the result of this operator!`) karena target bare-metal tidak memiliki runtime std untuk fitur deteksi CPUID dinamis.
* **Solusi:** Menghapus ketergantungan pada intrinsik compiler LLVM. Seluruh operasi GEMM, VectorAdd, dan ReLU ditulis menggunakan **inline assembly eksplisit murni** (`core::arch::asm!`) dengan direktif register eksplisit (`inout("ymm0")`, `vaddps`, `vmulps`, `vmaxps`).

### Mitigasi 4: Kalibrasi Kapasitas Antrean untuk Deteksi Backpressure

* **Masalah:** Kapasitas default antrean $N=32$ terlalu besar untuk mendeteksi *High Watermark* (75%) saat mengalirkan paket uji coba berjumlah 10 paket ($10 / 32 = 31\% < 75\%$), sehingga logika mitigasi *backpressure* tidak pernah terpicu.
* **Solusi:** Menyesuaikan kapasitas antrean SPSC hop pertama menjadi $N=8$. Dengan batch berukuran 10, pengiriman 6 paket langsung menyentuh batas $75\%$ ($6/8 = 75\%$), memvalidasi mekanisme penahanan produsen secara deterministik.

---

## 7. Protokol Telemetri Biner `CELLTM`

Untuk mengeliminasi parsing teks bebas (*regex string parsing*) pada agen AI, CELL memancarkan bingkai paket telemetri biner deterministik melalui UART COM1:

```text
┌──────────────┬─────────────┬────────────┬────────────────┬─────────────────┬──────────────────┐
│ Magic (4B)   │ Version(1B) │ Opcode(1B) │  Sequence(4B)  │ PayloadLen(2B)  │  Payload (N B)   │
│  0xCE11544D  │    0x01     │ 0x01..0x05 │ Little-Endian  │  Little-Endian  │  Fixed Struct    │
└──────────────┴─────────────┴────────────┴────────────────┴─────────────────┴──────────────────┘

```

### A. Layout Header Tetap (12 Byte)

* **Magic:** `[0xCE, 0x11, 0x54, 0x4D]` (`CELLTM`)
* **Version:** `0x01`
* **Opcode:** Menentukan skema deserialisasi payload.
* **Sequence:** Counter urutan paket 32-bit untuk mendeteksi kehilangan paket.
* **Payload Length:** Panjang byte payload biner yang mengikuti header.

### B. Katalog Opcode & Ukuran Payload

| Opcode | Tipe Event | Ukuran Payload | Deskripsi Data |
| --- | --- | --- | --- |
| `0x01` | **Heartbeat** | 10 Byte | `uptime_ticks (u64)` + `status_flags (u16)` |
| `0x02` | **PmmSnapshot** | 24 Byte | `usable_frames (u64)` + `free_frames (u64)` + `largest_contig (u64)` |
| `0x03` | **QueueMetrics** | 20 Byte | `queue_id (u32)` + `capacity (u32)` + `count (u32)` + `dropped (u32)` + `watermark_state (u8)` + padding (3B) |
| `0x04` | **TensorExecution** | 30 Byte | `phys_addr (u64)` + `elements (u32)` + `reduction_sum (f32)` + `vector_mode (u8)` + `trace_id (u64)` + padding (1B) |
| `0x05` | **FaultIncident** | 22 Byte | `failed_node (u8)` + `error_code (u16)` + `reason (19B fixed UTF-8/ASCII)` |

### C. Emisi Bare-Metal & Parser Host

* **Emisi Serial:** Setiap frame biner dipancarkan sebagai satu baris hex berawalan tag deterministik: `[CELL TM] ce11544d01...`.
* **Host Decoder:** Skrip `scripts/parse_telemetry.py` membaca stream serial dari QEMU, memvalidasi header magic, dan mengubah paket biner menjadi format JSON lines (NDJSON) secara *real-time*.

---

## 8. Verifikasi CI/CD & Invarian Sistem

Skrip otomatis `scripts/verify_run.py` menjalankan pengujian end-to-end pada lingkungan virtualisasi QEMU. Build dinyatakan lolos jika dan hanya jika **13 dari 13 asersi** terpenuhi secara simultan:

1. **SMP 4-Core Bootstrap:** Core 0, 1, 2, dan 3 berhasil online dan melaporkan kesiapan ke Limine.
2. **Flow Control Throttling (HWM Engage):** Ingress mendeteksi antrean $\ge 75\%$ dan menahan pemompaan data.
3. **Flow Control Recovery (LWM Release):** Ingress melanjutkan pemompaan setelah antrean surut $\le 25\%$.
4. **Zero Dropped Packets Invariant:** Tidak ada satupun elemen antrean yang hilang selama siklus hidup beban puncak.
5. **Chained AVX Compute Checksum:** Hasil komputasi $\text{ReLU}((A \times B) + \text{Bias})$ bernilai tepat **$51.200,0$**.
6. **Fault Containment (Trace 4):** Paket anomali dibypass dan dicatat di supervisor tanpa memicu *triple fault* atau *kernel panic*.
7. **PMM 4-Frame Contiguous Reclaim:** 4 frame fisik (16 KiB) yang dialokasikan di awal dikembalikan utuh ke bitmap pool.
8. **Supervisor Zero-Crash Guarantee:** Laporan akhir supervisor menunjukkan $9$ sukses, $1$ terisolasi, $0$ crash.
9. **CELLTM Telemetry Stream Coverage:** Seluruh jenis opcode event biner terdeteksi dalam aliran serial UART.
10. **20-Byte QueueMetrics Wire Format:** Format telemetri antrean memuat status `watermark_state` sesuai spesifikasi bitwise.
11. **AVX Attention Checksum (sum=162.42):** Scaled Dot-Product $Q@K^T/\sqrt{d_k}@V$ menghasilkan checksum tepat.
12. **AVX RMSNorm Checksum (sum=32.00):** Root Mean Square Normalization menghasilkan checksum tepat.
13. **Full Transformer Block Checksum (sum=418.42):** Rangkaian lengkap Norm→Attn→Res→Norm→FFN→Res menghasilkan checksum tepat.
