#!/usr/bin/env python3
"""
scripts/verify_run.py
End-to-End Automated CI/CD Assertion Runner for CELL Kernel.

Validasi:
  1. 4-Core SMP Bootstrap (Core 0, 1, 2, 3 online)
  2. Flow Control & Backpressure (HWM throttled, LWM released, 0 dropped)
  3. Dynamic Dispatch & Chained Compute (MatMul -> VectorAdd -> ReLU == 51200.0)
  4. Fault Containment (Trace 4 isolated, zero kernel crash)
  5. PMM 4-Frame Contiguous Memory Reclaim
  6. CELLTM Binary Telemetry Stream (20-byte QueueMetrics, Tensor, Fault, PMM)
"""

import os
import re
import struct
import subprocess
import sys
import time
from typing import Set

TIMEOUT_SECONDS = 30
HEX_TELEMETRY_REGEX = re.compile(r"\[CELL TM\]\s+([0-9a-fA-F]+)")


def run_verification() -> int:
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    runner_path = os.path.join(workspace_root, "scripts", "run_qemu.sh")

    if not os.path.exists(runner_path):
        print(f"\033[91m[ERROR]\033[0m Runner script not found: {runner_path}")
        return 1

    print("[CI/CD] Starting QEMU 4-Core DAG Pipeline with Flow Control & Telemetry...")
    start_time = time.time()

    proc = subprocess.Popen(
        [runner_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    import threading
    output_chunks = []
    stop_reading = threading.Event()

    def reader():
        while not stop_reading.is_set():
            chunk = proc.stdout.read(4096)
            if chunk:
                output_chunks.append(chunk)
            else:
                break

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        while time.time() - start_time < TIMEOUT_SECONDS:
            time.sleep(0.1)
            combined = b"".join(output_chunks)
            if b"Zero crash" in combined:
                break
    finally:
        stop_reading.set()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        t.join(timeout=2)

    full_output = b"".join(output_chunks).decode("utf-8", errors="replace")

    smp_cores_online: Set[int] = set()
    flow_control_engaged = False
    flow_control_released = False
    zero_dropped_confirmed = False
    chained_compute_sum_verified = False
    fault_trace_isolated = False
    pmm_frames_returned = False
    zero_crash_reported = False

    captured_telemetry_opcodes: Set[int] = set()
    queue_metrics_payload_valid = False
    attention_checksum_verified = False
    rmsnorm_checksum_verified = False
    pmm_free_count = 0

    for line_str in full_output.splitlines():
        line_str = line_str.strip()
        if not line_str:
            continue

        if "Core 0 (BSP Ingress) online" in line_str:
            smp_cores_online.add(0)
        elif "Core 1 (AP1 Arbiter) online" in line_str:
            smp_cores_online.add(1)
        elif "Core 2 (AP2 Compute) online" in line_str:
            smp_cores_online.add(2)
        elif "Core 3 (AP3 Supervisor) online" in line_str:
            smp_cores_online.add(3)

        if "Backpressure active" in line_str or "Backpressure engaged" in line_str:
            flow_control_engaged = True
        if "Backpressure released" in line_str or "Low watermark reached" in line_str:
            flow_control_released = True
        if "0 dropped" in line_str:
            zero_dropped_confirmed = True

        if "sum=51200" in line_str:
            chained_compute_sum_verified = True

        if "Attention verified sum=" in line_str and "expected=162.42" in line_str:
            attention_checksum_verified = True

        if "RMSNorm verified sum=" in line_str and "expected=32.00" in line_str:
            rmsnorm_checksum_verified = True

        if "Trace 4 isolated failure captured" in line_str:
            fault_trace_isolated = True

        if "contiguous frames count=4 returned to bitmap" in line_str:
            pmm_frames_returned = True
            pmm_free_count += 1

        if "Pipeline 4-Core tuntas" in line_str and "Zero crash" in line_str:
            zero_crash_reported = True

        tm_match = HEX_TELEMETRY_REGEX.search(line_str)
        if tm_match:
            try:
                raw_bytes = bytes.fromhex(tm_match.group(1))
                if len(raw_bytes) >= 12 and raw_bytes[:4] == b"\xce\x11\x54\x4d":
                    opcode = raw_bytes[5]
                    captured_telemetry_opcodes.add(opcode)
                    payload_len = struct.unpack("<H", raw_bytes[10:12])[0]
                    if opcode == 0x03 and payload_len == 20:
                        queue_metrics_payload_valid = True
            except ValueError:
                pass

    print("\n" + "=" * 65)
    print("           CELL KERNEL CI/CD ASSERTION REPORT")
    print("=" * 65)

    assertions = [
        (
            "SMP 4-Core Bootstrap",
            len(smp_cores_online) == 4,
            f"Active: {sorted(list(smp_cores_online))}/4 cores",
        ),
        (
            "Flow Control Throttling (HWM Engage)",
            flow_control_engaged,
            "Backpressure engaged on congestion",
        ),
        (
            "Flow Control Recovery (LWM Release)",
            flow_control_released,
            "Ingress resumed after buffer drain",
        ),
        (
            "Zero Dropped Packets Invariant",
            zero_dropped_confirmed,
            "Buffer lossless delivery confirmed",
        ),
        (
            "Chained AVX Compute (sum=51200.0)",
            chained_compute_sum_verified,
            "MatMul -> VectorAdd -> ReLU exact",
        ),
        (
            "Fault Containment (Trace 4)",
            fault_trace_isolated,
            "Corrupted payload bypassed cleanly",
        ),
        (
            "PMM 4-Frame Contiguous Reclaim",
            pmm_frames_returned,
            f"Physical frames returned to bitmap (count={pmm_free_count})",
        ),
        (
            "Supervisor Zero-Crash Guarantee",
            zero_crash_reported,
            "Clean pipeline completion",
        ),
        (
            "CELLTM Telemetry Stream Coverage",
            len(captured_telemetry_opcodes) >= 4,
            f"Opcodes: {sorted(list(captured_telemetry_opcodes))}",
        ),
        (
            "20-Byte QueueMetrics Wire Format",
            queue_metrics_payload_valid,
            "Includes watermark_state & padding",
        ),
        (
            "AVX Attention Checksum (sum=162.42)",
            attention_checksum_verified,
            "Scaled Dot-Product Q@K^T/sqrt(d)@V exact",
        ),
        (
            "AVX RMSNorm Checksum (sum=32.00)",
            rmsnorm_checksum_verified,
            "Root Mean Square Normalization exact",
        ),
    ]

    all_passed = True
    for name, passed, detail in assertions:
        status_tag = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
        print(f"{status_tag} {name:<42} : {detail}")
        if not passed:
            all_passed = False

    print("=" * 65)
    if all_passed:
        print("\033[92m[SUCCESS] All kernel invariants verified successfully.\033[0m\n")
        return 0
    else:
        print("\033[91m[FAILURE] One or more assertions failed. Check run logs.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_verification())
