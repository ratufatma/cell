#!/usr/bin/env python3
"""
scripts/generate_weights.py
Generates weights.bin binary for CELL Transformer Block.

Format:
  Header (64 bytes): Magic(8) + Version(2) + Reserved(6) + DataLen(4) + Checksum(4) + Pad(40)
  Payload (N floats): gamma1(64) + K0(64) + K1(64) + V0(64) + V1(64) + gamma2(64) + bias(64) + W_ffn(4096)
"""

import struct
import os
import sys

MAGIC = b"CELLWGHT"
VERSION = 1
HEADER_SIZE = 64


def generate_weights_bin(output_path: str):
    gamma1 = [1.0] * 64
    k0 = [1.0] * 64
    k1 = [0.5] * 64
    v0 = [2.0] * 64
    v1 = [4.0] * 64
    gamma2 = [1.0] * 64
    bias = [-1.0] * 64
    w_ffn = [0.0625] * (64 * 64)

    all_floats = gamma1 + k0 + k1 + v0 + v1 + gamma2 + bias + w_ffn
    payload_bytes = struct.pack(f"<{len(all_floats)}f", *all_floats)
    payload_len = len(payload_bytes)
    payload_sum = sum(all_floats)

    header = struct.pack(
        "<8sH6sIf40s",
        MAGIC,
        VERSION,
        b"\x00" * 6,
        payload_len,
        payload_sum,
        b"\x00" * 40,
    )

    assert len(header) == HEADER_SIZE, f"Header size mismatch: {len(header)} != {HEADER_SIZE}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(payload_bytes)

    total = HEADER_SIZE + payload_len
    print(f"[GENERATE] weights.bin: {total} bytes (header={HEADER_SIZE}, payload={payload_len}, floats={len(all_floats)})")
    print(f"[GENERATE] payload_sum={payload_sum}")
    return output_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "target", "weights.bin"
    )
    generate_weights_bin(out)
