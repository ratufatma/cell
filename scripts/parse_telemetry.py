#!/usr/bin/env python3
"""Decode CELLTM telemetry lines from the CELL QEMU serial stream."""

import argparse
import json
import re
import struct
import sys
from typing import Any, BinaryIO, Dict, Optional, TextIO, Tuple

MAGIC_BYTES = b"\xce\x11\x54\x4d"
VERSION = 1
HEADER_FORMAT = "<4sBBIH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

OPCODES = {
    0x01: "Heartbeat",
    0x02: "PmmSnapshot",
    0x03: "QueueMetrics",
    0x04: "TensorExecution",
    0x05: "FaultIncident",
}

HEX_LINE_REGEX = re.compile(r"\[CELL TM\]\s+([0-9a-fA-F]+)")


def decode_context(payload: bytes, offset: int = 0) -> Dict[str, Any]:
    trace_id, timestamp, origin_node = struct.unpack_from("<QQH", payload, offset)
    return {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "origin_node": origin_node,
    }


def decode_heartbeat(payload: bytes) -> Dict[str, Any]:
    if len(payload) != 10:
        return {"raw_hex": payload.hex()}
    timestamp, node_id = struct.unpack("<QH", payload)
    return {"timestamp": timestamp, "node_id": node_id}


def decode_pmm_snapshot(payload: bytes) -> Dict[str, Any]:
    if len(payload) != 24:
        return {"raw_hex": payload.hex()}
    usable, free, largest = struct.unpack("<QQQ", payload)
    return {
        "usable_frames": usable,
        "free_frames": free,
        "largest_free_run": largest,
        "usable_kib": usable * 4,
        "free_kib": free * 4,
        "largest_free_run_kib": largest * 4,
    }


def decode_queue_metrics(payload: bytes) -> Dict[str, Any]:
    if len(payload) != 16:
        return {"raw_hex": payload.hex()}
    queue_id, capacity, pushed, popped, dropped = struct.unpack("<HHIII", payload)
    return {
        "queue_id": queue_id,
        "capacity": capacity,
        "pushed": pushed,
        "popped": popped,
        "dropped": dropped,
    }


def decode_tensor_execution(payload: bytes) -> Dict[str, Any]:
    if len(payload) != 30:
        return {"raw_hex": payload.hex()}
    result = decode_context(payload)
    elements, frame_count = struct.unpack_from("<IH", payload, 18)
    dtype, simd_level = payload[24], payload[25]
    (sum_bits,) = struct.unpack_from("<I", payload, 26)
    result.update(
        {
            "elements": elements,
            "frame_count": frame_count,
            "dtype": {0: "F32", 1: "F16", 2: "BF16", 3: "I8", 4: "U8"}.get(
                dtype, f"Unknown({dtype})"
            ),
            "simd_level": {0: "Scalar", 1: "SSE-128", 2: "AVX-256"}.get(
                simd_level, f"Unknown({simd_level})"
            ),
            "sum": struct.unpack("<f", struct.pack("<I", sum_bits))[0],
        }
    )
    return result


def decode_fault_incident(payload: bytes) -> Dict[str, Any]:
    if len(payload) != 22:
        return {"raw_hex": payload.hex()}
    result = decode_context(payload)
    node_id, reason_code = struct.unpack_from("<HH", payload, 18)
    result.update({"node_id": node_id, "reason_code": reason_code})
    return result


PAYLOAD_DECODERS = {
    0x01: decode_heartbeat,
    0x02: decode_pmm_snapshot,
    0x03: decode_queue_metrics,
    0x04: decode_tensor_execution,
    0x05: decode_fault_incident,
}


def parse_packet(packet_bytes: bytes) -> Optional[Tuple[Dict[str, Any], int]]:
    if len(packet_bytes) < HEADER_SIZE:
        return None
    magic, version, opcode, sequence, payload_len = struct.unpack(
        HEADER_FORMAT, packet_bytes[:HEADER_SIZE]
    )
    if magic != MAGIC_BYTES or version != VERSION:
        return None
    total_size = HEADER_SIZE + payload_len
    if len(packet_bytes) != total_size:
        return None
    decoder = PAYLOAD_DECODERS.get(opcode)
    data = decoder(packet_bytes[HEADER_SIZE:]) if decoder else {
        "raw_hex": packet_bytes[HEADER_SIZE:].hex()
    }
    return (
        {
            "protocol": "CELLTM",
            "version": version,
            "sequence": sequence,
            "event_type": OPCODES.get(opcode, f"Unknown(0x{opcode:02x})"),
            "opcode": opcode,
            "payload_len": payload_len,
            "data": data,
        },
        total_size,
    )


def process_stream(input_stream: TextIO, output_stream: TextIO, pretty: bool, verbose: bool) -> None:
    for raw_line in input_stream:
        line = raw_line.strip()
        match = HEX_LINE_REGEX.search(line)
        if not match:
            if verbose and line:
                print(line, file=sys.stderr, flush=True)
            continue
        hex_text = match.group(1)
        try:
            packet = bytes.fromhex(hex_text)
            parsed = parse_packet(packet)
        except ValueError:
            parsed = None
            if verbose:
                print(f"[WARN] invalid telemetry hex: {hex_text}", file=sys.stderr, flush=True)
        if parsed is not None:
            event, _ = parsed
            output_stream.write(json.dumps(event, indent=2 if pretty else None) + "\n")
            output_stream.flush()
        elif verbose:
            print(f"[WARN] invalid CELLTM frame: {hex_text}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode CELLTM UART telemetry to JSON.")
    parser.add_argument("-i", "--input", help="Serial log path; defaults to stdin.")
    parser.add_argument("-o", "--output", help="JSONL output path; defaults to stdout.")
    parser.add_argument("-p", "--pretty", action="store_true", help="Pretty-print JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Forward kernel logs to stderr.")
    args = parser.parse_args()

    input_stream = open(args.input, encoding="utf-8", errors="replace") if args.input else sys.stdin
    output_stream = open(args.output, "a", encoding="utf-8") if args.output else sys.stdout
    try:
        process_stream(input_stream, output_stream, args.pretty, args.verbose)
    except KeyboardInterrupt:
        pass
    finally:
        if args.input:
            input_stream.close()
        if args.output:
            output_stream.close()


if __name__ == "__main__":
    main()
