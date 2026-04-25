"""
Decode the current board-facing single-output `output.bin`.

This helper targets the deploy package interface we observed in the generated driver:
  - output normal shape: (1, 256), laid out as [real128, imag128]
  - output datatype    : usually INT16 for the current low-bit build

Typical usage:
    python board_decode_output_bin.py ^
        --input_bin output.bin ^
        --out_npy output_1x256_raw.npy ^
        --length 256 ^
        --datatype INT16

Supported raw sizes:
  - length * datatype_bytes      : one output trace, exactly matching the current driver metadata
  - 2 * length * datatype_bytes  : two packed halves, where one half may be all-zero
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_LENGTH = 256
BYTES_PER_INT24 = 3


def _bytes_per_elem(datatype: str) -> int:
    if datatype == "INT16":
        return 2
    if datatype == "INT24":
        return 3
    raise ValueError(f"Unsupported datatype: {datatype}")


def _decode_int24_le(raw_bytes: np.ndarray) -> np.ndarray:
    if raw_bytes.dtype != np.uint8:
        raw_bytes = raw_bytes.astype(np.uint8, copy=False)
    if raw_bytes.size % BYTES_PER_INT24 != 0:
        raise ValueError(f"INT24 payload length must be a multiple of 3 bytes, got {raw_bytes.size}")

    triples = raw_bytes.reshape(-1, BYTES_PER_INT24).astype(np.uint32)
    vals = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    sign_bit = 1 << 23
    neg_mask = (vals & sign_bit) != 0
    vals = vals.astype(np.int32)
    vals[neg_mask] -= 1 << 24
    return vals


def _decode_int16_le(raw_bytes: np.ndarray) -> np.ndarray:
    if raw_bytes.dtype != np.uint8:
        raw_bytes = raw_bytes.astype(np.uint8, copy=False)
    if raw_bytes.size % 2 != 0:
        raise ValueError(f"INT16 payload length must be a multiple of 2 bytes, got {raw_bytes.size}")
    return np.frombuffer(raw_bytes.tobytes(), dtype="<i2").astype(np.int32)


def _decode_signed_le(raw_bytes: np.ndarray, datatype: str) -> np.ndarray:
    if datatype == "INT16":
        return _decode_int16_le(raw_bytes)
    if datatype == "INT24":
        return _decode_int24_le(raw_bytes)
    raise ValueError(f"Unsupported datatype: {datatype}")


def _select_payload(raw: np.ndarray, half: str, trace_bytes: int) -> tuple[np.ndarray, str]:
    if raw.size == trace_bytes:
        return raw, "single_trace"

    if raw.size != 2 * trace_bytes:
        raise ValueError(
            f"Expected either {trace_bytes} or {2 * trace_bytes} raw bytes, got {raw.size}"
        )

    first = raw[:trace_bytes]
    second = raw[trace_bytes:]
    first_nz = int(np.count_nonzero(first))
    second_nz = int(np.count_nonzero(second))

    if half == "first":
        return first, f"forced_first_half nonzero={first_nz}"
    if half == "second":
        return second, f"forced_second_half nonzero={second_nz}"

    if first_nz > 0 and second_nz == 0:
        return first, f"auto_first_half nonzero={first_nz}"
    if second_nz > 0 and first_nz == 0:
        return second, f"auto_second_half nonzero={second_nz}"

    raise ValueError(
        "Found a 1536-byte payload but could not auto-select a unique active half. "
        f"first_half_nonzero={first_nz}, second_half_nonzero={second_nz}. "
        "Re-run with --half first or --half second."
    )


def main(
    input_bin: str,
    out_npy: str,
    out_txt: str | None,
    half: str,
    datatype: str,
    length: int,
) -> None:
    input_path = Path(input_bin).resolve()
    out_npy_path = Path(out_npy).resolve()
    out_txt_path = Path(out_txt).resolve() if out_txt is not None else None
    datatype = datatype.upper()
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input bin not found: {input_path}")

    raw = np.fromfile(input_path, dtype=np.uint8)
    trace_bytes = length * _bytes_per_elem(datatype)
    selected_raw, payload_mode = _select_payload(raw, half, trace_bytes)
    decoded = _decode_signed_le(selected_raw, datatype).reshape(1, length)

    out_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, decoded.astype(np.int32))

    if out_txt_path is not None:
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_txt_path, decoded.astype(np.int32), fmt="%d")

    print(f"[OK] Wrote decoded output npy: {out_npy_path}")
    if out_txt_path is not None:
        print(f"[OK] Wrote decoded output txt: {out_txt_path}")
    print(f"raw bytes                : {raw.size}")
    print(f"datatype                 : {datatype}")
    print(f"output length            : {length}")
    print(f"expected trace bytes     : {trace_bytes}")
    print(f"payload selection        : {payload_mode}")
    print(f"decoded shape            : {decoded.shape}")
    print(f"decoded dtype            : {decoded.dtype}")
    print(f"decoded min/max          : {decoded.min()} / {decoded.max()}")
    print(f"decoded nonzero count    : {np.count_nonzero(decoded)}")
    print(f"board contract           : this file represents one batch of {length} scalar {datatype} outputs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_bin", type=str, default="output.bin")
    ap.add_argument("--out_npy", type=str, default="output_1x256_raw.npy")
    ap.add_argument(
        "--out_txt",
        type=str,
        default=None,
        help="Optional text dump for quick inspection.",
    )
    ap.add_argument(
        "--half",
        type=str,
        choices=["auto", "first", "second"],
        default="auto",
        help="How to handle payloads that contain two traces instead of one.",
    )
    ap.add_argument(
        "--datatype",
        type=str,
        choices=["INT16", "INT24", "int16", "int24"],
        default="INT16",
        help="Packed output datatype reported by the generated FINN driver.",
    )
    ap.add_argument(
        "--length",
        type=int,
        default=DEFAULT_OUTPUT_LENGTH,
        help="Number of scalar output elements in oshape_normal, e.g. 256 for the compact complex probe build.",
    )
    args = ap.parse_args()
    main(
        input_bin=args.input_bin,
        out_npy=args.out_npy,
        out_txt=args.out_txt,
        half=args.half,
        datatype=args.datatype,
        length=args.length,
    )
