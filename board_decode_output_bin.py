"""
Decode the current board-facing INT24 single-output `output.bin`.

This helper targets the deploy package interface we observed in the generated driver:
  - output normal shape: (1, 256)
  - output datatype    : INT24

Typical usage:
    python board_decode_output_bin.py ^
        --input_bin output.bin ^
        --out_npy output_1x256_raw.npy

Supported raw sizes:
  - 768 bytes  : one INT24 output trace, exactly matching the current driver metadata
  - 1536 bytes : two packed halves, where one half may be all-zero because the host code
                 was still assuming a historical (1, 2, 256) layout
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EXPECTED_OUTPUT_LENGTH = 256
BYTES_PER_INT24 = 3
TRACE_BYTES = EXPECTED_OUTPUT_LENGTH * BYTES_PER_INT24


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


def _select_payload(raw: np.ndarray, half: str) -> tuple[np.ndarray, str]:
    if raw.size == TRACE_BYTES:
        return raw, "single_trace"

    if raw.size != 2 * TRACE_BYTES:
        raise ValueError(
            f"Expected either {TRACE_BYTES} or {2 * TRACE_BYTES} raw bytes, got {raw.size}"
        )

    first = raw[:TRACE_BYTES]
    second = raw[TRACE_BYTES:]
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


def main(input_bin: str, out_npy: str, out_txt: str | None, half: str) -> None:
    input_path = Path(input_bin).resolve()
    out_npy_path = Path(out_npy).resolve()
    out_txt_path = Path(out_txt).resolve() if out_txt is not None else None

    if not input_path.is_file():
        raise FileNotFoundError(f"Input bin not found: {input_path}")

    raw = np.fromfile(input_path, dtype=np.uint8)
    selected_raw, payload_mode = _select_payload(raw, half)
    decoded = _decode_int24_le(selected_raw).reshape(1, EXPECTED_OUTPUT_LENGTH)

    out_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, decoded.astype(np.int32))

    if out_txt_path is not None:
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_txt_path, decoded.astype(np.int32), fmt="%d")

    print(f"[OK] Wrote decoded output npy: {out_npy_path}")
    if out_txt_path is not None:
        print(f"[OK] Wrote decoded output txt: {out_txt_path}")
    print(f"raw bytes                : {raw.size}")
    print(f"payload selection        : {payload_mode}")
    print(f"decoded shape            : {decoded.shape}")
    print(f"decoded dtype            : {decoded.dtype}")
    print(f"decoded min/max          : {decoded.min()} / {decoded.max()}")
    print(f"decoded nonzero count    : {np.count_nonzero(decoded)}")
    print("board contract           : this file represents one batch of 256 scalar INT24 outputs")


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
        help="How to handle 1536-byte payloads.",
    )
    args = ap.parse_args()
    main(
        input_bin=args.input_bin,
        out_npy=args.out_npy,
        out_txt=args.out_txt,
        half=args.half,
    )
