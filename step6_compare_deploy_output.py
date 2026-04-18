"""
Step 6: Compare board/runtime output against the expected deploy output.

Typical usage:
    python step6_compare_deploy_output.py \
        --expected verification_io/expected_output.npy \
        --actual board_output.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _normalize_output(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 2:
        return np.transpose(arr, (0, 2, 1))
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr[None, :, :]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.transpose(arr[None, :, :], (0, 2, 1))

    raise ValueError(
        f"{name} must have shape [B,2,N_t], [B,N_t,2], [2,N_t], or [N_t,2], "
        f"but got {arr.shape}"
    )


def _print_metric(label: str, value: float) -> None:
    print(f"{label:<28} {value:.6e}")


def main(expected: str, actual: str, atol: float, rtol: float, diff_out: str | None) -> None:
    expected_path = Path(expected).resolve()
    actual_path = Path(actual).resolve()

    if not expected_path.is_file():
        raise FileNotFoundError(f"Expected output file not found: {expected_path}")
    if not actual_path.is_file():
        raise FileNotFoundError(f"Actual output file not found: {actual_path}")

    expected_arr = _normalize_output(np.load(expected_path), "expected")
    actual_arr = _normalize_output(np.load(actual_path), "actual")

    if expected_arr.shape != actual_arr.shape:
        raise ValueError(
            f"Shape mismatch: expected {expected_arr.shape}, actual {actual_arr.shape}"
        )

    diff = actual_arr - expected_arr
    abs_diff = np.abs(diff)

    rmse = float(np.sqrt(np.mean(diff**2)))
    rel_rmse = float(rmse / (np.sqrt(np.mean(expected_arr**2)) + 1e-12))
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    real_rmse = float(np.sqrt(np.mean(diff[:, 0, :] ** 2)))
    imag_rmse = float(np.sqrt(np.mean(diff[:, 1, :] ** 2)))

    exp_complex = expected_arr[:, 0, :] + 1j * expected_arr[:, 1, :]
    act_complex = actual_arr[:, 0, :] + 1j * actual_arr[:, 1, :]
    complex_rmse = float(np.sqrt(np.mean(np.abs(act_complex - exp_complex) ** 2)))
    complex_rel_rmse = float(
        complex_rmse / (np.sqrt(np.mean(np.abs(exp_complex) ** 2)) + 1e-12)
    )

    if diff_out is not None:
        diff_out_path = Path(diff_out).resolve()
        diff_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(diff_out_path, diff.astype(np.float32))
        print(f"[OK] Saved diff tensor to: {diff_out_path}")

    print(f"expected file: {expected_path}")
    print(f"actual file  : {actual_path}")
    print(f"shape        : {expected_arr.shape}")
    print("")
    _print_metric("max_abs_diff", max_abs)
    _print_metric("mean_abs_diff", mean_abs)
    _print_metric("rmse", rmse)
    _print_metric("relative_rmse", rel_rmse)
    _print_metric("real_channel_rmse", real_rmse)
    _print_metric("imag_channel_rmse", imag_rmse)
    _print_metric("complex_rmse", complex_rmse)
    _print_metric("complex_relative_rmse", complex_rel_rmse)

    passed = bool(np.allclose(actual_arr, expected_arr, atol=atol, rtol=rtol))
    print("")
    print(f"tolerance    : atol={atol:.3e}, rtol={rtol:.3e}")
    print(f"verification : {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", type=str, required=True)
    ap.add_argument("--actual", type=str, required=True)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--diff_out", type=str, default=None)
    args = ap.parse_args()
    main(
        expected=args.expected,
        actual=args.actual,
        atol=args.atol,
        rtol=args.rtol,
        diff_out=args.diff_out,
    )
