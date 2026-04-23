"""
Step 6: Compare decoded board/runtime output against the SSFM reference.

Typical usage:
    python step6_compare_deploy_output.py \
        --case verification_io/verification_case.npz \
        --actual board_output.npy

Expected `board_output.npy` formats:
  - [B, 256] compact-complex probe output: first 128 real, last 128 imag
  - [B, N_t] legacy real-channel probe output
  - [B, 1, N_t] legacy real-channel probe output
  - [B, 2, N_t]
  - [B, N_t, 2]
  - [B, 2*N_t] with real first, imag second
  - [2, N_t]
  - [N_t, 2]
  - [2*N_t] with real first, imag second
  - complex [N_t]
  - complex [B, N_t]

Raw board-specific integer dumps such as [1, N_t] int32 are ambiguous and must be
decoded/dequantized first by the board host/runtime code before using this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_output_utils import make_figure_output_dir, save_figure


def _normalize_output(arr: np.ndarray, name: str, expected_n_t: int | None = None) -> np.ndarray:
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        if arr.ndim == 1:
            return np.stack([arr.real, arr.imag], axis=0)[None, :, :].astype(np.float32)
        if arr.ndim == 2:
            return np.stack([arr.real, arr.imag], axis=1).astype(np.float32)
        raise ValueError(f"{name} complex array must be rank 1 or 2, but got {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer) and expected_n_t is not None:
        if (arr.ndim == 1 and arr.shape[0] == expected_n_t) or (
            arr.ndim == 2 and arr.shape[-1] == expected_n_t and arr.shape[0] != 2
        ) or (
            arr.ndim == 1 and arr.shape[0] == 2 * expected_n_t
        ) or (
            arr.ndim == 2 and arr.shape[-1] == 2 * expected_n_t
        ):
            raise ValueError(
                f"{name} looks like a raw integer board dump with shape {arr.shape} and dtype {arr.dtype}. "
                "Decode/dequantize it into [B,N_t] float, [B,2,N_t] float, [B,2*N_t] float, "
                "or complex form before running this script."
            )

    arr = arr.astype(np.float32, copy=False)

    if arr.ndim == 3 and arr.shape[1] == 2:
        return arr
    if expected_n_t is not None and arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == expected_n_t:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 2:
        return np.transpose(arr, (0, 2, 1))
    if expected_n_t is not None and arr.ndim == 2 and arr.shape[1] == expected_n_t:
        return arr[:, None, :]
    if expected_n_t is not None and arr.ndim == 2 and arr.shape[1] == 2 * expected_n_t:
        return arr.reshape(arr.shape[0], 2, expected_n_t)
    if expected_n_t is not None and arr.ndim == 1 and arr.shape[0] == expected_n_t:
        return arr.reshape(1, 1, expected_n_t)
    if expected_n_t is not None and arr.ndim == 1 and arr.shape[0] == 2 * expected_n_t:
        return arr.reshape(1, 2, expected_n_t)
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr[None, :, :]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.transpose(arr[None, :, :], (0, 2, 1))

    raise ValueError(
        f"{name} must have shape [B,N_t], [B,1,N_t], [B,2,N_t], [B,N_t,2], "
        f"[B,2*N_t], [2,N_t], [N_t,2], [N_t], [2*N_t], complex [N_t], "
        f"or complex [B,N_t], but got {arr.shape}"
    )


def _case_string(npz: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz.files:
        return default
    value = npz[key]
    if np.asarray(value).size == 0:
        return default
    return str(np.asarray(value).reshape(-1)[0])


def _load_case(case_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    case = np.load(case_path)
    deploy_format = _case_string(case, "deploy_output_format", "")

    if "expected_output" in case.files:
        expected_raw = np.asarray(case["expected_output"])
        expected_n_t = expected_raw.shape[-1] // 2 if "compact_real" in deploy_format else expected_raw.shape[-1]
        expected = _normalize_output(expected_raw, "expected_output", expected_n_t=expected_n_t)
    elif "ssfm_output_probe" in case.files:
        expected_raw = np.asarray(case["ssfm_output_probe"])
        expected = _normalize_output(expected_raw, "ssfm_output_probe", expected_n_t=expected_raw.shape[-1] // 2)
    elif "ssfm_output_real" in case.files:
        expected_raw = np.asarray(case["ssfm_output_real"])
        expected = _normalize_output(expected_raw, "ssfm_output_real", expected_n_t=expected_raw.shape[-1])
    elif "ssfm_output" in case.files:
        expected = _normalize_output(case["ssfm_output"], "ssfm_output")
    elif "AL_clean_real" in case.files and "AL_clean_imag" in case.files:
        expected = np.stack(
            [
                case["AL_clean_real"].astype(np.float32),
                case["AL_clean_imag"].astype(np.float32),
            ],
            axis=0,
        )[None, :, :]
    else:
        raise ValueError(
            f"{case_path} does not contain SSFM reference data. "
            "Expected `ssfm_output` or `AL_clean_real`/`AL_clean_imag`."
        )

    if "A0_real" in case.files and "A0_imag" in case.files:
        A0 = case["A0_real"].astype(np.float32) + 1j * case["A0_imag"].astype(np.float32)
    elif "u_in" in case.files:
        u_in = np.asarray(case["u_in"], dtype=np.float32)
        if u_in.ndim != 2 or u_in.shape[0] != 1 or u_in.shape[1] % 2 != 0:
            raise ValueError(f"Cannot reconstruct A0 from u_in with shape {u_in.shape}")
        half = u_in.shape[1] // 2
        A0 = u_in[0, :half] + 1j * u_in[0, half:]
    else:
        raise ValueError(
            f"{case_path} does not contain A0 data. Expected `A0_real`/`A0_imag` or `u_in`."
        )

    if "probe_indices" in case.files and "A0_real" in case.files and "A0_imag" in case.files:
        probe_indices = np.asarray(case["probe_indices"], dtype=np.int64).reshape(-1)
        A0 = A0[probe_indices]

    if "deploy_t_grid" in case.files:
        t_grid = np.asarray(case["deploy_t_grid"], dtype=np.float32).reshape(-1)
    elif "t_grid" in case.files:
        t_grid = np.asarray(case["t_grid"], dtype=np.float32).reshape(-1)
        if "probe_indices" in case.files:
            t_grid = t_grid[np.asarray(case["probe_indices"], dtype=np.int64).reshape(-1)]
    else:
        t_grid = np.arange(expected.shape[-1], dtype=np.float32)

    source = _case_string(case, "source", "unknown_source")
    return expected.astype(np.float32), A0.astype(np.complex64), t_grid, source


def _metrics_text(metrics: dict[str, float]) -> str:
    if "complex_rmse" not in metrics:
        return (
            f"RMSE={metrics['rmse']:.3e} | RelRMSE={metrics['relative_rmse']:.3e}\n"
            f"RealRMSE={metrics['real_channel_rmse']:.3e} | MaxAbs={metrics['max_abs_diff']:.3e}"
        )
    return (
        f"RMSE={metrics['rmse']:.3e} | RelRMSE={metrics['relative_rmse']:.3e}\n"
        f"ComplexRMSE={metrics['complex_rmse']:.3e} | AmpRMSE={metrics['amplitude_rmse']:.3e}\n"
        f"MaxAbs={metrics['max_abs_diff']:.3e}"
    )


def _overlay_text(ax, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )


def _save_metrics(output_dir: Path, metrics: dict[str, float]) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] Saved metrics: {metrics_path}")


def _plot_amplitude(
    t_grid: np.ndarray,
    A0: np.ndarray,
    expected_complex: np.ndarray,
    actual_complex: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_grid, np.abs(A0), ":", alpha=0.7, label="Input |A(0)|")
    ax.plot(t_grid, np.abs(expected_complex), alpha=0.8, label="SSFM reference |A(L)|")
    ax.plot(t_grid, np.abs(actual_complex), "--", label="Board/runtime |A(L)|")
    ax.set_xlabel("Time")
    ax.set_ylabel("|A|")
    ax.set_title("Amplitude Comparison at z = L")
    ax.grid(True)
    ax.legend()
    _overlay_text(ax, _metrics_text(metrics))
    fig.tight_layout()
    save_figure(fig, output_dir, "amplitude_comparison")
    plt.close(fig)


def _plot_constellation(
    expected_complex: np.ndarray,
    actual_complex: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected_complex.real, expected_complex.imag, s=10, alpha=0.45, label="SSFM reference")
    ax.scatter(actual_complex.real, actual_complex.imag, s=10, alpha=0.65, label="Board/runtime")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title("Constellation Comparison at z = L")
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal", "box")
    _overlay_text(ax, _metrics_text(metrics))
    fig.tight_layout()
    save_figure(fig, output_dir, "constellation_comparison")
    plt.close(fig)


def _plot_error(
    t_grid: np.ndarray,
    expected_complex: np.ndarray,
    actual_complex: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
) -> None:
    diff_complex = actual_complex - expected_complex
    amp_diff = np.abs(actual_complex) - np.abs(expected_complex)
    abs_complex_diff = np.abs(diff_complex)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_grid, diff_complex.real, label="Real error")
    axes[0].plot(t_grid, diff_complex.imag, label="Imag error", alpha=0.8)
    axes[0].set_ylabel("Channel Error")
    axes[0].set_title("Real/Imag Error at z = L")
    axes[0].grid(True)
    axes[0].legend()
    _overlay_text(axes[0], _metrics_text(metrics))

    axes[1].plot(t_grid, amp_diff, label="Amplitude error")
    axes[1].plot(t_grid, abs_complex_diff, label="|Complex error|", alpha=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Amplitude and Complex Error at z = L")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, output_dir, "error_summary")
    plt.close(fig)


def _plot_real_channel(
    t_grid: np.ndarray,
    A0: np.ndarray,
    expected_real: np.ndarray,
    actual_real: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_grid, A0.real, ":", alpha=0.7, label="Input real(A(0))")
    ax.plot(t_grid, expected_real, alpha=0.8, label="SSFM reference real(A(L))")
    ax.plot(t_grid, actual_real, "--", label="Board/runtime real(A(L))")
    ax.set_xlabel("Time")
    ax.set_ylabel("Real channel")
    ax.set_title("Real-Channel Comparison at z = L")
    ax.grid(True)
    ax.legend()
    _overlay_text(ax, _metrics_text(metrics))
    fig.tight_layout()
    save_figure(fig, output_dir, "real_channel_comparison")
    plt.close(fig)


def _plot_real_error(
    t_grid: np.ndarray,
    expected_real: np.ndarray,
    actual_real: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_grid, actual_real - expected_real, label="Real-channel error")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.set_title("Real-Channel Error at z = L")
    ax.grid(True)
    ax.legend()
    _overlay_text(ax, _metrics_text(metrics))
    fig.tight_layout()
    save_figure(fig, output_dir, "real_channel_error")
    plt.close(fig)


def _compute_metrics(expected_arr: np.ndarray, actual_arr: np.ndarray) -> dict[str, float]:
    diff = actual_arr - expected_arr
    abs_diff = np.abs(diff)

    if expected_arr.shape[1] == 1:
        rmse = float(np.sqrt(np.mean(diff**2)))
        return {
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "rmse": rmse,
            "relative_rmse": float(rmse / (np.sqrt(np.mean(expected_arr**2)) + 1e-12)),
            "real_channel_rmse": rmse,
        }

    expected_complex = expected_arr[:, 0, :] + 1j * expected_arr[:, 1, :]
    actual_complex = actual_arr[:, 0, :] + 1j * actual_arr[:, 1, :]
    complex_diff = actual_complex - expected_complex

    metrics = {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "relative_rmse": float(
            np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(expected_arr**2)) + 1e-12)
        ),
        "real_channel_rmse": float(np.sqrt(np.mean(diff[:, 0, :] ** 2))),
        "imag_channel_rmse": float(np.sqrt(np.mean(diff[:, 1, :] ** 2))),
        "complex_rmse": float(np.sqrt(np.mean(np.abs(complex_diff) ** 2))),
        "complex_relative_rmse": float(
            np.sqrt(np.mean(np.abs(complex_diff) ** 2))
            / (np.sqrt(np.mean(np.abs(expected_complex) ** 2)) + 1e-12)
        ),
        "amplitude_mae": float(np.mean(np.abs(np.abs(actual_complex) - np.abs(expected_complex)))),
        "amplitude_rmse": float(
            np.sqrt(np.mean((np.abs(actual_complex) - np.abs(expected_complex)) ** 2))
        ),
    }
    return metrics


def _print_metric(label: str, value: float) -> None:
    print(f"{label:<28} {value:.6e}")


def main(
    case: str,
    actual: str,
    atol: float,
    rtol: float,
    diff_out: str | None,
    output_dir: str | None,
) -> None:
    case_path = Path(case).resolve()
    actual_path = Path(actual).resolve()

    if not case_path.is_file():
        raise FileNotFoundError(f"Verification case not found: {case_path}")
    if not actual_path.is_file():
        raise FileNotFoundError(f"Actual output file not found: {actual_path}")

    expected_arr, A0, t_grid, source = _load_case(case_path)
    actual_arr = _normalize_output(np.load(actual_path), "actual", expected_n_t=expected_arr.shape[-1])

    if expected_arr.shape != actual_arr.shape:
        raise ValueError(
            f"Shape mismatch: expected {expected_arr.shape}, actual {actual_arr.shape}"
        )

    metrics = _compute_metrics(expected_arr, actual_arr)
    passed = bool(np.allclose(actual_arr, expected_arr, atol=atol, rtol=rtol))
    metrics["passed"] = passed
    metrics["atol"] = float(atol)
    metrics["rtol"] = float(rtol)

    if output_dir is None:
        figures_dir = make_figure_output_dir(__file__)
    else:
        figures_dir = Path(output_dir).resolve()
        figures_dir.mkdir(parents=True, exist_ok=True)

    if expected_arr.shape[1] == 1:
        expected_real = expected_arr[0, 0, :]
        actual_real = actual_arr[0, 0, :]
        _plot_real_channel(t_grid, A0, expected_real, actual_real, figures_dir, metrics)
        _plot_real_error(t_grid, expected_real, actual_real, figures_dir, metrics)
    else:
        expected_complex = expected_arr[0, 0, :] + 1j * expected_arr[0, 1, :]
        actual_complex = actual_arr[0, 0, :] + 1j * actual_arr[0, 1, :]
        _plot_amplitude(t_grid, A0, expected_complex, actual_complex, figures_dir, metrics)
        _plot_constellation(expected_complex, actual_complex, figures_dir, metrics)
        _plot_error(t_grid, expected_complex, actual_complex, figures_dir, metrics)
    _save_metrics(figures_dir, metrics)

    if diff_out is not None:
        diff_out_path = Path(diff_out).resolve()
        diff_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(diff_out_path, (actual_arr - expected_arr).astype(np.float32))
        print(f"[OK] Saved diff tensor: {diff_out_path}")

    print(f"verification case: {case_path}")
    print(f"actual output    : {actual_path}")
    print(f"source           : {source}")
    print(f"shape            : {expected_arr.shape}")
    print(f"figures          : {figures_dir}")
    print("")
    _print_metric("max_abs_diff", metrics["max_abs_diff"])
    _print_metric("mean_abs_diff", metrics["mean_abs_diff"])
    _print_metric("rmse", metrics["rmse"])
    _print_metric("relative_rmse", metrics["relative_rmse"])
    _print_metric("real_channel_rmse", metrics["real_channel_rmse"])
    if expected_arr.shape[1] == 2:
        _print_metric("imag_channel_rmse", metrics["imag_channel_rmse"])
        _print_metric("complex_rmse", metrics["complex_rmse"])
        _print_metric("complex_relative_rmse", metrics["complex_relative_rmse"])
        _print_metric("amplitude_mae", metrics["amplitude_mae"])
        _print_metric("amplitude_rmse", metrics["amplitude_rmse"])
    print("")
    print(f"tolerance        : atol={atol:.3e}, rtol={rtol:.3e}")
    print(f"verification     : {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, required=True, help="Path to verification_case.npz.")
    ap.add_argument("--actual", type=str, required=True, help="Path to decoded board/runtime output .npy.")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--diff_out", type=str, default=None)
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory for saved figures and metrics.json.",
    )
    args = ap.parse_args()
    main(
        case=args.case,
        actual=args.actual,
        atol=args.atol,
        rtol=args.rtol,
        diff_out=args.diff_out,
        output_dir=args.output_dir,
    )
