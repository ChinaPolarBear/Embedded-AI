"""
Step 6: Compare decoded board/runtime output against the SSFM reference.

Typical usage:
    python step6_compare_deploy_output.py \
        --case verification/verification_case.npz \
        --actual board_output.npy

Batch usage:
    python step6_compare_deploy_output.py \
        --cases_dir verification \
        --actual_name output.npy

Expected `board_output.npy` formats:
  - [B, 2*N_probe] compact-complex probe output: first N_probe real, last N_probe imag
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
# python step6_compare_deploy_output.py --case verification/verification_case.npz --actual output.npy
# mutiple case
# python step6_compare_deploy_output.py --cases_dir verification --actual_name output.npy
# python step6_compare_deploy_output.py --cases_dir verification --actual_name output_dequant.npy


from __future__ import annotations

import argparse
import csv
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


def _save_batch_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    summary_json = output_dir / "batch_summary.json"
    summary_csv = output_dir / "batch_summary.csv"
    summary_json.write_text(json.dumps({"cases": rows}, indent=2), encoding="utf-8")

    fieldnames = [
        "case_name",
        "status",
        "passed",
        "source",
        "case_path",
        "actual_path",
        "shape",
        "max_abs_diff",
        "mean_abs_diff",
        "rmse",
        "relative_rmse",
        "real_channel_rmse",
        "imag_channel_rmse",
        "complex_rmse",
        "complex_relative_rmse",
        "amplitude_mae",
        "amplitude_rmse",
        "message",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[OK] Saved batch summary: {summary_json}")
    print(f"[OK] Saved batch CSV    : {summary_csv}")


def _combined_legend(ax_left, ax_right) -> None:
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right)


def _describe_actual_kind(actual_path: Path) -> str:
    name = actual_path.name.lower()
    if "dequant" in name:
        return "dequantized"
    if "raw" in name:
        return "raw"
    if name == "output.npy":
        return "raw"
    return "provided"


def _plot_amplitude(
    t_grid: np.ndarray,
    A0: np.ndarray,
    expected_complex: np.ndarray,
    actual_complex: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
    actual_kind: str,
) -> None:
    fig, ax_left = plt.subplots(figsize=(12, 4))
    ax_right = ax_left.twinx()

    ax_left.plot(
        t_grid,
        np.abs(A0),
        "--",
        color="tab:blue",
        alpha=0.28,
        linewidth=1.3,
        label="Input |A(0)|",
    )
    ax_left.plot(t_grid, np.abs(expected_complex), "-", color="tab:blue", alpha=0.9, label="SSFM reference |A(L)|")
    ax_right.plot(
        t_grid,
        np.abs(actual_complex),
        "--",
        color="tab:green",
        alpha=0.9,
        linewidth=2.4,
        label="Board/runtime |A(L)|",
    )

    ax_left.set_xlabel("Time")
    ax_left.set_ylabel("|A| input / SSFM", color="tab:blue")
    ax_right.set_ylabel("|A| board/runtime", color="tab:green")
    ax_left.tick_params(axis="y", colors="tab:blue")
    ax_right.tick_params(axis="y", colors="tab:green")
    ax_left.set_title(f"Amplitude Comparison at z = L ({actual_kind}, dual-axis)")
    ax_left.grid(True)
    _combined_legend(ax_left, ax_right)
    _overlay_text(ax_left, _metrics_text(metrics))
    fig.tight_layout()
    save_figure(fig, output_dir, "amplitude_comparison")
    plt.close(fig)


def _plot_constellation(
    expected_complex: np.ndarray,
    actual_complex: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
    actual_kind: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected_complex.real, expected_complex.imag, s=10, alpha=0.45, label="SSFM reference")
    ax.scatter(actual_complex.real, actual_complex.imag, s=10, alpha=0.65, label="Board/runtime")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(f"Constellation Comparison at z = L ({actual_kind})")
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
    actual_kind: str,
) -> None:
    diff_complex = actual_complex - expected_complex
    amp_diff = np.abs(actual_complex) - np.abs(expected_complex)
    abs_complex_diff = np.abs(diff_complex)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_grid, diff_complex.real, label="Real error")
    axes[0].plot(t_grid, diff_complex.imag, label="Imag error", alpha=0.8)
    axes[0].set_ylabel("Channel Error")
    axes[0].set_title(f"Real/Imag Error at z = L ({actual_kind})")
    axes[0].grid(True)
    axes[0].legend()
    _overlay_text(axes[0], _metrics_text(metrics))

    axes[1].plot(t_grid, amp_diff, label="Amplitude error")
    axes[1].plot(t_grid, abs_complex_diff, label="|Complex error|", alpha=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error")
    axes[1].set_title(f"Amplitude and Complex Error at z = L ({actual_kind})")
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
    actual_kind: str,
) -> None:
    fig, ax_left = plt.subplots(figsize=(12, 4))
    ax_right = ax_left.twinx()

    ax_left.plot(
        t_grid,
        A0.real,
        "--",
        color="tab:blue",
        alpha=0.28,
        linewidth=1.3,
        label="Input real(A(0))",
    )
    ax_left.plot(t_grid, expected_real, "-", color="tab:blue", alpha=0.9, label="SSFM reference real(A(L))")
    ax_right.plot(
        t_grid,
        actual_real,
        "--",
        color="tab:green",
        alpha=0.9,
        linewidth=2.4,
        label="Board/runtime real(A(L))",
    )

    ax_left.set_xlabel("Time")
    ax_left.set_ylabel("Real channel input / SSFM", color="tab:blue")
    ax_right.set_ylabel("Real channel board/runtime", color="tab:green")
    ax_left.tick_params(axis="y", colors="tab:blue")
    ax_right.tick_params(axis="y", colors="tab:green")
    ax_left.set_title(f"Real-Channel Comparison at z = L ({actual_kind}, dual-axis)")
    ax_left.grid(True)
    _combined_legend(ax_left, ax_right)
    _overlay_text(ax_left, _metrics_text(metrics))
    fig.tight_layout()
    save_figure(fig, output_dir, "real_channel_comparison")
    plt.close(fig)


def _plot_real_error(
    t_grid: np.ndarray,
    expected_real: np.ndarray,
    actual_real: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
    actual_kind: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_grid, actual_real - expected_real, label="Real-channel error")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.set_title(f"Real-Channel Error at z = L ({actual_kind})")
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


def _run_single_case(
    case: str,
    actual: str,
    atol: float,
    rtol: float,
    diff_out: str | None,
    output_dir: str | None,
) -> tuple[dict[str, float], bool, Path, str, tuple[int, ...]]:
    case_path = Path(case).resolve()
    actual_path = Path(actual).resolve()

    if not case_path.is_file():
        raise FileNotFoundError(f"Verification case not found: {case_path}")
    if not actual_path.is_file():
        raise FileNotFoundError(f"Actual output file not found: {actual_path}")

    expected_arr, A0, t_grid, source = _load_case(case_path)
    actual_arr = _normalize_output(np.load(actual_path), "actual", expected_n_t=expected_arr.shape[-1])
    actual_kind = _describe_actual_kind(actual_path)

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
        _plot_real_channel(t_grid, A0, expected_real, actual_real, figures_dir, metrics, actual_kind)
        _plot_real_error(t_grid, expected_real, actual_real, figures_dir, metrics, actual_kind)
    else:
        expected_complex = expected_arr[0, 0, :] + 1j * expected_arr[0, 1, :]
        actual_complex = actual_arr[0, 0, :] + 1j * actual_arr[0, 1, :]
        _plot_amplitude(t_grid, A0, expected_complex, actual_complex, figures_dir, metrics, actual_kind)
        _plot_constellation(expected_complex, actual_complex, figures_dir, metrics, actual_kind)
        _plot_error(t_grid, expected_complex, actual_complex, figures_dir, metrics, actual_kind)
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

    return metrics, passed, figures_dir, source, tuple(expected_arr.shape)


def _run_batch_cases(
    cases_dir: str,
    actual_name: str,
    atol: float,
    rtol: float,
    output_dir: str | None,
) -> None:
    cases_root = Path(cases_dir).resolve()
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_root}")

    case_dirs = sorted(
        path for path in cases_root.iterdir() if path.is_dir() and path.name.startswith("case_")
    )
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found under: {cases_root}")

    if output_dir is None:
        batch_output_dir = make_figure_output_dir(__file__)
    else:
        batch_output_dir = Path(output_dir).resolve()
        batch_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failed = False

    for case_dir in case_dirs:
        case_path = case_dir / "verification_case.npz"
        actual_path = case_dir / actual_name
        case_output_dir = batch_output_dir / case_dir.name
        case_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            metrics, passed, figures_dir, source, shape = _run_single_case(
                case=str(case_path),
                actual=str(actual_path),
                atol=atol,
                rtol=rtol,
                diff_out=None,
                output_dir=str(case_output_dir),
            )
            rows.append(
                {
                    "case_name": case_dir.name,
                    "status": "ok",
                    "passed": bool(passed),
                    "source": source,
                    "case_path": str(case_path),
                    "actual_path": str(actual_path),
                    "shape": list(shape),
                    "message": "",
                    **metrics,
                }
            )
            if not passed:
                failed = True
        except Exception as exc:
            failed = True
            rows.append(
                {
                    "case_name": case_dir.name,
                    "status": "error",
                    "passed": False,
                    "source": "",
                    "case_path": str(case_path),
                    "actual_path": str(actual_path),
                    "shape": "",
                    "message": str(exc),
                }
            )
            print(f"[WARN] {case_dir.name} failed: {exc}")

    _save_batch_summary(batch_output_dir, rows)
    ok_cases = sum(1 for row in rows if row.get("status") == "ok")
    pass_cases = sum(1 for row in rows if row.get("passed") is True)
    error_cases = sum(1 for row in rows if row.get("status") == "error")
    print("")
    print(f"cases root       : {cases_root}")
    print(f"actual filename  : {actual_name}")
    print(f"batch figures    : {batch_output_dir}")
    print(f"cases discovered : {len(case_dirs)}")
    print(f"cases processed  : {ok_cases}")
    print(f"cases passed     : {pass_cases}")
    print(f"cases failed     : {ok_cases - pass_cases}")
    print(f"cases errored    : {error_cases}")

    if failed:
        raise SystemExit(1)


def main(
    case: str | None,
    actual: str | None,
    cases_dir: str | None,
    actual_name: str,
    atol: float,
    rtol: float,
    diff_out: str | None,
    output_dir: str | None,
) -> None:
    if cases_dir is not None:
        if diff_out is not None:
            raise ValueError("--diff_out is only supported in single-case mode.")
        _run_batch_cases(
            cases_dir=cases_dir,
            actual_name=actual_name,
            atol=atol,
            rtol=rtol,
            output_dir=output_dir,
        )
        return

    if case is None or actual is None:
        raise ValueError("Single-case mode requires both --case and --actual.")

    _, passed, _, _, _ = _run_single_case(
        case=case,
        actual=actual,
        atol=atol,
        rtol=rtol,
        diff_out=diff_out,
        output_dir=output_dir,
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default=None, help="Path to verification_case.npz.")
    ap.add_argument("--actual", type=str, default=None, help="Path to decoded board/runtime output .npy.")
    ap.add_argument(
        "--cases_dir",
        type=str,
        default=None,
        help="Batch mode: directory containing case_000/, case_001/, ... subdirectories.",
    )
    ap.add_argument(
        "--actual_name",
        type=str,
        default="output.npy",
        help="Batch mode: decoded board/runtime filename expected inside each case_xxx directory.",
    )
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
        cases_dir=args.cases_dir,
        actual_name=args.actual_name,
        atol=args.atol,
        rtol=args.rtol,
        diff_out=args.diff_out,
        output_dir=args.output_dir,
    )
