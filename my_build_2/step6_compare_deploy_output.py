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
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def _load_case(case_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, np.ndarray]]:
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
        A0_full = case["A0_real"].astype(np.float32) + 1j * case["A0_imag"].astype(np.float32)
    elif "u_in" in case.files:
        u_in = np.asarray(case["u_in"], dtype=np.float32)
        if u_in.ndim != 2 or u_in.shape[0] != 1 or u_in.shape[1] % 2 != 0:
            raise ValueError(f"Cannot reconstruct A0 from u_in with shape {u_in.shape}")
        half = u_in.shape[1] // 2
        A0_full = u_in[0, :half] + 1j * u_in[0, half:]
    else:
        raise ValueError(
            f"{case_path} does not contain A0 data. Expected `A0_real`/`A0_imag` or `u_in`."
        )

    if "probe_indices" in case.files and "A0_real" in case.files and "A0_imag" in case.files:
        probe_indices = np.asarray(case["probe_indices"], dtype=np.int64).reshape(-1)
        A0_plot = A0_full[probe_indices]
    else:
        A0_plot = A0_full

    if "deploy_t_grid" in case.files:
        t_grid = np.asarray(case["deploy_t_grid"], dtype=np.float32).reshape(-1)
    elif "t_grid" in case.files:
        t_grid = np.asarray(case["t_grid"], dtype=np.float32).reshape(-1)
        if "probe_indices" in case.files:
            t_grid = t_grid[np.asarray(case["probe_indices"], dtype=np.int64).reshape(-1)]
    else:
        t_grid = np.arange(expected.shape[-1], dtype=np.float32)

    source = _case_string(case, "source", "unknown_source")
    aux = {
        "A0_full": np.asarray(A0_full).astype(np.complex64),
        "deploy_t_grid": np.asarray(t_grid, dtype=np.float32),
    }
    return expected.astype(np.float32), A0_plot.astype(np.complex64), t_grid, source, aux


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


def _metrics_and_timing_text(
    metrics: dict[str, float],
    timing: dict[str, float | int] | None = None,
) -> str:
    text = _metrics_text(metrics)
    if timing is None:
        return text
    return (
        text
        + "\n"
        + f"PyTorch mean={float(timing['mean_ms']):.3f} ms | "
        + f"std={float(timing['std_ms']):.3f} ms | "
        + f"n={int(timing['measure_runs'])}"
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


def _save_json(output_dir: Path, payload: dict[str, object], filename: str) -> Path:
    path = output_dir / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Saved {filename}: {path}")
    return path


def _save_metrics(output_dir: Path, metrics: dict[str, float], filename: str = "metrics.json") -> Path:
    return _save_json(output_dir, metrics, filename)


def _find_default_pytorch_ckpt(script_dir: Path) -> Path:
    candidates = [
        script_dir / "hybrid_pinn_deeponet.pth",
        script_dir.parent / "hybrid_pinn_deeponet.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_pytorch_runtime(checkpoint_path: Path) -> dict[str, object]:
    from pinn_physics_model import DeepONet, L, MODEL_PATH, N_t, P_dim, device, fourier_dim, make_branch_input

    ckpt_path = checkpoint_path.resolve() if checkpoint_path is not None else Path(MODEL_PATH).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {ckpt_path}")

    model = DeepONet(N_t, P_dim, fourier_dim).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "device": device,
        "make_branch_input": make_branch_input,
        "L": float(L),
        "N_t": int(N_t),
        "checkpoint_path": str(ckpt_path),
    }


def _sync_pytorch_device(runtime: dict[str, object]) -> None:
    runtime_device = runtime["device"]
    if torch.cuda.is_available() and getattr(runtime_device, "type", "") == "cuda":
        torch.cuda.synchronize(runtime_device)


@torch.no_grad()
def _predict_pytorch_probe(
    runtime: dict[str, object],
    A0_full: np.ndarray,
    deploy_t_grid: np.ndarray,
) -> np.ndarray:
    runtime_device = runtime["device"]
    n_t = int(runtime["N_t"])
    if int(np.asarray(A0_full).size) != n_t:
        raise ValueError(
            f"PyTorch comparison expects full A0 length {n_t}, but got {np.asarray(A0_full).shape}"
        )

    A0_tensor = torch.from_numpy(np.asarray(A0_full).astype(np.complex64)).to(runtime_device)
    t_probe = torch.from_numpy(np.asarray(deploy_t_grid, dtype=np.float32)).to(runtime_device).view(-1, 1)
    z_probe = torch.full_like(t_probe, float(runtime["L"]))

    make_branch_input = runtime["make_branch_input"]
    u_in = make_branch_input(A0_tensor.unsqueeze(0)).repeat(t_probe.shape[0], 1)
    model = runtime["model"]
    u_pred, v_pred = model(u_in, t_probe, z_probe)
    pred_complex = (u_pred + 1j * v_pred).detach().cpu().numpy().reshape(-1).astype(np.complex64)
    return np.concatenate([pred_complex.real, pred_complex.imag], axis=0).reshape(1, -1).astype(np.float32)


def _benchmark_pytorch_probe(
    runtime: dict[str, object],
    A0_full: np.ndarray,
    deploy_t_grid: np.ndarray,
    warmup_runs: int,
    measure_runs: int,
) -> dict[str, float | int]:
    warmup_runs = max(int(warmup_runs), 0)
    measure_runs = max(int(measure_runs), 1)

    for _ in range(warmup_runs):
        _ = _predict_pytorch_probe(runtime, A0_full, deploy_t_grid)

    _sync_pytorch_device(runtime)
    timings_ms: list[float] = []
    for _ in range(measure_runs):
        _sync_pytorch_device(runtime)
        t0 = time.perf_counter()
        _ = _predict_pytorch_probe(runtime, A0_full, deploy_t_grid)
        _sync_pytorch_device(runtime)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1e3)

    timings_arr = np.asarray(timings_ms, dtype=np.float64)
    return {
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "mean_ms": float(timings_arr.mean()),
        "std_ms": float(timings_arr.std()),
        "min_ms": float(timings_arr.min()),
        "max_ms": float(timings_arr.max()),
    }


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
        "pytorch_rmse",
        "pytorch_relative_rmse",
        "pytorch_complex_rmse",
        "pytorch_amplitude_rmse",
        "pytorch_max_abs_diff",
        "pytorch_mean_abs_diff",
        "pytorch_timing_mean_ms",
        "pytorch_timing_std_ms",
        "pytorch_timing_min_ms",
        "pytorch_timing_max_ms",
        "pytorch_timing_measure_runs",
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


def _default_output_dir(actual_kind: str) -> Path:
    if actual_kind == "dequantized":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "figures" / "dequantized" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    return make_figure_output_dir(__file__)


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


def _plot_pytorch_amplitude(
    t_grid: np.ndarray,
    A0: np.ndarray,
    expected_complex: np.ndarray,
    pytorch_complex: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
    timing: dict[str, float | int],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_grid, np.abs(A0), ":", alpha=0.35, linewidth=1.2, label="Input |A(0)|")
    ax.plot(t_grid, np.abs(expected_complex), "-", alpha=0.9, label="SSFM reference |A(L)|")
    ax.plot(t_grid, np.abs(pytorch_complex), "--", alpha=0.95, linewidth=2.0, label="PyTorch pred |A(L)|")
    ax.set_xlabel("Time")
    ax.set_ylabel("|A|")
    ax.set_title("PyTorch Prediction vs SSFM at z = L")
    ax.grid(True)
    ax.legend()
    _overlay_text(ax, _metrics_and_timing_text(metrics, timing))
    fig.tight_layout()
    save_figure(fig, output_dir, "pytorch_amplitude_comparison")
    plt.close(fig)


def _plot_pytorch_constellation(
    expected_complex: np.ndarray,
    pytorch_complex: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
    timing: dict[str, float | int],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected_complex.real, expected_complex.imag, s=10, alpha=0.45, label="SSFM reference")
    ax.scatter(pytorch_complex.real, pytorch_complex.imag, s=10, alpha=0.65, label="PyTorch pred")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title("PyTorch Constellation vs SSFM at z = L")
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal", "box")
    _overlay_text(ax, _metrics_and_timing_text(metrics, timing))
    fig.tight_layout()
    save_figure(fig, output_dir, "pytorch_constellation_comparison")
    plt.close(fig)


def _plot_pytorch_error(
    t_grid: np.ndarray,
    expected_complex: np.ndarray,
    pytorch_complex: np.ndarray,
    output_dir: Path,
    metrics: dict[str, float],
    timing: dict[str, float | int],
) -> None:
    diff_complex = pytorch_complex - expected_complex
    amp_diff = np.abs(pytorch_complex) - np.abs(expected_complex)
    abs_complex_diff = np.abs(diff_complex)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t_grid, diff_complex.real, label="Real error")
    axes[0].plot(t_grid, diff_complex.imag, label="Imag error", alpha=0.8)
    axes[0].set_ylabel("Channel Error")
    axes[0].set_title("PyTorch Real/Imag Error at z = L")
    axes[0].grid(True)
    axes[0].legend()
    _overlay_text(axes[0], _metrics_and_timing_text(metrics, timing))

    axes[1].plot(t_grid, amp_diff, label="Amplitude error")
    axes[1].plot(t_grid, abs_complex_diff, label="|Complex error|", alpha=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error")
    axes[1].set_title("PyTorch Amplitude and Complex Error at z = L")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, output_dir, "pytorch_error_summary")
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
    pytorch_runtime: dict[str, object] | None = None,
    pytorch_warmup_runs: int = 5,
    pytorch_measure_runs: int = 30,
) -> tuple[
    dict[str, float],
    bool,
    Path,
    str,
    tuple[int, ...],
    dict[str, float] | None,
    dict[str, float | int] | None,
]:
    case_path = Path(case).resolve()
    actual_path = Path(actual).resolve()

    if not case_path.is_file():
        raise FileNotFoundError(f"Verification case not found: {case_path}")
    if not actual_path.is_file():
        raise FileNotFoundError(f"Actual output file not found: {actual_path}")

    expected_arr, A0, t_grid, source, case_aux = _load_case(case_path)
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
        figures_dir = _default_output_dir(actual_kind)
    else:
        figures_dir = Path(output_dir).resolve()
        figures_dir.mkdir(parents=True, exist_ok=True)

    pytorch_metrics: dict[str, float] | None = None
    pytorch_timing: dict[str, float | int] | None = None

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

    if pytorch_runtime is not None:
        pytorch_raw = _predict_pytorch_probe(
            pytorch_runtime,
            case_aux["A0_full"],
            case_aux["deploy_t_grid"],
        )
        pytorch_arr = _normalize_output(pytorch_raw, "pytorch_pred", expected_n_t=expected_arr.shape[-1])
        pytorch_metrics = _compute_metrics(expected_arr, pytorch_arr)
        pytorch_metrics["atol"] = float(atol)
        pytorch_metrics["rtol"] = float(rtol)
        pytorch_timing = _benchmark_pytorch_probe(
            pytorch_runtime,
            case_aux["A0_full"],
            case_aux["deploy_t_grid"],
            warmup_runs=pytorch_warmup_runs,
            measure_runs=pytorch_measure_runs,
        )

        if expected_arr.shape[1] == 2:
            expected_complex = expected_arr[0, 0, :] + 1j * expected_arr[0, 1, :]
            pytorch_complex = pytorch_arr[0, 0, :] + 1j * pytorch_arr[0, 1, :]
            _plot_pytorch_amplitude(
                t_grid,
                A0,
                expected_complex,
                pytorch_complex,
                figures_dir,
                pytorch_metrics,
                pytorch_timing,
            )
            _plot_pytorch_constellation(
                expected_complex,
                pytorch_complex,
                figures_dir,
                pytorch_metrics,
                pytorch_timing,
            )
            _plot_pytorch_error(
                t_grid,
                expected_complex,
                pytorch_complex,
                figures_dir,
                pytorch_metrics,
                pytorch_timing,
            )
        _save_metrics(figures_dir, pytorch_metrics, filename="pytorch_metrics.json")
        _save_json(figures_dir, pytorch_timing, "pytorch_inference_timing.json")

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
    if pytorch_metrics is not None and pytorch_timing is not None:
        print("")
        print("PyTorch reference:")
        _print_metric("pytorch_rmse", pytorch_metrics["rmse"])
        _print_metric("pytorch_relative_rmse", pytorch_metrics["relative_rmse"])
        if expected_arr.shape[1] == 2:
            _print_metric("pytorch_complex_rmse", pytorch_metrics["complex_rmse"])
            _print_metric("pytorch_amplitude_rmse", pytorch_metrics["amplitude_rmse"])
        _print_metric("pytorch_mean_ms", float(pytorch_timing["mean_ms"]))
        _print_metric("pytorch_std_ms", float(pytorch_timing["std_ms"]))

    return metrics, passed, figures_dir, source, tuple(expected_arr.shape), pytorch_metrics, pytorch_timing


def _run_batch_cases(
    cases_dir: str,
    actual_name: str,
    atol: float,
    rtol: float,
    output_dir: str | None,
    pytorch_runtime: dict[str, object] | None = None,
    pytorch_warmup_runs: int = 5,
    pytorch_measure_runs: int = 30,
) -> None:
    cases_root = Path(cases_dir).resolve()
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_root}")

    case_dirs = sorted(
        path for path in cases_root.iterdir() if path.is_dir() and path.name.startswith("case_")
    )
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found under: {cases_root}")

    inferred_kind = _describe_actual_kind(Path(actual_name))
    if output_dir is None:
        batch_output_dir = _default_output_dir(inferred_kind)
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
            metrics, passed, figures_dir, source, shape, pytorch_metrics, pytorch_timing = _run_single_case(
                case=str(case_path),
                actual=str(actual_path),
                atol=atol,
                rtol=rtol,
                diff_out=None,
                output_dir=str(case_output_dir),
                pytorch_runtime=pytorch_runtime,
                pytorch_warmup_runs=pytorch_warmup_runs,
                pytorch_measure_runs=pytorch_measure_runs,
            )
            extra = {}
            if pytorch_metrics is not None:
                extra.update(
                    {
                        "pytorch_rmse": pytorch_metrics.get("rmse", ""),
                        "pytorch_relative_rmse": pytorch_metrics.get("relative_rmse", ""),
                        "pytorch_complex_rmse": pytorch_metrics.get("complex_rmse", ""),
                        "pytorch_amplitude_rmse": pytorch_metrics.get("amplitude_rmse", ""),
                        "pytorch_max_abs_diff": pytorch_metrics.get("max_abs_diff", ""),
                        "pytorch_mean_abs_diff": pytorch_metrics.get("mean_abs_diff", ""),
                    }
                )
            if pytorch_timing is not None:
                extra.update(
                    {
                        "pytorch_timing_mean_ms": pytorch_timing.get("mean_ms", ""),
                        "pytorch_timing_std_ms": pytorch_timing.get("std_ms", ""),
                        "pytorch_timing_min_ms": pytorch_timing.get("min_ms", ""),
                        "pytorch_timing_max_ms": pytorch_timing.get("max_ms", ""),
                        "pytorch_timing_measure_runs": pytorch_timing.get("measure_runs", ""),
                    }
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
                    **extra,
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
    include_pytorch: bool,
    pytorch_ckpt: str | None,
    pytorch_warmup_runs: int,
    pytorch_measure_runs: int,
) -> None:
    pytorch_runtime = None
    if include_pytorch:
        ckpt_path = Path(pytorch_ckpt).resolve() if pytorch_ckpt is not None else _find_default_pytorch_ckpt(Path(__file__).resolve().parent)
        pytorch_runtime = _load_pytorch_runtime(ckpt_path)

    if cases_dir is not None:
        if diff_out is not None:
            raise ValueError("--diff_out is only supported in single-case mode.")
        _run_batch_cases(
            cases_dir=cases_dir,
            actual_name=actual_name,
            atol=atol,
            rtol=rtol,
            output_dir=output_dir,
            pytorch_runtime=pytorch_runtime,
            pytorch_warmup_runs=pytorch_warmup_runs,
            pytorch_measure_runs=pytorch_measure_runs,
        )
        return

    if case is None or actual is None:
        raise ValueError("Single-case mode requires both --case and --actual.")

    _, passed, _, _, _, _, _ = _run_single_case(
        case=case,
        actual=actual,
        atol=atol,
        rtol=rtol,
        diff_out=diff_out,
        output_dir=output_dir,
        pytorch_runtime=pytorch_runtime,
        pytorch_warmup_runs=pytorch_warmup_runs,
        pytorch_measure_runs=pytorch_measure_runs,
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
    ap.add_argument(
        "--include_pytorch",
        action="store_true",
        help="Also run the original PyTorch model on the same case(s), save PyTorch comparison figures, and benchmark prediction time.",
    )
    ap.add_argument(
        "--pytorch_ckpt",
        type=str,
        default=None,
        help="Optional PyTorch checkpoint path. If omitted, step6 searches for hybrid_pinn_deeponet.pth near the script.",
    )
    ap.add_argument("--pytorch_warmup_runs", type=int, default=5)
    ap.add_argument("--pytorch_measure_runs", type=int, default=30)
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
        include_pytorch=args.include_pytorch,
        pytorch_ckpt=args.pytorch_ckpt,
        pytorch_warmup_runs=args.pytorch_warmup_runs,
        pytorch_measure_runs=args.pytorch_measure_runs,
    )
