"""
Probe what a current board-side `(1, 256)` output most likely represents.

This script is for the current single-output deploy interface where the generated
driver exposes:
  - oshape_normal = (1, 256)
  - odt           = INT24

It does NOT claim full end-to-end model correctness. Instead, it tries to answer:
  - Is the scalar trace most consistent with SSFM real, SSFM imag, |A|, |A|^2, or
    even the input waveform itself?
  - If yes, how good is the affine fit from raw INT24 values to that candidate?

Typical usage:
    python step6_probe_single_output_semantics.py ^
        --case verification_io/verification_case.npz ^
        --actual output_1x256_raw.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_output_utils import make_figure_output_dir, save_figure


def _case_string(npz: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz.files:
        return default
    value = np.asarray(npz[key])
    if value.size == 0:
        return default
    return str(value.reshape(-1)[0])


def _load_case(case_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    case = np.load(case_path)

    if "ssfm_output" not in case.files:
        raise ValueError(f"{case_path} does not contain `ssfm_output`.")
    if "A0_real" not in case.files or "A0_imag" not in case.files:
        raise ValueError(f"{case_path} does not contain `A0_real` and `A0_imag`.")

    ssfm_output = np.asarray(case["ssfm_output"], dtype=np.float32)
    if ssfm_output.shape != (1, 2, 256):
        raise ValueError(
            f"Expected ssfm_output shape (1, 2, 256) in {case_path}, got {ssfm_output.shape}"
        )

    A0 = np.asarray(case["A0_real"], dtype=np.float32) + 1j * np.asarray(case["A0_imag"], dtype=np.float32)

    if "t_grid" in case.files:
        t_grid = np.asarray(case["t_grid"], dtype=np.float32).reshape(-1)
    else:
        t_grid = np.arange(ssfm_output.shape[-1], dtype=np.float32)

    source = _case_string(case, "source", "unknown_source")
    return ssfm_output, A0.astype(np.complex64), t_grid, source


def _choose_channel(two_channel: np.ndarray, channel: str, name: str) -> tuple[np.ndarray, str]:
    ch0 = np.asarray(two_channel[0], dtype=np.float64)
    ch1 = np.asarray(two_channel[1], dtype=np.float64)
    nz0 = int(np.count_nonzero(ch0))
    nz1 = int(np.count_nonzero(ch1))

    if channel == "0":
        return ch0, f"{name}: forced channel 0"
    if channel == "1":
        return ch1, f"{name}: forced channel 1"

    if nz0 > 0 and nz1 == 0:
        return ch0, f"{name}: auto-selected channel 0 because channel 1 is all zero"
    if nz1 > 0 and nz0 == 0:
        return ch1, f"{name}: auto-selected channel 1 because channel 0 is all zero"
    if nz0 > 0 and nz1 > 0:
        raise ValueError(
            f"{name} contains two nonzero channels, so it is not a single-output trace. "
            "Use the full `step6_compare_deploy_output.py` path instead."
        )

    return ch0, f"{name}: both channels are zero; defaulted to channel 0"


def _extract_single_trace(arr: np.ndarray, channel: str, expected_n: int) -> tuple[np.ndarray, str]:
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        raise ValueError("Single-output probing expects real/integer arrays, not complex arrays.")

    if arr.ndim == 1:
        if arr.shape[0] != expected_n:
            raise ValueError(f"Expected a length-{expected_n} vector, got {arr.shape}")
        return arr.astype(np.float64), "actual: 1D trace"

    if arr.ndim == 2:
        if arr.shape == (1, expected_n):
            return arr[0].astype(np.float64), "actual: squeezed from (1, N)"
        if arr.shape == (expected_n, 1):
            return arr[:, 0].astype(np.float64), "actual: squeezed from (N, 1)"
        if arr.shape == (2, expected_n):
            return _choose_channel(arr, channel, "actual")
        if arr.shape == (expected_n, 2):
            return _choose_channel(arr.T, channel, "actual")

    if arr.ndim == 3:
        if arr.shape == (1, 2, expected_n):
            return _choose_channel(arr[0], channel, "actual")
        if arr.shape == (1, expected_n, 2):
            return _choose_channel(np.transpose(arr[0], (1, 0)), channel, "actual")

    raise ValueError(
        f"Unsupported actual array shape {arr.shape}. Expected (256,), (1,256), (256,1), "
        "(2,256), (256,2), (1,2,256), or (1,256,2)."
    )


def _candidate_signals(ssfm_output: np.ndarray, A0: np.ndarray) -> dict[str, np.ndarray]:
    ssfm_complex = ssfm_output[0, 0, :].astype(np.float64) + 1j * ssfm_output[0, 1, :].astype(np.float64)
    return {
        "ssfm_real": ssfm_complex.real,
        "ssfm_imag": ssfm_complex.imag,
        "ssfm_abs": np.abs(ssfm_complex),
        "ssfm_power": np.abs(ssfm_complex) ** 2,
        "input_real": A0.real.astype(np.float64),
        "input_imag": A0.imag.astype(np.float64),
        "input_abs": np.abs(A0).astype(np.float64),
        "input_power": (np.abs(A0) ** 2).astype(np.float64),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("Correlation inputs must have the same length.")
    if np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _fit_affine(raw: np.ndarray, target: np.ndarray) -> dict[str, float | np.ndarray | str]:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    design = np.column_stack([raw, np.ones_like(raw)])
    scale, bias = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = scale * raw + bias
    err = fitted - target

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    target_rms = float(np.sqrt(np.mean(target**2)))
    rel_rmse = float(rmse / (target_rms + 1e-12))

    return {
        "scale": float(scale),
        "bias": float(bias),
        "rmse": rmse,
        "mae": mae,
        "relative_rmse": rel_rmse,
        "corr_raw_target": _safe_corr(raw, target),
        "corr_fitted_target": _safe_corr(fitted, target),
        "target_rms": target_rms,
        "fitted": fitted,
        "residual": err,
    }


def _result_row(name: str, target: np.ndarray, fit: dict[str, float | np.ndarray | str]) -> dict[str, float | str]:
    target = np.asarray(target, dtype=np.float64)
    return {
        "candidate": name,
        "scale": float(fit["scale"]),
        "bias": float(fit["bias"]),
        "rmse": float(fit["rmse"]),
        "mae": float(fit["mae"]),
        "relative_rmse": float(fit["relative_rmse"]),
        "corr_raw_target": float(fit["corr_raw_target"]),
        "corr_fitted_target": float(fit["corr_fitted_target"]),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
    }


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "candidate",
        "scale",
        "bias",
        "rmse",
        "mae",
        "relative_rmse",
        "corr_raw_target",
        "corr_fitted_target",
        "target_min",
        "target_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_conclusion(rows: list[dict[str, float | str]]) -> tuple[str, str]:
    best = rows[0]
    second = rows[1] if len(rows) > 1 else None

    best_rel = float(best["relative_rmse"])
    best_corr = float(best["corr_fitted_target"])
    gap = None if second is None else float(second["relative_rmse"]) - best_rel

    if best_corr >= 0.995 and best_rel <= 0.10 and (gap is None or gap >= 0.03):
        confidence = "strong"
    elif best_corr >= 0.97 and best_rel <= 0.20:
        confidence = "tentative"
    else:
        confidence = "weak"

    candidate = str(best["candidate"])
    if confidence == "strong":
        text = (
            f"The current `(1, 256)` board output is strongly consistent with `{candidate}` "
            f"after affine remapping (scale={float(best['scale']):.6g}, bias={float(best['bias']):.6g}). "
            "This is enough to treat the current bitfile as a single-scalar trace interface, "
            "but it is still not evidence of full complex-output correctness."
        )
    elif confidence == "tentative":
        text = (
            f"The best match is `{candidate}`, but the evidence is only tentative "
            f"(corr={best_corr:.4f}, rel_rmse={best_rel:.4f}). You may use it as a debugging clue, "
            "not as a final deployment conclusion."
        )
    else:
        text = (
            f"No candidate cleanly explains the current `(1, 256)` output. The best match is `{candidate}`, "
            f"but corr={best_corr:.4f} and rel_rmse={best_rel:.4f} are too weak for a reliable conclusion."
        )

    return confidence, text


def _plot_top_candidates(
    t_grid: np.ndarray,
    raw_trace: np.ndarray,
    candidates: dict[str, np.ndarray],
    fits: dict[str, dict[str, float | np.ndarray | str]],
    ranked_rows: list[dict[str, float | str]],
    output_dir: Path,
) -> None:
    top_rows = ranked_rows[:4]
    fig, axes = plt.subplots(len(top_rows), 1, figsize=(12, 3 * len(top_rows)), sharex=True)
    if len(top_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, top_rows):
        name = str(row["candidate"])
        target = candidates[name]
        fitted = np.asarray(fits[name]["fitted"], dtype=np.float64)
        ax.plot(t_grid, target, label=f"{name} target")
        ax.plot(t_grid, fitted, "--", label="Affine-mapped board trace")
        ax.set_ylabel("Value")
        ax.set_title(
            f"{name} | corr={float(row['corr_fitted_target']):.4f} "
            f"| rel_rmse={float(row['relative_rmse']):.4f}"
        )
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    save_figure(fig, output_dir, "top_candidate_overlays")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t_grid, raw_trace, label="Raw board trace")
    ax.set_title("Raw Single-Output Board Trace")
    ax.set_xlabel("Time")
    ax.set_ylabel("INT24 value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "raw_board_trace")
    plt.close(fig)


def _plot_best_candidate(
    t_grid: np.ndarray,
    raw_trace: np.ndarray,
    candidate_name: str,
    target: np.ndarray,
    fit: dict[str, float | np.ndarray | str],
    output_dir: Path,
) -> None:
    fitted = np.asarray(fit["fitted"], dtype=np.float64)
    residual = np.asarray(fit["residual"], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    axes[0].scatter(raw_trace, target, s=14, alpha=0.6, label="Samples")
    order = np.argsort(raw_trace)
    axes[0].plot(raw_trace[order], fitted[order], color="tab:red", label="Affine fit")
    axes[0].set_title(
        f"Best Candidate: {candidate_name} | scale={float(fit['scale']):.6g}, "
        f"bias={float(fit['bias']):.6g}"
    )
    axes[0].set_xlabel("Raw board trace")
    axes[0].set_ylabel("Candidate target")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_grid, residual, label="Residual")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Fit error")
    axes[1].set_title("Best Candidate Residual Over Time")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    save_figure(fig, output_dir, "best_candidate_fit")
    plt.close(fig)


def main(case: str, actual: str, output_dir: str | None, channel: str) -> None:
    case_path = Path(case).resolve()
    actual_path = Path(actual).resolve()

    if not case_path.is_file():
        raise FileNotFoundError(f"Verification case not found: {case_path}")
    if not actual_path.is_file():
        raise FileNotFoundError(f"Actual output file not found: {actual_path}")

    ssfm_output, A0, t_grid, source = _load_case(case_path)
    actual_arr = np.load(actual_path)
    raw_trace, trace_desc = _extract_single_trace(actual_arr, channel=channel, expected_n=ssfm_output.shape[-1])
    candidates = _candidate_signals(ssfm_output, A0)

    fits: dict[str, dict[str, float | np.ndarray | str]] = {}
    rows: list[dict[str, float | str]] = []
    for name, target in candidates.items():
        fit = _fit_affine(raw_trace, target)
        fits[name] = fit
        rows.append(_result_row(name, target, fit))

    rows.sort(key=lambda row: (float(row["relative_rmse"]), -float(row["corr_fitted_target"])))
    confidence, conclusion = _build_conclusion(rows)

    if output_dir is None:
        figures_dir = make_figure_output_dir(__file__)
    else:
        figures_dir = Path(output_dir).resolve()
        figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_top_candidates(t_grid, raw_trace, candidates, fits, rows, figures_dir)
    best_name = str(rows[0]["candidate"])
    _plot_best_candidate(t_grid, raw_trace, best_name, candidates[best_name], fits[best_name], figures_dir)

    csv_path = figures_dir / "candidate_ranking.csv"
    json_path = figures_dir / "probe_summary.json"
    _write_csv(csv_path, rows)

    summary = {
        "case": str(case_path),
        "actual": str(actual_path),
        "source": source,
        "trace_description": trace_desc,
        "board_interface_meaning": "(1, 256) means one batch of 256 scalar outputs, not a full complex tensor.",
        "confidence": confidence,
        "conclusion": conclusion,
        "best_candidate": rows[0],
        "ranking": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"verification case        : {case_path}")
    print(f"actual output            : {actual_path}")
    print(f"source                   : {source}")
    print(f"trace interpretation     : {trace_desc}")
    print("board interface meaning  : (1, 256) = batch size 1 with 256 scalar outputs")
    print("")
    for idx, row in enumerate(rows[:5], start=1):
        print(
            f"{idx}. {row['candidate']:<12} "
            f"corr={float(row['corr_fitted_target']):.6f} "
            f"rel_rmse={float(row['relative_rmse']):.6e} "
            f"scale={float(row['scale']):.6g} "
            f"bias={float(row['bias']):.6g}"
        )
    print("")
    print(f"confidence               : {confidence}")
    print(f"conclusion               : {conclusion}")
    print(f"saved ranking csv        : {csv_path}")
    print(f"saved summary json       : {json_path}")
    print(f"saved figures            : {figures_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, required=True, help="Path to verification_case.npz.")
    ap.add_argument(
        "--actual",
        type=str,
        required=True,
        help="Path to a current single-output board npy, e.g. (256,), (1,256), or a zero-padded (1,2,256).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory for saved plots and probe summary files.",
    )
    ap.add_argument(
        "--channel",
        type=str,
        choices=["auto", "0", "1"],
        default="auto",
        help="How to select a channel if the actual file still has a redundant 2-channel wrapper.",
    )
    args = ap.parse_args()
    main(
        case=args.case,
        actual=args.actual,
        output_dir=args.output_dir,
        channel=args.channel,
    )
