"""
Step 5: Generate verification input/output tensors for FINN or board-side checking.

Typical usage:
    python step5_generate_verification_io.py \
        --model deeponet_u250_int8_qonnx_ready.onnx \
        --out_dir verification_io

This script:
  1) Builds one deploy input waveform from the existing project data path.
  2) Converts it into the deploy-model branch input `u_in`.
  3) Executes the exported QONNX / FINN-ready model with qonnx.
  4) Saves:
       - input.npy
       - expected_output.npy
       - verification_case.npz

Run this inside an environment that has qonnx installed, for example `.venv_finn`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import pinn_physics_model as pm
from finn_qonnx_utils import prepare_qonnx_for_finn

try:
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
except ImportError as exc:  # pragma: no cover - handled with a clear runtime error
    raise RuntimeError(
        "qonnx is required for step5_generate_verification_io.py. "
        "Run this script inside the FINN / .venv_finn environment."
    ) from exc


def _select_sample(source: str, sample_index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}")

    if source == "random":
        A0, _, _ = pm.generate_qam_waveform()
        AL_clean = pm.ssfm_propagate(A0, pm.L)
        return A0.detach().cpu(), AL_clean.detach().cpu(), "random_ssfm"

    if source == "train":
        cached = pm.load_dataset_cache(pm.N_train, pm.snr_db_train, cache_key=pm.TRAIN_DATASET_CACHE_KEY)
        if cached is not None:
            A0_all, AL_clean_all, _ = cached
            if sample_index >= A0_all.shape[0]:
                raise IndexError(
                    f"sample_index={sample_index} exceeds cached train dataset size {A0_all.shape[0]}"
                )
            return (
                A0_all[sample_index].detach().cpu(),
                AL_clean_all[sample_index].detach().cpu(),
                f"train_cache[{sample_index}]",
            )

        n_samples = sample_index + 1
        A0_all, AL_clean_all, _ = pm.build_dataset(
            n_samples=n_samples,
            snr_db=pm.snr_db_train,
            cache_key="verify_train_subset",
        )
        return (
            A0_all[sample_index].detach().cpu(),
            AL_clean_all[sample_index].detach().cpu(),
            f"train_subset[{sample_index}]",
        )

    if source == "test":
        cached = pm.load_dataset_cache(pm.N_test, pm.snr_db_eval, cache_key=pm.TEST_DATASET_CACHE_KEY)
        if cached is not None:
            A0_all, AL_clean_all, _ = cached
            if sample_index >= A0_all.shape[0]:
                raise IndexError(
                    f"sample_index={sample_index} exceeds cached test dataset size {A0_all.shape[0]}"
                )
            return (
                A0_all[sample_index].detach().cpu(),
                AL_clean_all[sample_index].detach().cpu(),
                f"test_cache[{sample_index}]",
            )

        n_samples = sample_index + 1
        A0_all, AL_clean_all, _ = pm.build_dataset(
            n_samples=n_samples,
            snr_db=pm.snr_db_eval,
            cache_key="verify_test_subset",
        )
        return (
            A0_all[sample_index].detach().cpu(),
            AL_clean_all[sample_index].detach().cpu(),
            f"test_subset[{sample_index}]",
        )

    raise ValueError(f"Unsupported source: {source}")


def _get_model_io_names(model: ModelWrapper) -> tuple[str, str]:
    initializer_names = {init.name for init in model.graph.initializer}
    graph_inputs = [value.name for value in model.graph.input if value.name not in initializer_names]
    input_name = graph_inputs[0] if graph_inputs else model.graph.input[0].name
    output_name = model.graph.output[0].name
    return input_name, output_name


def _load_executable_model(model_path: Path, prepared_model_path: Path | None) -> tuple[ModelWrapper, Path]:
    model = ModelWrapper(str(model_path))

    # Raw QONNX exports may still need shape/datatype cleanup before qonnx execution.
    try:
        _get_model_io_names(model)
        return model, model_path
    except Exception:
        pass

    if prepared_model_path is None:
        prepared_model_path = model_path.with_name(model_path.stem + "_ready.onnx")

    prepared = prepare_qonnx_for_finn(
        in_path=str(model_path),
        out_path=str(prepared_model_path),
        verbose=False,
    )
    return prepared, prepared_model_path


def _execute_model(model_path: Path, u_in: np.ndarray, prepared_model_path: Path | None) -> tuple[np.ndarray, str, str, Path]:
    model, resolved_model_path = _load_executable_model(model_path, prepared_model_path)
    input_name, output_name = _get_model_io_names(model)

    try:
        outputs = execute_onnx(model, {input_name: u_in})
    except Exception as exc:
        message = str(exc)
        if "infer_shapes" not in message:
            raise
        if prepared_model_path is None:
            prepared_model_path = model_path.with_name(model_path.stem + "_ready.onnx")
        model = prepare_qonnx_for_finn(
            in_path=str(model_path),
            out_path=str(prepared_model_path),
            verbose=False,
        )
        resolved_model_path = prepared_model_path
        input_name, output_name = _get_model_io_names(model)
        outputs = execute_onnx(model, {input_name: u_in})

    y = outputs[output_name]
    return np.asarray(y, dtype=np.float32), input_name, output_name, resolved_model_path


def _to_numpy_complex(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def main(
    model: str,
    out_dir: str,
    source: str,
    sample_index: int,
    prepared_model_out: str | None,
) -> None:
    model_path = Path(model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    prepared_model_path = Path(prepared_model_out).resolve() if prepared_model_out else None

    A0, AL_clean, source_desc = _select_sample(source, sample_index)
    with torch.no_grad():
        u_in_t = pm.make_branch_input(A0.unsqueeze(0)).float()
    u_in = u_in_t.detach().cpu().numpy().astype(np.float32)

    expected_output, input_name, output_name, resolved_model_path = _execute_model(
        model_path=model_path,
        u_in=u_in,
        prepared_model_path=prepared_model_path,
    )

    if expected_output.ndim != 3 or expected_output.shape[1] != 2:
        raise ValueError(
            "Expected deploy output shape [B, 2, N_t], "
            f"but got {expected_output.shape} from {resolved_model_path}"
        )

    ref_clean = np.stack(
        [
            _to_numpy_complex(AL_clean).real.astype(np.float32),
            _to_numpy_complex(AL_clean).imag.astype(np.float32),
        ],
        axis=0,
    )[None, :, :]

    deploy_rmse_vs_clean = float(np.sqrt(np.mean((expected_output - ref_clean) ** 2)))
    deploy_rel_rmse_vs_clean = float(
        deploy_rmse_vs_clean / (np.sqrt(np.mean(ref_clean**2)) + 1e-12)
    )

    np.save(out_dir_path / "input.npy", u_in.astype(np.float32))
    np.save(out_dir_path / "expected_output.npy", expected_output.astype(np.float32))
    np.savez(
        out_dir_path / "verification_case.npz",
        model_path=np.array([str(resolved_model_path)]),
        source=np.array([source_desc]),
        model_input_name=np.array([input_name]),
        model_output_name=np.array([output_name]),
        u_in=u_in.astype(np.float32),
        expected_output=expected_output.astype(np.float32),
        A0_real=_to_numpy_complex(A0).real.astype(np.float32),
        A0_imag=_to_numpy_complex(A0).imag.astype(np.float32),
        AL_clean_real=_to_numpy_complex(AL_clean).real.astype(np.float32),
        AL_clean_imag=_to_numpy_complex(AL_clean).imag.astype(np.float32),
    )

    print(f"[OK] Generated verification tensors in: {out_dir_path}")
    print(f"     model used: {resolved_model_path}")
    print(f"     source    : {source_desc}")
    print(f"     input     : {input_name} shape={tuple(u_in.shape)} dtype={u_in.dtype}")
    print(f"     output    : {output_name} shape={tuple(expected_output.shape)} dtype={expected_output.dtype}")
    print(f"     deploy vs SSFM clean RMSE      = {deploy_rmse_vs_clean:.6e}")
    print(f"     deploy vs SSFM clean rel. RMSE = {deploy_rel_rmse_vs_clean:.6e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="deeponet_u250_int8_qonnx_ready.onnx")
    ap.add_argument("--out_dir", type=str, default="verification_io")
    ap.add_argument("--source", type=str, choices=["train", "test", "random"], default="test")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument(
        "--prepared_model_out",
        type=str,
        default=None,
        help="Optional path for an auto-generated _ready model when the input model needs cleanup first.",
    )
    args = ap.parse_args()
    main(
        model=args.model,
        out_dir=args.out_dir,
        source=args.source,
        sample_index=args.sample_index,
        prepared_model_out=args.prepared_model_out,
    )
