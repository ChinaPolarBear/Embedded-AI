"""
Step 5: Generate one board-side verification case using an SSFM reference.

Typical usage:
    python step5_generate_verification.py \
        --out_dir verification \
        --source random \
        --seed 123

Multi-case usage:
    python step5_generate_verification.py \
        --out_dir verification \
        --source random \
        --seed 123 \
        --num_cases 5

This script:
  1) Selects one waveform sample from the existing project data path.
  2) Converts it into the deploy-model branch input `u_in`.
  3) Uses SSFM to obtain the clean reference output at z=L.
  4) Saves:
       - input.npy
       - input_int4.npy
       - input.bin
       - ssfm_output.npy
       - ssfm_output_probe.npy
       - verification_case.npz

Notes:
  - `input.npy` is the compact float32 complex input expected by the exported probe model:
    first N_probe values are real(A(0,t_probe)), last N_probe values are imag(A(0,t_probe)).
  - `ssfm_output_probe.npy` matches the compact deploy output layout:
    first N_probe values are real(A(L,t_probe)), last N_probe values are imag(A(L,t_probe)).
  - `input.bin` is the board-ready raw input generated from the exact same sample as
    `input.npy` and `verification_case.npz`.
  - Re-running this script automatically deletes stale board-returned files such as
    `output.bin`, `output.npy`, `output_dequant.npy`, and `output_raw*.npy` in the
    target case directory before writing the new verification bundle.
"""
# python step5_generate_verification.py --out_dir verification --source random --seed 123 --mat trunk_matrices.npz --model deeponet_u250_int4_qonnx.onnx
# generate 5 cases at one time
# python step5_generate_verification.py --out_dir verification --source random --seed 123 --num_cases 5
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import pinn_physics_model as pm


DEFAULT_INPUT_SCALE = 0.22732384502887726
DEFAULT_ZERO_POINT = 0.0
DEFAULT_BIT_WIDTH = 4


def _load_input_length_from_model(model_path: Path) -> int | None:
    try:
        import onnx
    except ModuleNotFoundError:
        return None

    if not model_path.is_file():
        return None

    model = onnx.load(str(model_path))
    if not model.graph.input:
        return None

    dims = model.graph.input[0].type.tensor_type.shape.dim
    shape = []
    for dim in dims:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            return None

    if len(shape) != 2 or shape[0] != 1:
        return None
    return shape[1]


def _load_quant_params_from_model(model_path: Path) -> tuple[float, float, int] | None:
    try:
        import onnx
        from onnx import numpy_helper
    except ModuleNotFoundError:
        return None

    if not model_path.is_file():
        return None

    model = onnx.load(str(model_path))
    if not model.graph.input:
        return None

    graph_input_name = model.graph.input[0].name
    value_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                value_map[node.output[0]] = numpy_helper.to_array(attr.t)
                break

    for node in model.graph.node:
        if node.op_type != "Quant":
            continue
        if not node.input or node.input[0] != graph_input_name:
            continue
        if len(node.input) < 4:
            return None

        scale_name = node.input[1]
        zero_name = node.input[2]
        bit_width_name = node.input[3]
        if scale_name not in value_map or zero_name not in value_map or bit_width_name not in value_map:
            return None

        scale = float(np.asarray(value_map[scale_name]).reshape(()))
        zero_point = float(np.asarray(value_map[zero_name]).reshape(()))
        bit_width = int(round(float(np.asarray(value_map[bit_width_name]).reshape(()))))
        return scale, zero_point, bit_width

    return None


def _resolve_quant_params(model_path: Path | None, scale: float | None) -> tuple[float, float, int, str]:
    if scale is not None:
        return float(scale), DEFAULT_ZERO_POINT, DEFAULT_BIT_WIDTH, "cli_override"

    if model_path is not None:
        model_params = _load_quant_params_from_model(model_path)
        if model_params is not None:
            q_scale, q_zero, q_bw = model_params
            return q_scale, q_zero, q_bw, f"model:{model_path}"

    return DEFAULT_INPUT_SCALE, DEFAULT_ZERO_POINT, DEFAULT_BIT_WIDTH, "built_in_default"


def _quantize_signed(arr: np.ndarray, scale: float, zero_point: float, bit_width: int) -> np.ndarray:
    if bit_width < 1 or bit_width > 8:
        raise ValueError(f"This helper expects a signed input bit width in [1, 8], got {bit_width}")
    if abs(zero_point) > 1e-9:
        raise ValueError(f"This helper expects zero_point=0 for signed integer input, got {zero_point}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    qmin = -(2 ** (bit_width - 1))
    qmax = (2 ** (bit_width - 1)) - 1
    quant = np.rint(arr / scale).astype(np.int32)
    quant = np.clip(quant, qmin, qmax).astype(np.int8)
    return quant


def prepare_board_input_array(
    arr: np.ndarray,
    model_path: Path | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | tuple[int, ...]]]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ValueError(
            f"Expected input.npy shape (1, N_scalar) for the current board interface, got {arr.shape}"
        )

    expected_input_len = _load_input_length_from_model(model_path) if model_path is not None else None
    if expected_input_len is not None and arr.shape[1] != expected_input_len:
        raise ValueError(
            f"Expected input.npy shape (1, {expected_input_len}) from model {model_path}, got {arr.shape}"
        )

    q_scale, q_zero, q_bw, scale_source = _resolve_quant_params(model_path, scale)
    q_arr = _quantize_signed(arr, q_scale, q_zero, q_bw)

    qmin = -(2 ** (q_bw - 1))
    qmax = (2 ** (q_bw - 1)) - 1
    meta: dict[str, float | int | str | tuple[int, ...]] = {
        "input_shape": tuple(arr.shape),
        "quantized_shape": tuple(q_arr.shape),
        "scale_source": scale_source,
        "scale": q_scale,
        "zero_point": q_zero,
        "bit_width": q_bw,
        "float_min": float(arr.min()),
        "float_max": float(arr.max()),
        "int_min": int(q_arr.min()),
        "int_max": int(q_arr.max()),
        "sat_min_count": int(np.count_nonzero(q_arr == qmin)),
        "sat_max_count": int(np.count_nonzero(q_arr == qmax)),
    }
    return q_arr, meta


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def _complex_to_two_channel(x: torch.Tensor) -> np.ndarray:
    x_np = x.detach().cpu().numpy()
    return np.stack([x_np.real.astype(np.float32), x_np.imag.astype(np.float32)], axis=0)


def _load_probe_indices(mat_path: str) -> torch.Tensor:
    data = np.load(mat_path)
    if "time_indices" in data.files:
        indices = np.asarray(data["time_indices"], dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError(f"time_indices in {mat_path} is empty")
        return torch.from_numpy(indices).long()

    if "M_real" in data.files:
        n_t_out = int(np.asarray(data["M_real"]).shape[0])
        if n_t_out == pm.N_t:
            return torch.arange(pm.N_t).long()
        raise ValueError(
            f"{mat_path} is missing time_indices, so probe positions for n_t_out={n_t_out} are ambiguous"
        )

    raise ValueError(f"{mat_path} does not contain M_real/time_indices needed for deploy probing")


def _build_deploy_views(
    A0: torch.Tensor,
    AL_clean: torch.Tensor,
    probe_indices: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        A0_probe = A0[probe_indices]
        u_in_t = torch.cat([A0_probe.real, A0_probe.imag], dim=0).unsqueeze(0).float()

    u_in = u_in_t.detach().cpu().numpy().astype(np.float32)
    ssfm_output = _complex_to_two_channel(AL_clean)[None, :, :]
    AL_probe = AL_clean[probe_indices]
    ssfm_output_probe = np.concatenate(
        [
            AL_probe.real.detach().cpu().numpy().astype(np.float32),
            AL_probe.imag.detach().cpu().numpy().astype(np.float32),
        ]
    )[None, :]
    t_grid = pm.t_grid.detach().cpu().numpy().astype(np.float32)
    probe_t_grid = t_grid[probe_indices.detach().cpu().numpy()]
    return u_in, ssfm_output, ssfm_output_probe, t_grid, probe_t_grid


def _save_board_input_set(
    out_dir_path: Path,
    u_in: np.ndarray,
    model_path: Path | None,
    scale: float | None,
) -> dict[str, object]:
    q_arr, quant_meta = prepare_board_input_array(u_in, model_path=model_path, scale=scale)
    bit_width = int(quant_meta["bit_width"])

    float_path = out_dir_path / "input.npy"
    bin_path = out_dir_path / "input.bin"
    quant_path = out_dir_path / f"input_int{bit_width}.npy"

    np.save(float_path, u_in)
    np.save(quant_path, q_arr)
    q_arr.reshape(-1).tofile(bin_path)

    entry: dict[str, object] = {
        "float_npy": float_path.name,
        "quant_npy": quant_path.name,
        "bin": bin_path.name,
        "scale_source": str(quant_meta["scale_source"]),
        "scale": float(quant_meta["scale"]),
        "zero_point": float(quant_meta["zero_point"]),
        "bit_width": bit_width,
        "input_shape": list(quant_meta["input_shape"]),
        "quantized_shape": list(quant_meta["quantized_shape"]),
        "int_min": int(quant_meta["int_min"]),
        "int_max": int(quant_meta["int_max"]),
        "sat_min_count": int(quant_meta["sat_min_count"]),
        "sat_max_count": int(quant_meta["sat_max_count"]),
    }
    return entry


def _cleanup_previous_runtime_outputs(out_dir_path: Path) -> list[str]:
    removed: list[str] = []
    patterns = [
        "output.bin",
        "output.npy",
        "output_dequant.npy",
        "output_raw*.npy",
        "local_qonnx_output.npy",
    ]

    for pattern in patterns:
        for path in sorted(out_dir_path.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(path.name)

    return removed


def _write_verification_case(
    out_dir_path: Path,
    source_desc: str,
    sample_index: int,
    seed: int | None,
    probe_indices: torch.Tensor,
    u_in: np.ndarray,
    ssfm_output: np.ndarray,
    ssfm_output_probe: np.ndarray,
    t_grid: np.ndarray,
    probe_t_grid: np.ndarray,
    A0: torch.Tensor,
    AL_clean: torch.Tensor,
    model_path: Path | None,
    scale: float | None,
) -> dict[str, object]:
    input_meta = _save_board_input_set(
        out_dir_path=out_dir_path,
        u_in=u_in,
        model_path=model_path,
        scale=scale,
    )
    n_probe = int(probe_indices.numel())

    np.save(out_dir_path / "ssfm_output.npy", ssfm_output)
    np.save(out_dir_path / "ssfm_output_probe.npy", ssfm_output_probe)
    np.savez(
        out_dir_path / "verification_case.npz",
        source=np.array([source_desc]),
        sample_index=np.array([sample_index], dtype=np.int32),
        seed=np.array([-1 if seed is None else seed], dtype=np.int64),
        input_format=np.array([f"compact_real{n_probe}_imag{n_probe}_float32"]),
        output_format=np.array(["two_channel_real_imag_float32"]),
        deploy_output_format=np.array([f"compact_real{n_probe}_imag{n_probe}_float32"]),
        t_grid=t_grid,
        deploy_t_grid=probe_t_grid,
        propagation_distance=np.array([pm.L], dtype=np.float32),
        u_in=u_in,
        ssfm_output=ssfm_output,
        ssfm_output_probe=ssfm_output_probe,
        expected_output=ssfm_output_probe,
        probe_indices=probe_indices.detach().cpu().numpy().astype(np.int32),
        input_quant_scale=np.array([float(input_meta["scale"])], dtype=np.float32),
        input_quant_zero_point=np.array([float(input_meta["zero_point"])], dtype=np.float32),
        input_quant_bit_width=np.array([int(input_meta["bit_width"])], dtype=np.int32),
        A0_real=A0.real.detach().cpu().numpy().astype(np.float32),
        A0_imag=A0.imag.detach().cpu().numpy().astype(np.float32),
        AL_clean_real=AL_clean.real.detach().cpu().numpy().astype(np.float32),
        AL_clean_imag=AL_clean.imag.detach().cpu().numpy().astype(np.float32),
    )
    return input_meta


def main(
    out_dir: str,
    source: str,
    sample_index: int,
    seed: int | None,
    mat: str,
    model: str | None,
    scale: float | None,
    num_cases: int,
) -> None:
    _set_seed(seed)
    if num_cases < 1:
        raise ValueError(f"num_cases must be >= 1, got {num_cases}")

    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(model).resolve() if model is not None else None

    probe_indices = _load_probe_indices(mat)
    n_probe = int(probe_indices.numel())
    manifest: list[dict[str, object]] = []

    for case_idx in range(num_cases):
        effective_sample_index = sample_index + case_idx
        case_out_dir = out_dir_path if num_cases == 1 else out_dir_path / f"case_{case_idx:03d}"
        case_out_dir.mkdir(parents=True, exist_ok=True)
        removed_outputs = _cleanup_previous_runtime_outputs(case_out_dir)

        A0, AL_clean, source_desc = _select_sample(source, effective_sample_index)
        u_in, ssfm_output, ssfm_output_probe, t_grid, probe_t_grid = _build_deploy_views(
            A0, AL_clean, probe_indices
        )
        input_meta = _write_verification_case(
            out_dir_path=case_out_dir,
            source_desc=source_desc,
            sample_index=effective_sample_index,
            seed=seed,
            probe_indices=probe_indices,
            u_in=u_in,
            ssfm_output=ssfm_output,
            ssfm_output_probe=ssfm_output_probe,
            t_grid=t_grid,
            probe_t_grid=probe_t_grid,
            A0=A0,
            AL_clean=AL_clean,
            model_path=model_path,
            scale=scale,
        )

        manifest.append(
            {
                "case_index": case_idx,
                "case_dir": "." if num_cases == 1 else case_out_dir.name,
                "source": source_desc,
                "sample_index": effective_sample_index,
                "seed": None if seed is None else int(seed),
                "probe_points": n_probe,
                "input_shape": list(u_in.shape),
                "input_bin": str((case_out_dir / "input.bin").name),
                "verification_case": str((case_out_dir / "verification_case.npz").name),
                "input_quant_bit_width": int(input_meta["bit_width"]),
                "input_quant_scale": float(input_meta["scale"]),
                "input_quant_zero_point": float(input_meta["zero_point"]),
            }
        )

        print(f"[OK] Generated verification tensors in: {case_out_dir}")
        print(f"     source        : {source_desc}")
        print(f"     seed          : {seed if seed is not None else 'none'}")
        print(f"     probe points  : {n_probe} complex samples")
        print(f"     input.npy     : shape={tuple(u_in.shape)} dtype={u_in.dtype}")
        print(f"     input.bin     : {case_out_dir / 'input.bin'}")
        print(
            f"     input quant   : int{int(input_meta['bit_width'])} "
            f"scale={float(input_meta['scale']):.9g} zero_point={float(input_meta['zero_point']):.9g}"
        )
        print(f"     ssfm_output   : shape={tuple(ssfm_output.shape)} dtype={ssfm_output.dtype}")
        print(f"     ssfm_probe    : shape={tuple(ssfm_output_probe.shape)} dtype={ssfm_output_probe.dtype}")
        if removed_outputs:
            print(f"     cleaned       : removed stale runtime files: {', '.join(removed_outputs)}")
        if num_cases == 1:
            print("     note          : input.npy, input_int4.npy, input.bin, and verification_case.npz")
            print("                     all refer to the same single sample.")

    if num_cases > 1:
        manifest_path = out_dir_path / "cases_manifest.json"
        manifest_path.write_text(json.dumps({"cases": manifest}, indent=2), encoding="utf-8")
        print(f"[OK] Wrote multi-case manifest: {manifest_path}")
        print("     note          : each case_xxx directory contains one self-consistent sample bundle.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="verification")
    ap.add_argument("--source", type=str, choices=["train", "test", "random"], default="random")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Only affects --source random.",
    )
    ap.add_argument(
        "--mat",
        type=str,
        default="trunk_matrices.npz",
        help="Deploy trunk matrix file used to recover the exact time_indices for the board probe.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="deeponet_u250_int4_qonnx.onnx",
        help="Optional QONNX model used to auto-read the input quantization scale.",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional manual override for the input quantization scale.",
    )
    ap.add_argument(
        "--num_cases",
        type=int,
        default=1,
        help="Number of verification cases to generate. Uses case_000, case_001, ... subdirectories when > 1.",
    )
    args = ap.parse_args()
    main(
        out_dir=args.out_dir,
        source=args.source,
        sample_index=args.sample_index,
        seed=args.seed,
        mat=args.mat,
        model=args.model,
        scale=args.scale,
        num_cases=args.num_cases,
    )
