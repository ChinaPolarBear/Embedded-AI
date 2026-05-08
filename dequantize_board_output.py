"""
Dequantize raw board output using the scale inferred from a raw QONNX export.

Typical usage:
    python dequantize_board_output.py --model deeponet_u250_int4_qonnx.onnx --input output.bin

Batch usage:
    python dequantize_board_output.py --model deeponet_u250_int4_qonnx.onnx --cases_dir verification --input_name output.bin

The script supports either:
  - a raw binary dump such as `output.bin`
  - a saved integer `.npy` array such as `output_raw_int16_1x64.npy`

Outputs:
  - `output_raw_<dtype>_<batch>x<width>.npy` when decoding from `.bin`
  - `output_dequant.npy`

The scale is inferred from the QONNX graph. For the current deploy models in this
repository, the output is typically produced by:

    latent_quant -> out_fc weight quant -> MatMul -> y_out

so the dequantization scale is:

    scale(output) = scale(latent_quant) * scale(out_fc_weight_quant)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper


SUPPORTED_DTYPES = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
}


@dataclass
class DequantResult:
    raw_path: Path
    dequant_path: Path
    raw_shape: tuple[int, ...]
    dtype_name: str
    scale: float


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    mapping: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            mapping[output] = node
    return mapping


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _constant_scalar_map(model: onnx.ModelProto) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        value_attr = next((attr for attr in node.attribute if attr.name == "value"), None)
        if value_attr is None:
            continue
        arr = np.asarray(numpy_helper.to_array(value_attr.t))
        if arr.size == 1:
            scalars[node.output[0]] = float(arr.reshape(()))
    return scalars


def _get_shape_from_model(model: onnx.ModelProto) -> list[int | None]:
    if len(model.graph.output) != 1:
        raise ValueError("Expected exactly one graph output in the deploy QONNX model.")
    dims: list[int | None] = []
    for dim in model.graph.output[0].type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else None)
    return dims


def _infer_tensor_scale(model: onnx.ModelProto) -> float:
    producers = _producer_map(model)
    initializers = _initializer_map(model)
    constant_scalars = _constant_scalar_map(model)
    memo: dict[str, float | None] = {}

    def scalar_value(name: str) -> float | None:
        if name in constant_scalars:
            return constant_scalars[name]
        if name in initializers:
            arr = np.asarray(initializers[name])
            if arr.size == 1:
                return float(arr.reshape(()))
        return None

    def scale_of_tensor(name: str) -> float | None:
        if name in memo:
            return memo[name]

        node = producers.get(name)
        if node is None:
            memo[name] = None
            return None

        result: float | None = None

        if node.op_type == "Quant" and node.domain == "qonnx.custom_op.general":
            result = scalar_value(node.input[1])
        elif node.op_type in {"Identity", "Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Cast"}:
            result = scale_of_tensor(node.input[0]) if node.input else None
        elif node.op_type in {"MatMul", "Gemm", "Mul"}:
            if len(node.input) >= 2:
                lhs = scale_of_tensor(node.input[0])
                rhs = scale_of_tensor(node.input[1])
                if lhs is not None and rhs is not None:
                    result = lhs * rhs
        elif node.op_type in {"Add", "Sub"}:
            known_scales = [scale_of_tensor(inp) for inp in node.input[:2]]
            known = [val for val in known_scales if val is not None]
            if len(known) == 2 and np.isclose(known[0], known[1], rtol=1e-5, atol=1e-8):
                result = known[0]
            elif len(known) == 1:
                result = known[0]
        elif node.op_type in {"Relu"}:
            result = scale_of_tensor(node.input[0]) if node.input else None

        memo[name] = result
        return result

    output_name = model.graph.output[0].name
    scale = scale_of_tensor(output_name)
    if scale is None:
        raise RuntimeError(
            "Could not infer the output scale from the QONNX graph. "
            "The graph tail may use an unsupported operation sequence."
        )
    return float(scale)


def _normalised_dtype_name(dtype_name: str) -> str:
    key = dtype_name.lower()
    if key not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype `{dtype_name}`. Supported values: {', '.join(SUPPORTED_DTYPES)}")
    return key


def _read_binary(input_path: Path, dtype_name: str, shape_hint: tuple[int, int] | None) -> np.ndarray:
    dtype = SUPPORTED_DTYPES[dtype_name]
    raw = np.fromfile(input_path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f"{input_path} is empty.")

    if shape_hint is None:
        return raw

    _, width = shape_hint
    if width <= 0:
        raise ValueError(f"Invalid width inferred from model shape: {shape_hint}")
    if raw.size % width != 0:
        raise ValueError(
            f"Binary file {input_path} contains {raw.size} elements of dtype {dtype_name}, "
            f"which is not divisible by output width {width}."
        )
    batch = raw.size // width
    return raw.reshape(batch, width)


def _load_raw_array(input_path: Path, dtype_name: str, shape_hint: tuple[int, int] | None) -> np.ndarray:
    if input_path.suffix.lower() == ".npy":
        arr = np.load(input_path)
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{input_path} must contain an integer array for dequantization, got {arr.dtype}.")
        return np.asarray(arr, dtype=SUPPORTED_DTYPES[dtype_name])

    return _read_binary(input_path, dtype_name, shape_hint)


def _default_raw_output_name(dtype_name: str, arr: np.ndarray) -> str:
    if arr.ndim == 1:
        return f"output_raw_{dtype_name}_{arr.shape[0]}.npy"
    dims = "x".join(str(x) for x in arr.shape)
    return f"output_raw_{dtype_name}_{dims}.npy"


def _dequantize_single(
    model_path: Path,
    input_path: Path,
    dtype_name: str,
    raw_out_name: str | None,
    dequant_out_name: str,
    shape_override: tuple[int, int] | None,
    save_raw_for_npy_input: bool,
) -> DequantResult:
    model = onnx.load(str(model_path))
    model_shape = _get_shape_from_model(model)

    shape_hint = shape_override
    if shape_hint is None and len(model_shape) == 2 and model_shape[1] is not None:
        batch_hint = model_shape[0] if model_shape[0] not in (None, 0) else 1
        shape_hint = (int(batch_hint), int(model_shape[1]))

    scale = _infer_tensor_scale(model)
    raw_arr = _load_raw_array(input_path, dtype_name, shape_hint)

    output_dir = input_path.parent
    save_raw = input_path.suffix.lower() != ".npy" or save_raw_for_npy_input

    raw_path = output_dir / (raw_out_name or _default_raw_output_name(dtype_name, raw_arr))
    if save_raw:
        np.save(raw_path, raw_arr.astype(np.int32))
    dequant_path = output_dir / dequant_out_name
    dequant = raw_arr.astype(np.float32) * scale
    np.save(dequant_path, dequant)

    print(f"[OK] Model               : {model_path}")
    print(f"[OK] Input               : {input_path}")
    print(f"[OK] Output scale        : {scale}")
    print(f"[OK] Raw shape           : {raw_arr.shape}")
    print(f"[OK] Raw dtype           : {raw_arr.dtype}")
    if save_raw:
        print(f"[OK] Saved raw integer   : {raw_path}")
    else:
        print(f"[OK] Raw integer source  : {input_path}")
    print(f"[OK] Saved dequantized   : {dequant_path}")

    return DequantResult(
        raw_path=raw_path if save_raw else input_path,
        dequant_path=dequant_path,
        raw_shape=tuple(int(x) for x in raw_arr.shape),
        dtype_name=dtype_name,
        scale=scale,
    )


def _iter_case_dirs(cases_dir: Path) -> Iterable[Path]:
    for path in sorted(cases_dir.iterdir()):
        if path.is_dir() and path.name.startswith("case_"):
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dequantize raw board output using the output scale inferred from a raw QONNX model.")
    parser.add_argument("--model", type=Path, default=Path("deeponet_u250_int4_qonnx.onnx"), help="raw QONNX model used to infer the output scale")
    parser.add_argument("--input", type=Path, help="single input file to decode/dequantize (`.bin` or integer `.npy`)")
    parser.add_argument("--cases_dir", type=Path, help="directory containing case_000, case_001, ... subdirectories")
    parser.add_argument("--input_name", type=str, default="output.bin", help="input filename to read inside each case directory when using --cases_dir")
    parser.add_argument("--dtype", type=str, default="int16", help="integer dtype stored in the raw board output")
    parser.add_argument("--raw_out_name", type=str, help="filename for the saved raw integer `.npy` when decoding a single input")
    parser.add_argument("--dequant_out_name", type=str, default="output_dequant.npy", help="filename for the saved dequantized output")
    parser.add_argument("--shape", nargs=2, type=int, metavar=("BATCH", "WIDTH"), help="override the inferred [B, width] shape when decoding binary input")
    parser.add_argument("--save_raw_for_npy_input", action="store_true", help="also save a normalised raw integer `.npy` when the input is already a `.npy` file")
    args = parser.parse_args()

    if args.input is None and args.cases_dir is None:
        parser.error("Provide either --input for a single file or --cases_dir for batch mode.")
    if args.input is not None and args.cases_dir is not None:
        parser.error("Use either --input or --cases_dir, not both.")

    args.dtype = _normalised_dtype_name(args.dtype)
    return args


def main() -> None:
    args = parse_args()
    shape_override = tuple(args.shape) if args.shape is not None else None

    if args.input is not None:
        _dequantize_single(
            model_path=args.model,
            input_path=args.input,
            dtype_name=args.dtype,
            raw_out_name=args.raw_out_name,
            dequant_out_name=args.dequant_out_name,
            shape_override=shape_override,
            save_raw_for_npy_input=args.save_raw_for_npy_input,
        )
        return

    case_dirs = list(_iter_case_dirs(args.cases_dir))
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found under {args.cases_dir}")

    print(f"[INFO] Batch dequantization for {len(case_dirs)} cases")
    for case_dir in case_dirs:
        input_path = case_dir / args.input_name
        if not input_path.is_file():
            print(f"[WARN] Skipping {case_dir.name}: {input_path.name} not found")
            continue
        print(f"\n[CASE] {case_dir.name}")
        _dequantize_single(
            model_path=args.model,
            input_path=input_path,
            dtype_name=args.dtype,
            raw_out_name=args.raw_out_name,
            dequant_out_name=args.dequant_out_name,
            shape_override=shape_override,
            save_raw_for_npy_input=args.save_raw_for_npy_input,
        )


if __name__ == "__main__":
    main()
