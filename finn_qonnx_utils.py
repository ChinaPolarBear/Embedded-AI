from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib
import warnings

import onnx
from onnx import numpy_helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup_model


QONNX_CONST_OPS = {"Quant", "BipolarQuant", "Trunc"}
LINEAR_OPS = {"Gemm", "MatMul"}


def ensure_brevitas_qonnx_export_support() -> None:
    """Patch Brevitas export so QONNX export still works without onnxoptimizer."""

    export_manager = importlib.import_module("brevitas.export.onnx.manager")
    if export_manager.onnx is None:
        export_manager.onnx = importlib.import_module("onnx")

    if export_manager.opt is None:
        class _NoOpOptimizer:
            @staticmethod
            def optimize(model, passes):
                warnings.warn(
                    "onnxoptimizer is not installed; falling back to a no-op optimizer. "
                    "QONNX export will continue and the qonnx cleanup pass will normalize the graph.",
                    stacklevel=2,
                )
                return model

        export_manager.opt = _NoOpOptimizer()


def derive_output_path(in_path: str | Path, suffix: str) -> str:
    path = Path(in_path)
    return str(path.with_name(path.stem + suffix + path.suffix))


def _op_key(node) -> str:
    return f"{node.domain}:{node.op_type}" if node.domain else node.op_type


def _get_tensor_shape(model: ModelWrapper, tensor_name: str):
    shape = model.get_tensor_shape(tensor_name)
    if shape is not None:
        return shape
    initializer = model.get_initializer(tensor_name)
    if initializer is not None:
        return list(initializer.shape)
    return None


def _producer_map(model: ModelWrapper):
    mapping = {}
    for node in model.graph.node:
        for output_name in node.output:
            mapping[output_name] = node
    return mapping


def _is_const_like_tensor(model: ModelWrapper, tensor_name: str, producers) -> tuple[bool, str]:
    if model.get_initializer(tensor_name) is not None:
        return True, "initializer"

    producer = producers.get(tensor_name)
    if producer is None:
        return False, "dynamic graph input"

    if producer.op_type in QONNX_CONST_OPS and all(model.get_initializer(inp) is not None for inp in producer.input):
        return True, f"{producer.op_type} from initializers"

    return False, f"{_op_key(producer)} output"


def print_model_summary(model: ModelWrapper, label: str) -> None:
    ops = Counter(_op_key(node) for node in model.graph.node)
    print(f"\n[{label}]")
    print(f"  nodes={len(model.graph.node)} initializers={len(model.graph.initializer)}")
    print(f"  ops={dict(ops)}")
    for graph_input in model.graph.input:
        print(f"  input  {graph_input.name}: shape={_get_tensor_shape(model, graph_input.name)} dtype={model.get_tensor_datatype(graph_input.name)}")
    for graph_output in model.graph.output:
        print(f"  output {graph_output.name}: shape={_get_tensor_shape(model, graph_output.name)} dtype={model.get_tensor_datatype(graph_output.name)}")


def print_quant_summary(model: ModelWrapper) -> None:
    const_quant = 0
    dynamic_quant = 0
    for node in model.graph.node:
        if node.op_type in QONNX_CONST_OPS:
            if all(model.get_initializer(inp) is not None for inp in node.input):
                const_quant += 1
            else:
                dynamic_quant += 1
    print(f"  quant nodes: const_like={const_quant} dynamic={dynamic_quant}")


def validate_linear_nodes(model: ModelWrapper, verbose: bool = True) -> list[str]:
    issues = []
    producers = _producer_map(model)

    for node in model.graph.node:
        if node.op_type not in LINEAR_OPS:
            continue

        if verbose:
            print(f"  {node.op_type} {node.name}:")

        if len(node.input) < 2:
            issues.append(f"{node.op_type} node {node.name} has fewer than 2 inputs.")
            continue

        for index, tensor_name in enumerate(node.input):
            tensor_shape = _get_tensor_shape(model, tensor_name)
            tensor_dtype = model.get_tensor_datatype(tensor_name)
            initializer = model.get_initializer(tensor_name)
            is_const_like, const_reason = _is_const_like_tensor(model, tensor_name, producers)
            if verbose:
                init_shape = list(initializer.shape) if initializer is not None else None
                print(
                    f"    input[{index}] {tensor_name}: shape={tensor_shape} dtype={tensor_dtype} "
                    f"initializer_shape={init_shape} const_like={is_const_like} ({const_reason})"
                )

        weight_name = node.input[1]
        weight_shape = _get_tensor_shape(model, weight_name)
        weight_const_like, weight_reason = _is_const_like_tensor(model, weight_name, producers)
        if weight_shape is None:
            issues.append(f"{node.op_type} node {node.name} weight tensor {weight_name} has no inferred shape.")
        elif len(weight_shape) != 2:
            issues.append(
                f"{node.op_type} node {node.name} weight tensor {weight_name} is not rank-2: shape={weight_shape}."
            )
        if not weight_const_like:
            issues.append(
                f"{node.op_type} node {node.name} weight tensor {weight_name} is not constant-like: {weight_reason}."
            )

        if node.op_type == "Gemm" and len(node.input) > 2:
            bias_name = node.input[2]
            bias_init = model.get_initializer(bias_name)
            if bias_init is None:
                issues.append(f"Gemm node {node.name} bias tensor {bias_name} is not an initializer.")

    return issues


def print_initializer_shapes(onnx_path: str) -> None:
    model = onnx.load(onnx_path)
    print("\n[Initializers]")
    for initializer in sorted(model.graph.initializer, key=lambda init: init.name):
        tensor = numpy_helper.to_array(initializer)
        print(f"  {initializer.name}: shape={list(tensor.shape)}")


def prepare_qonnx_for_finn(
    in_path: str,
    out_path: str | None = None,
    infer_shapes: bool = True,
    infer_datatypes: bool = True,
    verbose: bool = True,
) -> ModelWrapper:
    model = ModelWrapper(in_path)
    if verbose:
        print_model_summary(model, "Raw QONNX")

    model = cleanup_model(model, preserve_qnt_ops=True)
    if infer_shapes:
        model = model.transform(InferShapes())
    if infer_datatypes:
        model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())
    onnx.checker.check_model(model.model)

    if verbose:
        print_model_summary(model, "Prepared QONNX")
        print_quant_summary(model)
        linear_issues = validate_linear_nodes(model, verbose=True)
    else:
        linear_issues = validate_linear_nodes(model, verbose=False)

    if linear_issues:
        issue_text = "\n".join(f"  - {issue}" for issue in linear_issues)
        raise RuntimeError(f"QONNX validation failed:\n{issue_text}")

    if out_path is not None:
        model.save(out_path)
        if verbose:
            print(f"\n[Saved] {out_path}")

    return model


def convert_qonnx_to_finn(
    in_path: str,
    out_path: str,
    prepared_qonnx_path: str | None = None,
    verbose: bool = True,
) -> ModelWrapper:
    prepared_model = prepare_qonnx_for_finn(in_path, out_path=prepared_qonnx_path, verbose=verbose)

    try:
        from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FINN is not installed in the current Python environment. "
            "The QONNX preparation step completed successfully, but FINN conversion cannot run here."
        ) from exc

    try:
        finn_model = prepared_model.transform(ConvertQONNXtoFINN())
    except Exception as exc:
        print("\n[Debug] ConvertQONNXtoFINN failed. Linear node snapshot before conversion:")
        validate_linear_nodes(prepared_model, verbose=True)
        raise RuntimeError(f"ConvertQONNXtoFINN failed: {exc}") from exc

    # ConvertQONNXtoFINN can introduce unnamed nodes, which later cause empty
    # HLS top-function names such as `top_.cpp` / `set_top` with no argument.
    finn_model = finn_model.transform(GiveUniqueNodeNames())
    finn_model = finn_model.transform(GiveReadableTensorNames())
    finn_model = finn_model.transform(InferShapes())
    finn_model = finn_model.transform(InferDataTypes())
    onnx.checker.check_model(finn_model.model)
    finn_model.save(out_path)

    if verbose:
        print_model_summary(finn_model, "FINN-ONNX")
        print(f"\n[Saved] {out_path}")

    return finn_model
