"""Small ONNX graph-stage helpers shared by pipeline builders."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def copy_missing_opset_domains(target: onnx.ModelProto, source: onnx.ModelProto) -> None:
    """Copy opset imports from source when target has not declared that domain."""
    target_domains = {opset.domain for opset in target.opset_import}
    for opset in source.opset_import:
        if opset.domain in target_domains:
            continue
        new_opset = target.opset_import.add()
        new_opset.domain = opset.domain
        new_opset.version = opset.version
        target_domains.add(opset.domain)


def clear_outputs(graph: onnx.GraphProto) -> None:
    while len(graph.output) > 0:
        graph.output.pop()


def add_embedding_pooling(graph: onnx.GraphProto, pooling: str) -> str:
    if pooling == "mean":
        pool_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="pool_axes_1")
        graph.initializer.append(pool_axes)
        graph.node.append(
            helper.make_node(
                "ReduceMean",
                ["core_last_hidden_state", "pool_axes_1"],
                ["pooled"],
                keepdims=0,
            )
        )
        return "pooled"

    if pooling == "cls":
        cls_indices = numpy_helper.from_array(np.array([0], dtype=np.int64), name="cls_indices")
        graph.initializer.append(cls_indices)
        graph.node.append(
            helper.make_node(
                "Gather",
                ["core_last_hidden_state", "cls_indices"],
                ["pooled_2d"],
                axis=1,
            )
        )
        squeeze_cls = numpy_helper.from_array(np.array([1], dtype=np.int64), name="squeeze_cls_axes")
        graph.initializer.append(squeeze_cls)
        graph.node.append(
            helper.make_node("Squeeze", ["pooled_2d", "squeeze_cls_axes"], ["pooled"])
        )
        return "pooled"

    raise ValueError(f"Unsupported pooling: {pooling!r}")


def add_l2_normalization(graph: onnx.GraphProto, input_name: str) -> str:
    pow_exp = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="pow_exp")
    graph.initializer.append(pow_exp)
    graph.node.append(helper.make_node("Pow", [input_name, "pow_exp"], ["squared"]))

    axes_neg1 = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes_neg1")
    graph.initializer.append(axes_neg1)
    graph.node.append(
        helper.make_node("ReduceSum", ["squared", "axes_neg1"], ["sum_sq"], keepdims=1)
    )
    graph.node.append(helper.make_node("Sqrt", ["sum_sq"], ["l2_norm"]))

    eps = numpy_helper.from_array(np.array(1e-12, dtype=np.float32), name="eps_val")
    graph.initializer.append(eps)
    graph.node.append(helper.make_node("Max", ["l2_norm", "eps_val"], ["l2_safe"]))
    graph.node.append(helper.make_node("Div", [input_name, "l2_safe"], ["emb_2d"]))
    return "emb_2d"


def expose_squeezed_float_output(
    graph: onnx.GraphProto,
    input_name: str,
    output_name: str,
    axes: Iterable[int],
    shape: list[int],
    axes_initializer_name: str,
) -> None:
    squeeze_axes = numpy_helper.from_array(np.array(list(axes), dtype=np.int64), name=axes_initializer_name)
    graph.initializer.append(squeeze_axes)
    graph.node.append(helper.make_node("Squeeze", [input_name, axes_initializer_name], [output_name]))
    graph.output.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape))


def expose_dynamic_int64_sequence_outputs(
    graph: onnx.GraphProto,
    names: Iterable[str],
    dim_param: str,
) -> None:
    clear_outputs(graph)
    for name in names:
        vi = helper.make_tensor_value_info(name, TensorProto.INT64, [1, None])
        seq_dim = vi.type.tensor_type.shape.dim[1]
        seq_dim.ClearField("dim_value")
        seq_dim.dim_param = dim_param
        graph.output.append(vi)


def pin_dynamic_batch_to_one(graph: onnx.GraphProto) -> None:
    for vi in list(graph.value_info) + list(graph.input):
        if vi.type.tensor_type.shape:
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_param == "batch_size":
                    dim.ClearField("dim_param")
                    dim.dim_value = 1
