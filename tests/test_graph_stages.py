import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2oracle.graph_stages import (
    add_embedding_pooling,
    add_l2_normalization,
    clear_outputs,
    copy_missing_opset_domains,
    expose_dynamic_int64_sequence_outputs,
    expose_squeezed_float_output,
    pin_dynamic_batch_to_one,
)


def test_copy_missing_opset_domains_preserves_existing_domains():
    target = helper.make_model(
        helper.make_graph([], "target", [], []),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    source = helper.make_model(
        helper.make_graph([], "source", [], []),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.contrib", 1)],
    )

    copy_missing_opset_domains(target, source)

    versions = {opset.domain: opset.version for opset in target.opset_import}
    assert versions[""] == 18
    assert versions["ai.onnx.contrib"] == 1


def test_embedding_pooling_and_l2_normalization_add_expected_nodes():
    graph = helper.make_graph([], "g", [], [])

    pooled = add_embedding_pooling(graph, "mean")
    normalized = add_l2_normalization(graph, pooled)

    assert pooled == "pooled"
    assert normalized == "emb_2d"
    assert [node.op_type for node in graph.node] == [
        "ReduceMean",
        "Pow",
        "ReduceSum",
        "Sqrt",
        "Max",
        "Div",
    ]
    initializers = {init.name: numpy_helper.to_array(init).tolist() for init in graph.initializer}
    assert initializers["pool_axes_1"] == [1]
    assert initializers["axes_neg1"] == [-1]


def test_embedding_pooling_rejects_unknown_strategy():
    graph = helper.make_graph([], "g", [], [])

    with pytest.raises(ValueError, match="Unsupported pooling"):
        add_embedding_pooling(graph, "last_token")


def test_expose_helpers_replace_outputs_and_pin_batch_dims():
    batch_dim_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])
    batch_dim_input.type.tensor_type.shape.dim[0].ClearField("dim_value")
    batch_dim_input.type.tensor_type.shape.dim[0].dim_param = "batch_size"
    old_output = helper.make_tensor_value_info("old", TensorProto.FLOAT, [1])
    graph = helper.make_graph([], "g", [batch_dim_input], [old_output])

    clear_outputs(graph)
    expose_squeezed_float_output(
        graph,
        input_name="x",
        output_name="embedding",
        axes=[0],
        shape=[4],
        axes_initializer_name="squeeze_axes_final",
    )
    pin_dynamic_batch_to_one(graph)

    assert [output.name for output in graph.output] == ["embedding"]
    assert graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1


def test_expose_dynamic_int64_sequence_outputs_declares_dynamic_second_dim():
    graph = helper.make_graph(
        [],
        "g",
        [],
        [helper.make_tensor_value_info("old", TensorProto.FLOAT, [1])],
    )

    expose_dynamic_int64_sequence_outputs(graph, ["input_ids"], "pair_sequence_length")

    output = graph.output[0]
    assert output.name == "input_ids"
    dims = output.type.tensor_type.shape.dim
    assert dims[0].dim_value == 1
    assert dims[1].dim_param == "pair_sequence_length"
