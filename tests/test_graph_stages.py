import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2oracle.graph_stages import (
    add_embedding_pooling,
    add_l2_normalization,
    clear_outputs,
    copy_missing_opset_domains,
    dedupe_opset_imports,
    expose_dynamic_int64_sequence_outputs,
    expose_squeezed_float_output,
    hidden_size,
    hidden_state_output,
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


def test_dedupe_opset_imports_collapses_duplicates_and_pins():
    model = helper.make_model(
        helper.make_graph([], "g", [], []),
        opset_imports=[
            helper.make_opsetid("", 14),
            helper.make_opsetid("", 17),
            helper.make_opsetid("ai.onnx.contrib", 1),
        ],
    )

    dedupe_opset_imports(model, pin={"": 18, "ai.onnx": 18})

    versions = {opset.domain: opset.version for opset in model.opset_import}
    # One entry per domain (duplicates collapsed).
    assert len(model.opset_import) == len(versions)
    # Pin overrides the resolved version; untouched domains keep their max.
    assert versions[""] == 18
    assert versions["ai.onnx"] == 18
    assert versions["ai.onnx.contrib"] == 1


def test_embedding_pooling_and_l2_normalization_add_expected_nodes():
    graph = helper.make_graph([], "g", [], [])

    pooled = add_embedding_pooling(graph, "mean", "core_last_hidden_state")
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
        add_embedding_pooling(graph, "last_token", "core_last_hidden_state")


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_pooling_and_normalization_read_the_named_hidden_state(pooling):
    """The pooled vector must come from whichever tensor the caller names, not a fixed one."""
    hidden = np.array([[[3.0, 0.0], [0.0, 8.0], [6.0, 16.0]]], dtype=np.float32)  # [1 batch, 3 tokens, 2 dims]
    graph = helper.make_graph(
        [],
        "pool",
        [helper.make_tensor_value_info("core_token_embeddings", TensorProto.FLOAT, [1, None, 2])],
        [],
    )
    pooled = add_embedding_pooling(graph, pooling, "core_token_embeddings")
    normalized = add_l2_normalization(graph, pooled)
    graph.output.append(helper.make_tensor_value_info(normalized, TensorProto.FLOAT, [1, 2]))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    # helper.make_model defaults to the installed onnx release's IR version, which older
    # onnxruntime builds cannot load; the pooling behavior is IR-independent.
    model.ir_version = 10

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    (result,) = session.run(None, {"core_token_embeddings": hidden})

    expected = hidden.mean(axis=1) if pooling == "mean" else hidden[:, 0, :]
    expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_hidden_state_output_prefers_last_hidden_state_by_name():
    model = helper.make_model(
        helper.make_graph(
            [],
            "core",
            [],
            [
                helper.make_tensor_value_info("token_embeddings", TensorProto.FLOAT, [1, "seq", 8]),
                helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, "seq", 8]),
            ],
        )
    )

    assert hidden_state_output(model).name == "last_hidden_state"


def test_hidden_state_output_falls_back_to_first_rank3_float32_output():
    model = helper.make_model(
        helper.make_graph(
            [],
            "core",
            [],
            [
                helper.make_tensor_value_info("sentence_embedding", TensorProto.FLOAT, [1, 8]),
                helper.make_tensor_value_info("token_ids", TensorProto.INT64, [1, "seq", 8]),
                helper.make_tensor_value_info("half_states", TensorProto.FLOAT16, [1, "seq", 8]),
                helper.make_tensor_value_info("token_embeddings", TensorProto.FLOAT, [1, "seq", 8]),
                helper.make_tensor_value_info("other_states", TensorProto.FLOAT, [1, "seq", 8]),
            ],
        )
    )

    assert hidden_state_output(model).name == "token_embeddings"


def test_hidden_state_output_error_lists_what_the_model_does_output():
    model = helper.make_model(
        helper.make_graph(
            [],
            "core",
            [],
            [
                helper.make_tensor_value_info("sentence_embedding", TensorProto.FLOAT, [1, 8]),
                helper.make_tensor_value_info("pooler", TensorProto.FLOAT, ["batch", 8]),
            ],
        )
    )

    with pytest.raises(ValueError, match="hidden-state output") as excinfo:
        hidden_state_output(model)

    assert "sentence_embedding: float[1, 8]" in str(excinfo.value)
    assert "pooler: float[batch, 8]" in str(excinfo.value)


def test_hidden_size_reads_static_last_dimension_only():
    static = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, "seq", 384])
    dynamic = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, "seq", "hidden"])
    unknown_shape = helper.make_tensor_value_info("c", TensorProto.FLOAT, None)

    assert hidden_size(static) == 384
    assert hidden_size(dynamic) is None
    assert hidden_size(unknown_shape) is None


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
