"""Unit tests for the augmented ONNX pipeline.

These tests do real HF downloads on first run — they are slow (~30-60s first time,
cached thereafter under ~/.cache/huggingface/). They do NOT require Oracle.

Run with: pytest tests/test_pipeline.py -v -m slow
"""

import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import onnx2oracle.pipeline as pipeline
from onnx2oracle.pipeline import (
    _external_data_locations,
    _external_data_repo_path,
    _generate_tokenizer_model,
    _require_torch_for_export_fallback,
    _should_use_export_fallback,
    _splice_query_doc_subgraph,
    _truncate_and_unsqueeze_tokenizer_outputs,
    build_augmented,
    build_reranker,
)
from onnx2oracle.presets import get_preset


def test_external_data_locations_reads_onnx_sidecar_metadata():
    tensor = TensorProto()
    tensor.name = "weights"
    tensor.data_type = TensorProto.FLOAT
    tensor.dims.extend([1])
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = "model.onnx_data"
    graph = helper.make_graph(nodes=[], name="g", inputs=[], outputs=[], initializer=[tensor])
    model = helper.make_model(graph)

    assert _external_data_locations(model) == ["model.onnx_data"]


def test_external_data_repo_path_is_relative_to_onnx_dir():
    assert _external_data_repo_path("model.onnx_data") == "onnx/model.onnx_data"
    assert _external_data_repo_path("shards/model_00001.bin") == "onnx/shards/model_00001.bin"


@pytest.mark.parametrize("location", ["../secret", "/tmp/model.bin", ".", ""])
def test_external_data_repo_path_rejects_unsafe_locations(location):
    with pytest.raises(ValueError, match="Unsafe ONNX external data location"):
        _external_data_repo_path(location)


def test_truncate_and_unsqueeze_tokenizer_outputs_adds_slice_before_batch_dim():
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["source_ids"], ["input_ids"])],
        name="g",
        inputs=[helper.make_tensor_value_info("source_ids", TensorProto.INT64, [None])],
        outputs=[helper.make_tensor_value_info("input_ids", TensorProto.INT64, [None])],
    )
    model = helper.make_model(graph)

    _truncate_and_unsqueeze_tokenizer_outputs(model, ["input_ids"], max_length=7)

    nodes = list(model.graph.node)
    assert [node.op_type for node in nodes] == ["Identity", "Slice", "Unsqueeze"]
    assert nodes[0].output == ["input_ids_raw"]
    assert nodes[1].input == ["input_ids_raw", "truncate_starts", "truncate_ends", "truncate_axes", "truncate_steps"]
    assert nodes[1].output == ["input_ids_flat"]
    assert nodes[2].input == ["input_ids_flat", "unsqueeze_axes_0"]
    assert nodes[2].output == ["input_ids"]

    initializers = {
        initializer.name: numpy_helper.to_array(initializer).tolist() for initializer in model.graph.initializer
    }
    assert initializers["truncate_ends"] == [7]

    dims = model.graph.output[0].type.tensor_type.shape.dim
    assert dims[0].dim_value == 1
    assert dims[1].dim_param == "sequence_length"


def test_export_fallback_requires_torch_extra_when_torch_is_missing(monkeypatch):
    monkeypatch.setattr(pipeline, "find_spec", lambda name: None if name == "torch" else object())

    with pytest.raises(RuntimeError, match=r'onnx2oracle\[export\]'):
        _require_torch_for_export_fallback()


def test_export_fallback_only_handles_missing_hub_entry():
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError

    assert _should_use_export_fallback(EntryNotFoundError("missing model.onnx"))
    assert not _should_use_export_fallback(LocalEntryNotFoundError("offline cache miss"))
    assert not _should_use_export_fallback(RepositoryNotFoundError("private or missing repo"))


def test_generate_tokenizer_model_rejects_unexpected_outputs(monkeypatch):
    import onnxruntime_extensions

    def fake_gen_processing_models(*args, **kwargs):
        graph = helper.make_graph(
            nodes=[],
            name="tokenizer",
            inputs=[],
            outputs=[helper.make_tensor_value_info("tokens", TensorProto.INT64, [None])],
        )
        return helper.make_model(graph), None

    monkeypatch.setattr(onnxruntime_extensions, "gen_processing_models", fake_gen_processing_models)

    with pytest.raises(ValueError, match="unexpected outputs"):
        _generate_tokenizer_model(object())


def test_generate_tokenizer_model_preserves_unexpected_errors(monkeypatch):
    import onnxruntime_extensions

    def fake_gen_processing_models(*args, **kwargs):
        raise RuntimeError("converter crashed")

    monkeypatch.setattr(onnxruntime_extensions, "gen_processing_models", fake_gen_processing_models)

    with pytest.raises(RuntimeError, match="converter crashed"):
        _generate_tokenizer_model(object())


@pytest.mark.slow
def test_build_augmented_miniLM_L6_shape_and_valid_onnx():
    spec = get_preset("all-MiniLM-L6-v2")
    data = build_augmented(spec)  # Uses default HF cache
    assert isinstance(data, bytes)
    assert len(data) > 1_000_000  # augmented ONNX is >1 MB
    model = onnx.load_from_string(data)
    onnx.checker.check_model(model)
    outs = list(model.graph.output)
    assert len(outs) == 1
    assert outs[0].name == "embedding"
    shape = outs[0].type.tensor_type.shape.dim
    assert len(shape) == 1
    assert shape[0].dim_value == 384


@pytest.mark.slow
def test_build_augmented_runs_end_to_end_on_cpu():
    """Load the augmented model in onnxruntime, feed a string, get a 384-d L2-normalized vec."""
    import numpy as np
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path

    spec = get_preset("all-MiniLM-L6-v2")
    data = build_augmented(spec)

    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    sess = ort.InferenceSession(data, sess_options=so, providers=["CPUExecutionProvider"])

    # Input name is "pre_text" because pipeline.py uses prefix1="pre_" when merging
    input_names = [i.name for i in sess.get_inputs()]
    assert "pre_text" in input_names, f"expected pre_text in inputs, got {input_names}"
    input_name = "pre_text"

    out = sess.run(None, {input_name: np.array(["hello world"])})[0]
    assert isinstance(out, np.ndarray)
    out_array = out
    assert out_array.shape == (384,)
    norm = float(np.linalg.norm(out_array))
    assert 0.99 < norm < 1.01, f"expected L2 norm ~1.0, got {norm}"


def test_splice_query_doc_subgraph_produces_bert_pair_layout():
    """Build a tiny graph: feed fake q_/d_ tokenizer outputs, run the splice in CPU
    onnxruntime, and assert the spliced ids / segments / mask are correct."""
    import numpy as np
    import onnxruntime as ort

    q_ids = helper.make_tensor_value_info("q_input_ids", TensorProto.INT64, [1, None])
    q_mask = helper.make_tensor_value_info("q_attention_mask", TensorProto.INT64, [1, None])
    d_ids = helper.make_tensor_value_info("d_input_ids", TensorProto.INT64, [1, None])
    d_mask = helper.make_tensor_value_info("d_attention_mask", TensorProto.INT64, [1, None])

    out_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, None])
    out_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, None])
    out_seg = helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, [1, None])

    graph = helper.make_graph(
        nodes=[],
        name="splice_only",
        inputs=[q_ids, q_mask, d_ids, d_mask],
        outputs=[out_ids, out_mask, out_seg],
    )
    _splice_query_doc_subgraph(graph)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    # q = [CLS=101, 1, 2, SEP=102] (length 4); d = [CLS=101, 7, 8, 9, SEP=102] (length 5).
    q = np.array([[101, 1, 2, 102]], dtype=np.int64)
    qm = np.array([[1, 1, 1, 1]], dtype=np.int64)
    d = np.array([[101, 7, 8, 9, 102]], dtype=np.int64)
    dm = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)

    ids, mask, seg = sess.run(None, {
        "q_input_ids": q, "q_attention_mask": qm,
        "d_input_ids": d, "d_attention_mask": dm,
    })
    # Expected: [CLS] 1 2 [SEP] 7 8 9 [SEP] (4 + 4 = 8 tokens; d's leading CLS dropped).
    assert ids.tolist() == [[101, 1, 2, 102, 7, 8, 9, 102]]
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1, 1, 1]]
    # Segments: zeros for the q span (length 4), ones for the d-tail (length 4).
    assert seg.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]


@pytest.mark.slow
def test_build_reranker_shape_and_valid_onnx():
    spec = get_preset("ms-marco-MiniLM-L-6-v2")
    data = build_reranker(spec)
    assert isinstance(data, bytes)
    assert len(data) > 1_000_000
    model = onnx.load_from_string(data)
    onnx.checker.check_model(model)

    input_names = {i.name for i in model.graph.input}
    assert input_names == {"pre_text_1", "pre_text_2"}

    outs = list(model.graph.output)
    assert len(outs) == 1
    assert outs[0].name == "logits"


@pytest.mark.slow
def test_build_reranker_runs_end_to_end_on_cpu():
    """Build a real ms-marco-MiniLM reranker and verify relevance ordering."""
    import numpy as np
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path

    spec = get_preset("ms-marco-MiniLM-L-6-v2")
    data = build_reranker(spec)

    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    sess = ort.InferenceSession(data, sess_options=so, providers=["CPUExecutionProvider"])

    input_names = [i.name for i in sess.get_inputs()]
    assert set(input_names) == {"pre_text_1", "pre_text_2"}

    query = np.array(["How many people live in Berlin?"])
    relevant = np.array(["Berlin has a population of 3.7 million inhabitants."])
    irrelevant = np.array(["Bananas are a popular tropical fruit."])

    r_score = sess.run(None, {"pre_text_1": query, "pre_text_2": relevant})[0]
    i_score = sess.run(None, {"pre_text_1": query, "pre_text_2": irrelevant})[0]
    assert float(r_score) > float(i_score), (
        f"reranker did not order docs correctly: relevant={r_score} irrelevant={i_score}"
    )
