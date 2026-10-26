"""Tests for the augmented ONNX pipeline.

Most tests run offline in well under a second: they use tiny synthetic ONNX graphs, or push a
miniature BERT (a 14-word vocabulary and a one-op "transformer") through the real build
functions, which exercises the real tokenizer op, the graph surgery and the onnxruntime check.

Tests marked ``slow`` download real models from HuggingFace (~30-60 s the first time, cached
under ~/.cache/huggingface/ afterwards) and compare the built graphs with the reference
``transformers`` implementation. None of the tests need Oracle.

Run with: pytest tests/test_pipeline.py -v -m slow
"""

import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, compose, helper, numpy_helper

import onnx2oracle.pipeline as pipeline
from onnx2oracle.pipeline import (
    _core_hidden_state,
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
from onnx2oracle.presets import EmbeddingSpec, RerankerSpec, get_preset

_BERT_TENSORS = ("input_ids", "attention_mask", "token_type_ids")
_CLS, _SEP = 101, 102


@pytest.fixture(autouse=True)
def build_cache(tmp_path, monkeypatch) -> Path:
    """Point the cache of built graphs at a per-test directory so tests never touch ~/.onnx2oracle."""
    path = tmp_path / "build-cache"
    monkeypatch.setenv(pipeline.CACHE_DIR_ENV, str(path))
    return path


def _runnable(graph: onnx.GraphProto, opset: int = 18) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    # helper.make_model defaults to the installed onnx release's IR version (13 on onnx 1.22),
    # which the onnxruntime build resolved for Python 3.10 (1.23.x) cannot load. The behavior
    # under test is IR-independent, so pin a version every supported ORT reads.
    model.ir_version = 10
    return model


def _session(model: onnx.ModelProto | bytes, *, extensions: bool = False) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if extensions:
        from onnxruntime_extensions import get_library_path

        options.register_custom_ops_library(get_library_path())
    data = model if isinstance(model, bytes) else model.SerializeToString()
    return ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


# --- Truncation keeps the closing [SEP] ---------------------------------------------


def _stub_tokenizer(prefix: str = "", names: tuple[str, ...] = _BERT_TENSORS) -> onnx.ModelProto:
    """Stand-in for the tokenizer subgraph: forwards each ``<prefix>src_<name>`` input to ``<prefix><name>``."""
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", [f"{prefix}src_{n}"], [f"{prefix}{n}"]) for n in names],
        name=f"{prefix}tokenizer",
        inputs=[helper.make_tensor_value_info(f"{prefix}src_{n}", TensorProto.INT64, [None]) for n in names],
        outputs=[helper.make_tensor_value_info(f"{prefix}{n}", TensorProto.INT64, [None]) for n in names],
    )
    return _runnable(graph)


def _wrapped_ids(length: int) -> list[int]:
    """Tokenizer-style ids for a text of ``length - 2`` tokens: ``[CLS] t1 ... tn [SEP]``."""
    return [_CLS, *range(1000, 1000 + length - 2), _SEP]


@pytest.mark.parametrize("length", [2, 5, 8, 9, 40])
def test_truncation_trims_over_long_sequences_but_keeps_the_closing_sep(length):
    max_length = 8
    model = _stub_tokenizer()
    _truncate_and_unsqueeze_tokenizer_outputs(model, list(_BERT_TENSORS), max_length=max_length)
    onnx.checker.check_model(model)

    ids = _wrapped_ids(length)
    feeds = {
        "src_input_ids": np.array(ids, dtype=np.int64),
        "src_attention_mask": np.ones(length, dtype=np.int64),
        "src_token_type_ids": np.zeros(length, dtype=np.int64),
    }
    got_ids, got_mask, got_types = _session(model).run(["input_ids", "attention_mask", "token_type_ids"], feeds)

    # Sequences that fit are untouched; longer ones become exactly max_length tokens ending in [SEP].
    expected = ids if length <= max_length else [*ids[: max_length - 1], _SEP]
    assert got_ids.tolist() == [expected]
    assert got_mask.tolist() == [[1] * len(expected)]
    assert got_types.tolist() == [[0] * len(expected)]


def test_truncation_declares_a_batched_dynamic_length_output():
    model = _stub_tokenizer(names=("input_ids",))

    _truncate_and_unsqueeze_tokenizer_outputs(model, ["input_ids"], max_length=7)

    dims = model.graph.output[0].type.tensor_type.shape.dim
    assert dims[0].dim_value == 1
    assert dims[1].dim_param == "sequence_length"


def test_truncation_runs_on_both_sides_of_a_merged_query_document_graph():
    """The reranker truncates query and document with different budgets inside one merged graph."""
    query = _stub_tokenizer(prefix="q_", names=("input_ids",))
    document = _stub_tokenizer(prefix="d_", names=("input_ids",))
    _truncate_and_unsqueeze_tokenizer_outputs(query, ["q_input_ids"], max_length=4, init_prefix="q_")
    _truncate_and_unsqueeze_tokenizer_outputs(document, ["d_input_ids"], max_length=6, init_prefix="d_")
    merged = compose.merge_models(query, document, io_map=[], prefix1="", prefix2="")

    q_ids, d_ids = _session(merged).run(
        ["q_input_ids", "d_input_ids"],
        {
            "q_src_input_ids": np.array(_wrapped_ids(10), dtype=np.int64),
            "d_src_input_ids": np.array(_wrapped_ids(10), dtype=np.int64),
        },
    )

    assert q_ids.tolist() == [[_CLS, 1000, 1001, _SEP]]
    assert d_ids.tolist() == [[_CLS, 1000, 1001, 1002, 1003, _SEP]]


def test_truncation_rejects_a_non_positive_max_length():
    with pytest.raises(ValueError, match="max_length must be >= 1"):
        _truncate_and_unsqueeze_tokenizer_outputs(_stub_tokenizer(), ["input_ids"], max_length=0)


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


def test_splice_query_doc_subgraph_produces_bert_pair_layout():
    """Build a tiny graph: feed fake q_/d_ tokenizer outputs, run the splice in CPU
    onnxruntime, and assert the spliced ids / segments / mask are correct."""
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
    model = _runnable(graph)
    onnx.checker.check_model(model)

    sess = _session(model)
    # q = [CLS=101, 1, 2, SEP=102] (length 4); d = [CLS=101, 7, 8, 9, SEP=102] (length 5).
    q = np.array([[101, 1, 2, 102]], dtype=np.int64)
    qm = np.array([[1, 1, 1, 1]], dtype=np.int64)
    d = np.array([[101, 7, 8, 9, 102]], dtype=np.int64)
    dm = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)

    ids_raw, mask_raw, seg_raw = sess.run(None, {
        "q_input_ids": q, "q_attention_mask": qm,
        "d_input_ids": d, "d_attention_mask": dm,
    })
    ids = cast(np.ndarray, ids_raw)
    mask = cast(np.ndarray, mask_raw)
    seg = cast(np.ndarray, seg_raw)
    # Expected: [CLS] 1 2 [SEP] 7 8 9 [SEP] (4 + 4 = 8 tokens; d's leading CLS dropped).
    assert ids.tolist() == [[101, 1, 2, 102, 7, 8, 9, 102]]
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1, 1, 1]]
    # Segments: zeros for the q span (length 4), ones for the d-tail (length 4).
    assert seg.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]


# --- Hidden-state output detection and dims validation --------------------------------


def _embedding_spec(**changes) -> EmbeddingSpec:
    spec = EmbeddingSpec(hf_repo="org/some-model", dims=384, pooling="mean", normalize=True, oracle_name="SOME_MODEL")
    return dataclasses.replace(spec, **changes)


def _core_with_outputs(*outputs: onnx.ValueInfoProto) -> onnx.ModelProto:
    return helper.make_model(helper.make_graph([], "core", [], list(outputs)))


def test_core_hidden_state_returns_the_detected_output_name():
    core = _core_with_outputs(
        helper.make_tensor_value_info("sentence_embedding", TensorProto.FLOAT, [1, 384]),
        helper.make_tensor_value_info("token_embeddings", TensorProto.FLOAT, [1, "seq", 384]),
    )

    assert _core_hidden_state(core, _embedding_spec()) == "token_embeddings"


def test_core_hidden_state_rejects_dims_that_contradict_the_model():
    core = _core_with_outputs(helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, "seq", 768]))

    with pytest.raises(ValueError, match=r"768-dimensional token embeddings.*dims=384"):
        _core_hidden_state(core, _embedding_spec(dims=384))


def test_core_hidden_state_accepts_a_dynamic_hidden_size():
    """With an unknown hidden size the early check cannot fail; the post-build run still verifies it."""
    hidden = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, "seq", "hidden"])
    core = _core_with_outputs(hidden)

    assert _core_hidden_state(core, _embedding_spec(dims=384)) == "last_hidden_state"


# --- Cache of built graphs ------------------------------------------------------------


class _FakeBuild:
    """Stands in for the expensive build step and counts how often it runs."""

    def __init__(self, payload: bytes = b"built-onnx"):
        self.payload = payload
        self.specs: list = []

    def __call__(self, spec, cache_dir):
        self.specs.append(spec)
        return self.payload

    @property
    def calls(self) -> int:
        return len(self.specs)


@pytest.fixture
def fake_embedding_build(monkeypatch) -> _FakeBuild:
    build = _FakeBuild()
    monkeypatch.setattr(pipeline, "_build_augmented_uncached", build)
    return build


def _set_hf_revision(hf_cache: Path, hf_repo: str, revision: str) -> None:
    """Record *revision* as the cached ``main`` of *hf_repo*, the way a HuggingFace download does."""
    ref = hf_cache / f"models--{hf_repo.replace('/', '--')}" / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision)


def test_second_build_of_the_same_spec_is_served_from_the_cache(fake_embedding_build, build_cache):
    spec = _embedding_spec()

    first = build_augmented(spec)
    second = build_augmented(spec)

    assert first == second == b"built-onnx"
    assert fake_embedding_build.calls == 1
    # One finished entry, no temp files left behind.
    assert [entry.suffix for entry in build_cache.iterdir()] == [".onnx"]


def test_use_cache_false_neither_reads_nor_writes_the_cache(fake_embedding_build, build_cache):
    spec = _embedding_spec()

    assert build_augmented(spec, use_cache=False) == b"built-onnx"
    assert not build_cache.exists()

    cached = build_augmented(spec)
    fake_embedding_build.payload = b"fresh-build"
    assert build_augmented(spec, use_cache=False) == b"fresh-build"
    # The bypassing build must not have replaced the stored entry.
    assert build_augmented(spec) == cached == b"built-onnx"
    assert fake_embedding_build.calls == 3


def test_cache_ignores_the_oracle_name_and_the_informational_size(fake_embedding_build):
    spec = _embedding_spec()

    build_augmented(spec)
    build_augmented(dataclasses.replace(spec, oracle_name="ANOTHER_NAME", approx_size_mb=1))

    assert fake_embedding_build.calls == 1


@pytest.mark.parametrize(
    "change",
    [{"max_length": 128}, {"dims": 768}, {"pooling": "cls"}, {"normalize": False}, {"hf_repo": "org/other-model"}],
    ids=lambda change: next(iter(change)),
)
def test_graph_changing_spec_fields_get_their_own_cache_entry(fake_embedding_build, change):
    spec = _embedding_spec()

    build_augmented(spec)
    build_augmented(dataclasses.replace(spec, **change))

    assert fake_embedding_build.calls == 2


def test_cache_entry_is_tied_to_the_locally_cached_hf_revision(fake_embedding_build, tmp_path):
    hf_cache = tmp_path / "hf"
    spec = _embedding_spec()
    _set_hf_revision(hf_cache, spec.hf_repo, "rev1")

    build_augmented(spec, cache_dir=hf_cache)
    build_augmented(spec, cache_dir=hf_cache)
    assert fake_embedding_build.calls == 1

    _set_hf_revision(hf_cache, spec.hf_repo, "rev2")  # the repo moved on and was downloaded again
    build_augmented(spec, cache_dir=hf_cache)
    assert fake_embedding_build.calls == 2

    _set_hf_revision(hf_cache, spec.hf_repo, "rev1")
    build_augmented(spec, cache_dir=hf_cache)
    assert fake_embedding_build.calls == 2


def test_a_first_build_that_downloads_the_repo_is_cached_under_the_revision_it_fetched(monkeypatch, tmp_path):
    hf_cache = tmp_path / "hf"
    spec = _embedding_spec()
    builds = []

    def downloading_build(spec, cache_dir):
        builds.append(spec)
        _set_hf_revision(hf_cache, spec.hf_repo, "rev1")  # the download is what creates refs/main
        return b"built-onnx"

    monkeypatch.setattr(pipeline, "_build_augmented_uncached", downloading_build)

    build_augmented(spec, cache_dir=hf_cache)
    build_augmented(spec, cache_dir=hf_cache)

    assert len(builds) == 1


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("__version__", "99.0.0"),
        ("_build_sources_digest", lambda: "edited-build-code"),
        ("_library_version", lambda name: "0.0.0"),
    ],
    ids=["package-version", "build-code", "library-versions"],
)
def test_cache_is_invalidated_when_the_build_code_or_toolchain_changes(
    fake_embedding_build, monkeypatch, target, value
):
    spec = _embedding_spec()
    build_augmented(spec)

    monkeypatch.setattr(pipeline, target, value)
    build_augmented(spec)

    assert fake_embedding_build.calls == 2


def test_a_cache_that_cannot_be_written_does_not_fail_the_build(fake_embedding_build, tmp_path, monkeypatch, caplog):
    not_a_directory = tmp_path / "plain-file"
    not_a_directory.write_text("")
    monkeypatch.setenv(pipeline.CACHE_DIR_ENV, str(not_a_directory / "cache"))

    with caplog.at_level(logging.WARNING, logger="onnx2oracle.pipeline"):
        assert build_augmented(_embedding_spec()) == b"built-onnx"

    assert "Could not write ONNX build cache entry" in caplog.text


def test_cache_entries_stay_inside_the_cache_dir_for_odd_repo_names(fake_embedding_build, build_cache):
    build_augmented(_embedding_spec(hf_repo="../escape/with space:and*chars"))

    (entry,) = build_cache.iterdir()
    assert entry.suffix == ".onnx"


def test_default_cache_lives_under_the_home_directory(fake_embedding_build, tmp_path, monkeypatch):
    monkeypatch.delenv(pipeline.CACHE_DIR_ENV)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    build_augmented(_embedding_spec())

    assert len(list((tmp_path / ".onnx2oracle" / "cache").glob("*.onnx"))) == 1


def test_cache_dir_override_expands_the_home_shortcut(fake_embedding_build, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(pipeline.CACHE_DIR_ENV, "~/custom-cache")

    build_augmented(_embedding_spec())

    assert len(list((tmp_path / "custom-cache").glob("*.onnx"))) == 1


def test_build_reranker_is_cached_and_requires_a_reranker_spec(monkeypatch):
    build = _FakeBuild()
    monkeypatch.setattr(pipeline, "_build_reranker_uncached", build)
    spec = cast(RerankerSpec, get_preset("ms-marco-MiniLM-L-6-v2"))

    build_reranker(spec)
    build_reranker(spec)
    assert build.calls == 1

    build_reranker(dataclasses.replace(spec, query_max_length=32))  # the query budget changes the graph
    assert build.calls == 2

    with pytest.raises(ValueError, match="spec.task == 'reranker'"):
        build_reranker(cast(RerankerSpec, _embedding_spec()))
    assert build.calls == 2


# --- Miniature BERT: the real build functions, offline ---------------------------------

_TINY_VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "foo", "bar", "baz", "qux", "quux", "the", "a",
]
_TINY_HIDDEN = 4
_TINY_MAX_LENGTH = 8
# 4, 8 and 14 tokens with [CLS]/[SEP]: below, exactly at and above _TINY_MAX_LENGTH.
_TINY_TEXTS = {
    "below the limit": "hello world",
    "at the limit": "foo bar baz qux quux the",
    "above the limit": "foo bar baz qux quux the a hello world foo bar baz",
}


@pytest.fixture(scope="module")
def tiny_repo(tmp_path_factory) -> Path:
    """A local 'repository' holding a 14-word WordPiece tokenizer that ``AutoTokenizer`` can load."""
    from transformers import BertTokenizerFast

    root = tmp_path_factory.mktemp("tiny-bert")
    vocab = root / "vocab.txt"
    vocab.write_text("\n".join(_TINY_VOCAB) + "\n")
    BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True).save_pretrained(root / "repo")
    return root / "repo"


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_repo):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tiny_repo))


def _tiny_table(columns: int) -> np.ndarray:
    return np.random.default_rng(7).normal(size=(len(_TINY_VOCAB), columns)).astype(np.float32)


@pytest.fixture
def serve_core_from_disk(monkeypatch, tmp_path):
    """Serve ``onnx/model.onnx`` from a local file instead of the HuggingFace Hub."""

    def install(core: onnx.ModelProto) -> None:
        core_path = tmp_path / "model.onnx"
        onnx.save(core, str(core_path))

        def fake_download(repo_id, filename, **kwargs):
            assert filename == "onnx/model.onnx"
            return str(core_path)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    return install


def _tiny_embedding_core(
    table: np.ndarray,
    output_name: str = "last_hidden_state",
    hidden_dim: int | str = _TINY_HIDDEN,
    pooled_output_first: bool = False,
) -> onnx.ModelProto:
    """A one-op "transformer": each position's hidden state is its token embedding plus the mean
    embedding of the whole sequence, so every position depends on every token, [SEP] included."""
    nodes = [
        helper.make_node("Gather", ["embedding_table", "input_ids"], ["token_states"]),
        helper.make_node("ReduceMean", ["token_states"], ["sequence_mean"], axes=[1], keepdims=1),
        helper.make_node("Add", ["token_states", "sequence_mean"], [output_name]),
    ]
    hidden_shape = ["batch_size", "sequence_length", hidden_dim]
    outputs = [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, hidden_shape)]
    if pooled_output_first:
        nodes.append(helper.make_node("ReduceMean", ["token_states"], ["sentence_embedding"], axes=[1], keepdims=0))
        pooled = helper.make_tensor_value_info("sentence_embedding", TensorProto.FLOAT, ["batch_size", hidden_dim])
        outputs.insert(0, pooled)
    graph = helper.make_graph(
        nodes,
        "tiny_core",
        [helper.make_tensor_value_info(n, TensorProto.INT64, ["batch_size", "sequence_length"]) for n in _BERT_TENSORS],
        outputs,
        initializer=[numpy_helper.from_array(table, "embedding_table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


def _tiny_embedding_spec(tiny_repo: Path, **changes) -> EmbeddingSpec:
    spec = EmbeddingSpec(
        hf_repo=str(tiny_repo),
        dims=_TINY_HIDDEN,
        pooling="mean",
        normalize=True,
        oracle_name="TINY_BERT",
        max_length=_TINY_MAX_LENGTH,
    )
    return dataclasses.replace(spec, **changes)


def _reference_tiny_embedding(tokenizer, table: np.ndarray, text: str, spec: EmbeddingSpec) -> np.ndarray:
    ids = tokenizer(text, truncation=True, max_length=spec.max_length)["input_ids"]
    states = table[ids]
    states = states + states.mean(axis=0)  # what _tiny_embedding_core computes
    pooled = states.mean(axis=0) if spec.pooling == "mean" else states[0]
    return pooled / np.linalg.norm(pooled) if spec.normalize else pooled


@pytest.mark.parametrize("pooling", ["mean", "cls"])
@pytest.mark.parametrize("text", _TINY_TEXTS.values(), ids=_TINY_TEXTS.keys())
def test_built_embedding_graph_matches_the_reference_for_short_and_over_long_text(
    serve_core_from_disk, tiny_repo, tiny_tokenizer, pooling, text
):
    table = _tiny_table(_TINY_HIDDEN)
    serve_core_from_disk(_tiny_embedding_core(table))
    spec = _tiny_embedding_spec(tiny_repo, pooling=pooling)

    data = build_augmented(spec, use_cache=False)

    ours = _session(data, extensions=True).run(None, {"pre_text": np.array([text])})[0]
    np.testing.assert_allclose(ours, _reference_tiny_embedding(tiny_tokenizer, table, text, spec), atol=1e-5)


def _embed(data: bytes, text: str) -> np.ndarray:
    """Run a built embedding graph on one text."""
    return _session(data, extensions=True).run(None, {"pre_text": np.array([text])})[0]


def test_built_embedding_graph_skips_normalization_when_disabled(serve_core_from_disk, tiny_repo, tiny_tokenizer):
    table = _tiny_table(_TINY_HIDDEN)
    serve_core_from_disk(_tiny_embedding_core(table))
    spec = _tiny_embedding_spec(tiny_repo, normalize=False)
    text = _TINY_TEXTS["above the limit"]

    ours = _embed(build_augmented(spec, use_cache=False), text)

    assert abs(float(np.linalg.norm(ours)) - 1.0) > 0.1  # a normalized vector would have norm 1
    np.testing.assert_allclose(ours, _reference_tiny_embedding(tiny_tokenizer, table, text, spec), atol=1e-5)


def test_built_embedding_graph_follows_the_cores_hidden_state_output_name(
    serve_core_from_disk, tiny_repo, tiny_tokenizer
):
    """A core exposing sentence-transformers style outputs (pooled vector first, 'token_embeddings' second)."""
    table = _tiny_table(_TINY_HIDDEN)
    serve_core_from_disk(_tiny_embedding_core(table, output_name="token_embeddings", pooled_output_first=True))
    spec = _tiny_embedding_spec(tiny_repo)
    text = _TINY_TEXTS["below the limit"]

    ours = _embed(build_augmented(spec, use_cache=False), text)

    np.testing.assert_allclose(ours, _reference_tiny_embedding(tiny_tokenizer, table, text, spec), atol=1e-5)


def test_build_augmented_rejects_dims_that_contradict_the_core_before_building(serve_core_from_disk, tiny_repo):
    serve_core_from_disk(_tiny_embedding_core(_tiny_table(_TINY_HIDDEN)))

    with pytest.raises(ValueError, match=r"4-dimensional token embeddings.*dims=5"):
        build_augmented(_tiny_embedding_spec(tiny_repo, dims=5), use_cache=False)


def test_build_augmented_checks_the_embedding_shape_when_the_core_hides_its_hidden_size(
    serve_core_from_disk, tiny_repo
):
    serve_core_from_disk(_tiny_embedding_core(_tiny_table(_TINY_HIDDEN), hidden_dim="hidden"))

    with pytest.raises(ValueError, match=r"embedding of shape \(4,\), expected \(5,\)"):
        build_augmented(_tiny_embedding_spec(tiny_repo, dims=5), use_cache=False)


def test_build_augmented_fails_on_a_graph_that_cannot_run_and_caches_nothing(
    serve_core_from_disk, tiny_repo, build_cache
):
    core = _tiny_embedding_core(_tiny_table(_TINY_HIDDEN))
    # An input the tokenizer cannot feed: Oracle would reject this graph at query time.
    core.graph.input.append(
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch_size", "sequence_length"])
    )
    serve_core_from_disk(core)

    with pytest.raises(RuntimeError, match="could not be run locally") as excinfo:
        build_augmented(_tiny_embedding_spec(tiny_repo))

    assert "position_ids" in str(excinfo.value)
    assert not list(build_cache.glob("*.onnx"))


_TINY_QUERY_BUDGET = 5
_TINY_PAIR_MAX_LENGTH = 12
_TINY_DOC_BUDGET = _TINY_PAIR_MAX_LENGTH - _TINY_QUERY_BUDGET


def _tiny_cross_encoder(
    token_scores: np.ndarray, segment_scores: np.ndarray, keep_sequence_axis: bool = False
) -> onnx.ModelProto:
    """A one-op "cross-encoder": the logits are the sum of per-token and per-segment scores, so they
    change whenever a [SEP] is lost or a token lands in the wrong segment. One column of scores means
    one output label; ``keep_sequence_axis`` leaves the reduced axis in place ([batch, 1, labels])."""
    labels = token_scores.shape[1]
    logits_shape = ["batch_size", 1, labels] if keep_sequence_axis else ["batch_size", labels]
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["token_scores", "input_ids"], ["token_part"]),
            helper.make_node("Gather", ["segment_scores", "token_type_ids"], ["segment_part"]),
            helper.make_node("Add", ["token_part", "segment_part"], ["both"]),
            helper.make_node("ReduceSum", ["both", "sum_axes"], ["logits"], keepdims=int(keep_sequence_axis)),
        ],
        "tiny_cross_encoder",
        [helper.make_tensor_value_info(n, TensorProto.INT64, ["batch_size", "sequence_length"]) for n in _BERT_TENSORS],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, logits_shape)],
        initializer=[
            numpy_helper.from_array(token_scores, "token_scores"),
            numpy_helper.from_array(segment_scores, "segment_scores"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "sum_axes"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


@pytest.fixture
def tiny_cross_encoder_scores(monkeypatch) -> tuple[np.ndarray, np.ndarray]:
    """Replace the PyTorch export of the cross-encoder with the miniature one."""
    token_scores = _tiny_table(1)
    segment_scores = np.array([[0.5], [-1.25]], dtype=np.float32)

    def fake_export(hf_repo, cache_dir, out_path):
        onnx.save(_tiny_cross_encoder(token_scores, segment_scores), str(out_path))

    monkeypatch.setattr(pipeline, "_export_cross_encoder_onnx", fake_export)
    return token_scores, segment_scores


def _tiny_reranker_spec(tiny_repo: Path) -> RerankerSpec:
    return RerankerSpec(
        hf_repo=str(tiny_repo),
        oracle_name="TINY_RERANKER",
        max_length=_TINY_PAIR_MAX_LENGTH,
        query_max_length=_TINY_QUERY_BUDGET,
    )


@pytest.mark.parametrize(
    ("query", "document"),
    [
        ("hello", "world"),
        ("foo bar baz qux quux the a", "hello"),
        ("hello", "foo bar baz qux quux the a hello world"),
        ("foo bar baz qux quux the a", "foo bar baz qux quux the a hello world"),
    ],
    ids=["both short", "long query", "long document", "both long"],
)
def test_built_reranker_graph_scores_the_pair_a_cross_encoder_would_see(
    tiny_cross_encoder_scores, tiny_repo, tiny_tokenizer, query, document
):
    token_scores, segment_scores = tiny_cross_encoder_scores
    data = build_reranker(_tiny_reranker_spec(tiny_repo), use_cache=False)

    feeds = {"pre_text_1": np.array([query]), "pre_text_2": np.array([document])}
    ours = float(_session(data, extensions=True).run(None, feeds)[0])

    # Query and document are cut to their budgets, each keeping its own [CLS]/[SEP]; HuggingFace
    # then lays out the pair as [CLS] q [SEP] d [SEP] with segment ids 0...0 1...1.
    q_ids = tiny_tokenizer(query, add_special_tokens=False)["input_ids"][: _TINY_QUERY_BUDGET - 2]
    d_ids = tiny_tokenizer(document, add_special_tokens=False)["input_ids"][: _TINY_DOC_BUDGET - 2]
    ids = tiny_tokenizer.build_inputs_with_special_tokens(q_ids, d_ids)
    segments = tiny_tokenizer.create_token_type_ids_from_sequences(q_ids, d_ids)
    expected = float(token_scores[ids].sum() + segment_scores[segments].sum())

    assert ours == pytest.approx(expected, abs=1e-4)


def test_build_reranker_rejects_a_cross_encoder_with_several_output_labels(monkeypatch, tiny_repo):
    def fake_export(hf_repo, cache_dir, out_path):
        two_labels = _tiny_cross_encoder(np.zeros((len(_TINY_VOCAB), 2), np.float32), np.zeros((2, 2), np.float32))
        onnx.save(two_labels, str(out_path))

    monkeypatch.setattr(pipeline, "_export_cross_encoder_onnx", fake_export)

    with pytest.raises(RuntimeError, match="could not be run locally"):
        build_reranker(_tiny_reranker_spec(tiny_repo), use_cache=False)


def test_build_reranker_checks_that_the_graph_returns_a_scalar(monkeypatch, tiny_repo):
    def fake_export(hf_repo, cache_dir, out_path):
        extra_axis = _tiny_cross_encoder(_tiny_table(1), np.zeros((2, 1), np.float32), keep_sequence_axis=True)
        onnx.save(extra_axis, str(out_path))

    monkeypatch.setattr(pipeline, "_export_cross_encoder_onnx", fake_export)

    with pytest.raises(ValueError, match=r"returned shape \(1,\), expected a scalar logit"):
        build_reranker(_tiny_reranker_spec(tiny_repo), use_cache=False)


@pytest.fixture
def scratch_tempdir(tmp_path, monkeypatch) -> Path:
    """Make every ``tempfile`` directory land in a directory the test can inspect afterwards."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return scratch


def test_reranker_build_leaves_no_export_directory_behind(tiny_cross_encoder_scores, tiny_repo, scratch_tempdir):
    build_reranker(_tiny_reranker_spec(tiny_repo), use_cache=False)

    assert list(scratch_tempdir.iterdir()) == []


def test_failed_reranker_export_leaves_no_export_directory_behind(monkeypatch, tiny_repo, scratch_tempdir):
    def failing_export(hf_repo, cache_dir, out_path):
        out_path.write_bytes(b"partial export")
        raise RuntimeError("export failed halfway")

    monkeypatch.setattr(pipeline, "_export_cross_encoder_onnx", failing_export)

    with pytest.raises(RuntimeError, match="export failed halfway"):
        build_reranker(_tiny_reranker_spec(tiny_repo), use_cache=False)

    assert list(scratch_tempdir.iterdir()) == []


# --- Real models (slow): parity with the reference transformers implementation ----------

_SHORT_TEXT = "The quick brown fox jumps over the lazy dog."
# 400 words that tokenize to ~1500 word pieces, far beyond the 512-token limit.
_LONG_WORDS = [
    "internationalization", "counterrevolutionary", "electroencephalography", "uncharacteristically",
    "disproportionately", "pharmacokinetics", "incomprehensibility", "telecommunications",
    "unconstitutional", "bioluminescence", "misunderstanding", "responsibilities",
]
_LONG_TEXT = " ".join(_LONG_WORDS[i % len(_LONG_WORDS)] for i in range(400))

_SHORT_QUERY = "How many people live in Berlin?"
_SHORT_DOCUMENT = "Berlin has a population of 3.7 million inhabitants."
_LONG_QUERY = (
    "How many people live in the city of Berlin in the year two thousand and twenty three, including everyone "
    "registered at a residential address as well as people without papers, tourists, commuters, students, short "
    "term visitors and everybody else who stays there "
) * 2
_LONG_DOCUMENT = (
    "Berlin has a population of about three point seven million inhabitants according to the official "
    "statistics office, which makes it the most populous city in the European Union. "
) * 25


@pytest.fixture(scope="module")
def minilm_spec() -> EmbeddingSpec:
    spec = get_preset("all-MiniLM-L6-v2")
    assert isinstance(spec, EmbeddingSpec)
    return spec


@pytest.fixture(scope="module")
def minilm_graph(minilm_spec) -> bytes:
    return build_augmented(minilm_spec, use_cache=False)  # Uses the default HF cache


@pytest.fixture(scope="module")
def minilm_session(minilm_graph) -> ort.InferenceSession:
    return _session(minilm_graph, extensions=True)


@pytest.fixture(scope="module")
def minilm_reference(minilm_spec):
    """The PyTorch model with the standard recipe: HF truncation, mean pooling, L2 normalization."""
    torch = pytest.importorskip("torch")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(minilm_spec.hf_repo)
    model = AutoModel.from_pretrained(minilm_spec.hf_repo).eval()

    def embed(text: str) -> np.ndarray:
        encoded = tokenizer(text, truncation=True, max_length=minilm_spec.max_length, return_tensors="pt")
        with torch.no_grad():
            pooled = model(**encoded).last_hidden_state.mean(dim=1)[0]
        return (pooled / pooled.norm()).numpy()

    return tokenizer, embed


@pytest.mark.slow
def test_build_augmented_miniLM_L6_shape_and_valid_onnx(minilm_graph):
    data = minilm_graph
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
def test_build_augmented_runs_end_to_end_on_cpu(minilm_session):
    """Load the augmented model in onnxruntime, feed a string, get a 384-d L2-normalized vec."""
    # Input name is "pre_text" because pipeline.py uses prefix1="pre_" when merging
    input_names = [i.name for i in minilm_session.get_inputs()]
    assert "pre_text" in input_names, f"expected pre_text in inputs, got {input_names}"

    out = minilm_session.run(None, {"pre_text": np.array(["hello world"])})[0]
    assert isinstance(out, np.ndarray)
    assert out.shape == (384,)
    norm = float(np.linalg.norm(out))
    assert 0.99 < norm < 1.01, f"expected L2 norm ~1.0, got {norm}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("text", "exceeds_max_length"),
    [(_SHORT_TEXT, False), (_LONG_TEXT, True)],
    ids=["short text", "text longer than max_length"],
)
def test_build_augmented_miniLM_matches_transformers_reference(
    minilm_spec, minilm_session, minilm_reference, text, exceeds_max_length
):
    tokenizer, embed = minilm_reference
    token_count = len(tokenizer(text, verbose=False)["input_ids"])
    assert (token_count > minilm_spec.max_length) == exceeds_max_length

    ours = minilm_session.run(None, {"pre_text": np.array([text])})[0]

    cosine = _cosine(ours, embed(text))
    # Before the closing [SEP] was kept on truncation, the long text scored ~0.9997 here.
    assert cosine > 0.99999, f"cosine to the transformers reference is {cosine:.8f} for {token_count} tokens"


@pytest.fixture(scope="module")
def reranker_spec() -> RerankerSpec:
    spec = get_preset("ms-marco-MiniLM-L-6-v2")
    assert isinstance(spec, RerankerSpec)
    return spec


@pytest.fixture(scope="module")
def reranker_graph(reranker_spec) -> bytes:
    pytest.importorskip("torch", reason="the reranker is exported through PyTorch (the onnx2oracle[export] extra)")
    return build_reranker(reranker_spec, use_cache=False)


@pytest.fixture(scope="module")
def reranker_session(reranker_graph) -> ort.InferenceSession:
    return _session(reranker_graph, extensions=True)


@pytest.fixture(scope="module")
def reranker_reference(reranker_spec):
    """The PyTorch cross-encoder fed the pair the graph is documented to build: the query cut to
    ``query_max_length`` tokens and the document to the rest of ``max_length``, each side keeping its
    own [CLS]/[SEP], laid out as [CLS] q [SEP] d [SEP]."""
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(reranker_spec.hf_repo)
    model = AutoModelForSequenceClassification.from_pretrained(reranker_spec.hf_repo).eval()
    query_budget = reranker_spec.query_max_length
    document_budget = reranker_spec.max_length - reranker_spec.query_max_length

    def token_counts(query: str, document: str) -> tuple[int, int]:
        return (
            len(tokenizer(query, add_special_tokens=False, verbose=False)["input_ids"]),
            len(tokenizer(document, add_special_tokens=False, verbose=False)["input_ids"]),
        )

    def score(query: str, document: str) -> float:
        q_ids = tokenizer(query, add_special_tokens=False, verbose=False)["input_ids"][: query_budget - 2]
        d_ids = tokenizer(document, add_special_tokens=False, verbose=False)["input_ids"][: document_budget - 2]
        ids = tokenizer.build_inputs_with_special_tokens(q_ids, d_ids)
        segments = tokenizer.create_token_type_ids_from_sequences(q_ids, d_ids)
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                token_type_ids=torch.tensor([segments]),
            ).logits
        return float(logits[0, 0])

    return token_counts, score, query_budget, document_budget


@pytest.mark.slow
def test_build_reranker_shape_and_valid_onnx(reranker_graph):
    data = reranker_graph
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
def test_build_reranker_runs_end_to_end_on_cpu(reranker_session):
    """Build a real ms-marco-MiniLM reranker and verify relevance ordering."""
    input_names = [i.name for i in reranker_session.get_inputs()]
    assert set(input_names) == {"pre_text_1", "pre_text_2"}

    query = np.array([_SHORT_QUERY])
    relevant = np.array([_SHORT_DOCUMENT])
    irrelevant = np.array(["Bananas are a popular tropical fruit."])

    r_score = cast(np.ndarray, reranker_session.run(None, {"pre_text_1": query, "pre_text_2": relevant})[0])
    i_score = cast(np.ndarray, reranker_session.run(None, {"pre_text_1": query, "pre_text_2": irrelevant})[0])
    assert float(r_score) > float(i_score), (
        f"reranker did not order docs correctly: relevant={r_score} irrelevant={i_score}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("query", "document", "query_too_long", "document_too_long"),
    [
        (_SHORT_QUERY, _SHORT_DOCUMENT, False, False),
        (_LONG_QUERY, _SHORT_DOCUMENT, True, False),
        (_SHORT_QUERY, _LONG_DOCUMENT, False, True),
        (_LONG_QUERY, _LONG_DOCUMENT, True, True),
    ],
    ids=["short pair", "long query", "long document", "long query and document"],
)
def test_build_reranker_matches_transformers_reference(
    reranker_session, reranker_reference, query, document, query_too_long, document_too_long
):
    token_counts, score, query_budget, document_budget = reranker_reference
    q_tokens, d_tokens = token_counts(query, document)
    assert (q_tokens > query_budget - 2) == query_too_long
    assert (d_tokens > document_budget - 2) == document_too_long

    ours = float(reranker_session.run(None, {"pre_text_1": np.array([query]), "pre_text_2": np.array([document])})[0])

    # Before each side kept its closing [SEP] on truncation, the long variants were off by 0.06-0.44.
    assert ours == pytest.approx(score(query, document), abs=1e-3)
