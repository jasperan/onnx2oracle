"""Augmented ONNX pipeline builder.

Two graph shapes are produced:

1. ``build_augmented`` — sentence-transformer embedding pipeline. One string
   input ``pre_text``; one ``embedding`` float32[dims] output. Loaded with
   ``function:"embedding"`` metadata, queried via ``VECTOR_EMBEDDING``.

2. ``build_reranker`` — cross-encoder reranker pipeline. Two string inputs
   ``pre_text_1`` / ``pre_text_2``; one scalar ``logits`` float32 output.
   Loaded with ``function:"regression"`` metadata, queried via
   ``PREDICTION(model USING q AS DATA1, d AS DATA2)``.

Both share the same downstream Oracle loader and identifier-safety machinery.

Every built graph is run once through onnxruntime (with the onnxruntime-extensions
custom ops) before it is returned, so a graph that cannot run fails at build time
rather than at Oracle load time. The finished bytes are cached on disk in
``~/.onnx2oracle/cache/`` (or the directory named by ``ONNX2ORACLE_CACHE_DIR``), so
rebuilding the same spec is instant; ``use_cache=False`` bypasses the cache.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Callable
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper, version_converter

from onnx2oracle import __version__
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
from onnx2oracle.presets import EmbeddingSpec, ModelSpec, RerankerSpec

logger = logging.getLogger(__name__)

CACHE_DIR_ENV = "ONNX2ORACLE_CACHE_DIR"

# Spec fields that cannot change the built graph: the Oracle name only matters at load
# time and the size is informational. Every other field is part of the cache key.
_CACHE_KEY_IGNORED_FIELDS = frozenset({"oracle_name", "approx_size_mb"})
# Libraries whose version can change the bytes of the emitted graph.
_CACHE_KEY_LIBRARIES = ("onnx", "onnxruntime-extensions", "transformers")

_SMOKE_TEXT = "hello world"
_INT64_MAX = int(np.iinfo(np.int64).max)


def _external_data_locations(model: onnx.ModelProto) -> list[str]:
    """Return external-data sidecar paths referenced by an ONNX model."""
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value:
                locations.add(entry.value)
    return sorted(locations)


def _external_data_repo_path(location: str) -> str:
    path = PurePosixPath(location)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe ONNX external data location: {location!r}")
    return str(PurePosixPath("onnx") / path)


def _require_torch_for_export_fallback() -> None:
    if find_spec("torch") is None:
        raise RuntimeError(
            "This HuggingFace repository does not provide onnx/model.onnx, so onnx2oracle "
            "must export the transformer from PyTorch. Install the optional export extra "
            'with `pip install "onnx2oracle[export]"`, then retry.'
        )


def _should_use_export_fallback(exc: Exception) -> bool:
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    return isinstance(exc, EntryNotFoundError) and not isinstance(exc, LocalEntryNotFoundError)


def _generate_tokenizer_model(tokenizer: Any) -> onnx.ModelProto:
    from onnxruntime_extensions import gen_processing_models

    pre_model_raw, _ = gen_processing_models(
        tokenizer, pre_kwargs={}, post_kwargs=cast(Any, None), opset=14
    )
    pre_model = cast(onnx.ModelProto, pre_model_raw)
    pre_output_names = {o.name for o in pre_model.graph.output}
    if "input_ids" not in pre_output_names:
        raise ValueError(
            f"gen_processing_models produced unexpected outputs: {pre_output_names}"
        )
    return pre_model


def _truncate_and_unsqueeze_tokenizer_outputs(
    pre_model: onnx.ModelProto,
    names: list[str],
    max_length: int,
    init_prefix: str = "",
) -> None:
    """Truncate 1-D tokenizer outputs to ``max_length``, then add the transformer batch dimension.

    The tokenizer wraps every text as ``[CLS] ... [SEP]``. Cutting that with a bare
    ``[:max_length]`` slice would drop the closing ``[SEP]`` of an over-long text, whereas
    HuggingFace truncation shortens the text and keeps both special tokens. The nodes added
    here compute ``raw[:-1][:max_length - 1] + raw[-1:]``: a sequence that fits passes through
    unchanged, a longer one becomes exactly ``max_length`` tokens with the original closing
    token restored at the end (``[SEP]`` in ``input_ids``, the matching 1 / 0 in
    ``attention_mask`` / ``token_type_ids``).

    ``init_prefix`` lets the helper run twice in one graph (q-side and d-side
    of a reranker pipeline) without colliding on shared initializer names.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")

    def constant(suffix: str, values: list[int]) -> str:
        name = f"{init_prefix}{suffix}"
        pre_model.graph.initializer.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name=name))
        return name

    starts = constant("truncate_starts", [0])
    axes = constant("truncate_axes", [0])
    steps = constant("truncate_steps", [1])
    drop_last_end = constant("truncate_drop_last_end", [-1])
    head_end = constant("truncate_head_end", [max_length - 1])
    last_start = constant("truncate_last_start", [-1])
    last_end = constant("truncate_last_end", [_INT64_MAX])
    unsqueeze_axes = constant("unsqueeze_axes_0", [0])

    for name in names:
        raw_name = f"{name}_raw"
        body_name = f"{name}_body"
        head_name = f"{name}_head"
        last_name = f"{name}_last"
        flat_name = f"{name}_flat"
        for node in pre_model.graph.node:
            for i, out in enumerate(node.output):
                if out == name:
                    node.output[i] = raw_name
        pre_model.graph.node.extend(
            [
                helper.make_node("Slice", [raw_name, starts, drop_last_end, axes, steps], [body_name]),
                helper.make_node("Slice", [body_name, starts, head_end, axes, steps], [head_name]),
                helper.make_node("Slice", [raw_name, last_start, last_end, axes, steps], [last_name]),
                helper.make_node("Concat", [head_name, last_name], [flat_name], axis=0),
                helper.make_node("Unsqueeze", [flat_name, unsqueeze_axes], [name]),
            ]
        )
        for out in pre_model.graph.output:
            if out.name == name:
                shape = out.type.tensor_type.shape
                while len(shape.dim) > 0:
                    shape.dim.pop()
                shape.dim.add().dim_value = 1
                shape.dim.add().dim_param = "sequence_length"


# --- Cache of built graphs ---------------------------------------------------------


def _build_cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".onnx2oracle" / "cache"


def _cached_hf_revision(hf_repo: str, hf_cache_dir: Path | None) -> str | None:
    """Commit hash of the ``main`` revision of *hf_repo* in the local HuggingFace cache, if there is one.

    Read from the cache's ``refs/main`` file, so it never touches the network.
    """
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name

    root = Path(hf_cache_dir) if hf_cache_dir is not None else Path(constants.HF_HUB_CACHE)
    try:
        ref = root / repo_folder_name(repo_id=hf_repo, repo_type="model") / "refs" / "main"
        return ref.read_text().strip() or None
    except (ValueError, OSError):  # not a valid hub repo id, or no such repo in the local cache
        return None


def _library_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _build_sources_digest() -> str:
    """Digest of the graph-building code, so editing it invalidates cached graphs even without a version bump."""
    digest = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).with_name("graph_stages.py")):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _cache_key(spec: ModelSpec, hf_cache_dir: Path | None) -> str:
    payload = {
        "spec": {k: v for k, v in dataclasses.asdict(spec).items() if k not in _CACHE_KEY_IGNORED_FIELDS},
        "onnx2oracle": __version__,
        "sources": _build_sources_digest(),
        "libraries": {name: _library_version(name) for name in _CACHE_KEY_LIBRARIES},
        "hf_revision": _cached_hf_revision(spec.hf_repo, hf_cache_dir),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_file(spec: ModelSpec, key: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", spec.hf_repo.replace("/", "--"))
    return _build_cache_dir() / f"{slug}-{key}.onnx"


def _read_cached_build(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        logger.warning("Ignoring unreadable ONNX build cache entry %s: %s", path, exc)
        return None


def _write_cached_build(path: Path, data: bytes) -> None:
    """Atomically store *data* at *path*. A cache that cannot be written never fails the build."""
    tmp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f"{path.name}.", suffix=".tmp", delete=False
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(data)
        os.replace(tmp_name, path)
        tmp_name = None
    except OSError as exc:
        logger.warning("Could not write ONNX build cache entry %s: %s", path, exc)
    finally:
        if tmp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)


def _build_or_load(
    spec: ModelSpec,
    hf_cache_dir: Path | None,
    use_cache: bool,
    build: Callable[[Any, Path | None], bytes],
) -> bytes:
    if not use_cache:
        return build(spec, hf_cache_dir)

    cached = _read_cached_build(_cache_file(spec, _cache_key(spec, hf_cache_dir)))
    if cached is not None:
        logger.info("Reusing cached ONNX build for %s", spec.hf_repo)
        return cached

    data = build(spec, hf_cache_dir)
    # Derive the key again: a first build downloads the repo, which is what makes its
    # HuggingFace revision known locally, so the key may have changed during the build.
    _write_cached_build(_cache_file(spec, _cache_key(spec, hf_cache_dir)), data)
    return data


# --- Local validation of built graphs ----------------------------------------------


def _core_hidden_state(core_model: onnx.ModelProto, spec: EmbeddingSpec) -> str:
    """Name of the core's token-level output; fails early when its hidden size contradicts ``spec.dims``."""
    output = hidden_state_output(core_model)
    size = hidden_size(output)
    if size is not None and size != spec.dims:
        raise ValueError(
            f"{spec.hf_repo} produces {size}-dimensional token embeddings (output {output.name!r}), "
            f"but the spec declares dims={spec.dims}. Use dims={size} (--dims {size}) for this model."
        )
    return output.name


def _run_locally(data: bytes, feeds: dict[str, np.ndarray], label: str) -> np.ndarray:
    """Run the built graph once in onnxruntime (CPU + onnxruntime-extensions ops); returns its first output."""
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path

    try:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.register_custom_ops_library(get_library_path())
        session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
        return np.asarray(session.run(None, feeds)[0])
    except Exception as exc:
        raise RuntimeError(
            f"The ONNX graph built for {label} could not be run locally with onnxruntime: {exc}"
        ) from exc


def _require_finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"The ONNX graph built for {label} returned NaN or infinite values for a test input.")


def _validate_embedding_graph(data: bytes, spec: EmbeddingSpec) -> None:
    """Run a test sentence through the built embedding graph and check the vector it returns."""
    embedding = _run_locally(data, {"pre_text": np.array([_SMOKE_TEXT])}, spec.hf_repo)
    if embedding.shape != (spec.dims,):
        raise ValueError(
            f"The ONNX graph built for {spec.hf_repo} returned an embedding of shape {embedding.shape}, "
            f"expected ({spec.dims},). dims must equal the model's hidden size."
        )
    _require_finite(embedding, spec.hf_repo)
    if spec.normalize:
        norm = float(np.linalg.norm(embedding))
        if not math.isclose(norm, 1.0, abs_tol=1e-3):
            raise RuntimeError(
                f"The ONNX graph built for {spec.hf_repo} returned an embedding with L2 norm {norm:.4f}, "
                "expected 1.0 because normalization is enabled."
            )


def _validate_reranker_graph(data: bytes, spec: RerankerSpec) -> None:
    """Run a test pair through the built reranker graph and check that it returns one scalar logit."""
    feeds = {"pre_text_1": np.array([_SMOKE_TEXT]), "pre_text_2": np.array([_SMOKE_TEXT])}
    logit = _run_locally(data, feeds, spec.hf_repo)
    if logit.shape != ():
        raise ValueError(
            f"The ONNX reranker graph built for {spec.hf_repo} returned shape {logit.shape}, "
            "expected a scalar logit (a cross-encoder with a single output label)."
        )
    _require_finite(logit, spec.hf_repo)


# --- Embedding pipeline ------------------------------------------------------------


def _hf_cache_kwargs(cache_dir: Path | None) -> dict[str, Any]:
    return {} if cache_dir is None else {"cache_dir": str(cache_dir)}


def _load_embedding_core(hf_repo: str, cache_kwargs: dict[str, Any]) -> onnx.ModelProto:
    """Load the transformer body: the repo's ``onnx/model.onnx``, or a PyTorch export when it has none."""
    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        core_path = hf_hub_download(hf_repo, "onnx/model.onnx", **cache_kwargs)
    except Exception as exc:
        if not _should_use_export_fallback(exc):
            raise
        _require_torch_for_export_fallback()
        # Fallback: export from PyTorch (slower but universal)
        from transformers import AutoModel, AutoTokenizer
        from transformers.onnx.convert import export as hf_onnx_export
        from transformers.onnx.features import FeaturesManager

        snap = Path(snapshot_download(hf_repo, **cache_kwargs))
        tokenizer_pt = AutoTokenizer.from_pretrained(snap)
        model_pt = AutoModel.from_pretrained(snap)
        _model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model_pt, feature="default"
        )
        config = model_onnx_config(model_pt.config)
        with tempfile.TemporaryDirectory(prefix="onnx2oracle_export_") as export_dir:
            export_path = Path(export_dir) / "model.onnx"
            hf_onnx_export(
                preprocessor=tokenizer_pt, model=model_pt, config=config, opset=14, output=export_path
            )
            # onnx.load inlines any external data, so the export directory can go afterwards.
            return onnx.load(str(export_path))

    # Some repos store large tensors beside model.onnx as external data.
    # Probe without loading sidecars, then fetch the exact files referenced.
    core_probe = onnx.load(core_path, load_external_data=False)
    for location in _external_data_locations(core_probe):
        repo_path = _external_data_repo_path(location)
        logger.info("Downloading ONNX external data sidecar %s", repo_path)
        hf_hub_download(hf_repo, repo_path, **cache_kwargs)

    return onnx.load(core_path)


def build_augmented(spec: EmbeddingSpec, cache_dir: Path | None = None, use_cache: bool = True) -> bytes:
    """Build the augmented ONNX pipeline for *spec* and return it as bytes.

    Graph shape: string -> tokenizer -> transformer -> pool -> l2-normalize -> [dims] float32.

    *cache_dir* is the HuggingFace download cache. *use_cache* controls the separate cache of
    finished graphs (see the module docstring): when ``False`` the graph is always rebuilt and
    that cache is neither read nor written.
    """
    return _build_or_load(spec, cache_dir, use_cache, _build_augmented_uncached)


def _build_augmented_uncached(spec: EmbeddingSpec, cache_dir: Path | None) -> bytes:
    from transformers import AutoTokenizer

    cache_kwargs = _hf_cache_kwargs(cache_dir)

    # 1) Load the core transformer ONNX (prefer pre-exported onnx/model.onnx) and find its
    # token-level output; a hidden size that contradicts spec.dims fails here, before the
    # tokenizer and graph surgery.
    core_model = _load_embedding_core(spec.hf_repo, cache_kwargs)
    hidden_state = _core_hidden_state(core_model, spec)

    # 2) Generate tokenizer ONNX
    # Some tokenizer classes (MPNetTokenizer, XLMRobertaTokenizer) are not
    # supported by gen_processing_models, or produce non-BERT output names
    # (tokens/instance_indices/token_indices instead of input_ids/...).
    # Fallback: BertTokenizerFast shares the WordPiece vocabulary and produces
    # the standard input_ids/token_type_ids/attention_mask outputs that the
    # core transformer expects.
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, **cache_kwargs)
    try:
        pre_model = _generate_tokenizer_model(tokenizer)
    except ValueError as _tok_err:
        logger.warning(
            "Tokenizer %s not directly supported by gen_processing_models (%s); "
            "falling back to BertTokenizerFast.",
            type(tokenizer).__name__,
            _tok_err,
        )
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(spec.hf_repo, **cache_kwargs)
        # Guard: BertTokenizerFast requires a WordPiece vocab with [UNK].
        # SentencePiece-based models (XLM-R, T5, mE5) use <unk> instead and
        # will produce an ONNX that Oracle rejects at load time.
        if "[UNK]" not in tokenizer.get_vocab():
            raise NotImplementedError(
                f"{spec.hf_repo} uses a SentencePiece/Unigram tokenizer "
                f"({type(tokenizer).__name__}) which cannot be represented as a "
                f"BertTokenizer ONNX graph compatible with Oracle's DBMS_VECTOR. "
                f"Use a model with a WordPiece vocabulary (BertTokenizer family)."
            ) from None
        pre_model = _generate_tokenizer_model(tokenizer)

    # 3) Align opsets — bump core to 18 + copy custom domains from pre
    core_model = version_converter.convert_version(core_model, 18)
    core_model.ir_version = pre_model.ir_version
    copy_missing_opset_domains(core_model, pre_model)

    # 4) Truncate to max_length (keeping the closing [SEP]), then unsqueeze the
    # tokenizer outputs: [seq_len] -> [1, seq_len]
    # Only process the tensor names that actually exist in the pre-model outputs
    # (e.g. MPNet has no token_type_ids input in its core model).
    pre_output_names = {o.name for o in pre_model.graph.output}
    core_input_names = {i.name for i in core_model.graph.input}
    bert_tensor_names = [n for n in ("input_ids", "token_type_ids", "attention_mask")
                         if n in pre_output_names]

    _truncate_and_unsqueeze_tokenizer_outputs(pre_model, bert_tensor_names, spec.max_length)

    # 5) Merge pre + core — only connect tensors present in both graphs
    io_map = [
        (n, n) for n in ("input_ids", "attention_mask", "token_type_ids")
        if n in pre_output_names and n in core_input_names
    ]
    merged = compose.merge_models(
        pre_model, core_model, io_map=io_map, prefix1="pre_", prefix2="core_"
    )

    clear_outputs(merged.graph)

    # 6) Pool (merge_models prefixed the core's tensors with "core_")
    pool_out = add_embedding_pooling(merged.graph, spec.pooling, f"core_{hidden_state}")

    # 7) L2 normalize (optional)
    final_2d = add_l2_normalization(merged.graph, pool_out) if spec.normalize else pool_out

    # 8) Squeeze batch dim: [1, dims] -> [dims]
    expose_squeezed_float_output(
        merged.graph,
        final_2d,
        "embedding",
        axes=[0],
        shape=[spec.dims],
        axes_initializer_name="squeeze_axes_final",
    )

    # Fix leftover dynamic batch dims in value_info / inputs
    pin_dynamic_batch_to_one(merged.graph)

    # Serialize. The merged proto has no external-data refs (onnx.load inlined
    # any sidecars), so SerializeToString covers the whole job.
    data = merged.SerializeToString()
    _validate_embedding_graph(data, spec)
    return data


# --- Reranker pipeline -------------------------------------------------------------


def _export_cross_encoder_onnx(hf_repo: str, cache_dir: Path | None, out_path: Path) -> None:
    """Export a HuggingFace cross-encoder to ONNX via PyTorch.

    Cross-encoder repos rarely ship a pre-exported ``onnx/model.onnx``, so we
    always go through ``transformers.onnx`` with feature
    ``"sequence-classification"`` (BertForSequenceClassification head with one
    output label producing ``logits[batch, 1]``).
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.onnx.convert import export as hf_onnx_export
    from transformers.onnx.features import FeaturesManager

    _require_torch_for_export_fallback()

    cache_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        cache_kwargs["cache_dir"] = str(cache_dir)

    snap = Path(snapshot_download(hf_repo, **cache_kwargs))
    tokenizer_pt = AutoTokenizer.from_pretrained(snap)
    model_pt = AutoModelForSequenceClassification.from_pretrained(snap)
    _model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model_pt, feature="sequence-classification"
    )
    config = model_onnx_config(model_pt.config)
    hf_onnx_export(
        preprocessor=tokenizer_pt, model=model_pt, config=config, opset=14, output=out_path
    )


def _bert_special_ids(tokenizer: Any) -> tuple[int, int, int]:
    """Return (cls_id, sep_id, pad_id) for a WordPiece BERT tokenizer.

    Cross-encoders are trained with the layout ``[CLS] q [SEP] d [SEP]`` and
    segment ids ``0...0 1...1``. We need the actual integer ids to bake them
    into the splice graph as initializers.
    """
    cls = tokenizer.cls_token_id
    sep = tokenizer.sep_token_id
    pad = tokenizer.pad_token_id
    if cls is None or sep is None or pad is None:
        raise NotImplementedError(
            "Reranker pipeline requires a tokenizer with [CLS], [SEP] and [PAD] tokens. "
            f"Tokenizer {type(tokenizer).__name__} did not provide all three."
        )
    return int(cls), int(sep), int(pad)


def _make_tokenizer_subgraph(
    tokenizer: Any, prefix: str, input_name: str
) -> tuple[onnx.ModelProto, list[str]]:
    """Generate a BERT tokenizer ONNX subgraph and prefix every tensor name.

    The single string input is renamed to ``input_name`` (caller chooses
    ``pre_text_1`` / ``pre_text_2``). All other tensor names get ``prefix``
    prepended so the q-side and d-side subgraphs can coexist in one merged
    graph without collisions.

    Returns the renamed model and the list of output names produced
    (``{prefix}input_ids`` etc.).
    """
    model = _generate_tokenizer_model(tokenizer)
    output_names = {o.name for o in model.graph.output}

    original_input_name = model.graph.input[0].name
    rename: dict[str, str] = {original_input_name: input_name}

    def _rn(name: str) -> str:
        if not name:
            return name
        if name not in rename:
            rename[name] = f"{prefix}{name}"
        return rename[name]

    for init in model.graph.initializer:
        init.name = _rn(init.name)
    for node in model.graph.node:
        node.name = f"{prefix}{node.name}" if node.name else node.name
        for i, n in enumerate(node.input):
            node.input[i] = _rn(n)
        for i, n in enumerate(node.output):
            node.output[i] = _rn(n)
    for inp in model.graph.input:
        inp.name = _rn(inp.name)
    for outp in model.graph.output:
        outp.name = _rn(outp.name)
    for vi in model.graph.value_info:
        vi.name = _rn(vi.name)

    # gen_processing_models declares only its custom domain. The truncate/unsqueeze
    # nodes we'll append use standard ONNX ops, so make sure the default domain is
    # in opset_import to keep the model checker happy before merging with the core.
    standard_domains = {o.domain for o in model.opset_import}
    if "" not in standard_domains:
        new_o = model.opset_import.add()
        new_o.domain = ""
        new_o.version = 14

    bert_outputs = [
        f"{prefix}{n}" for n in ("input_ids", "token_type_ids", "attention_mask")
        if n in output_names
    ]
    return model, bert_outputs


def _splice_query_doc_subgraph(graph: onnx.GraphProto) -> None:
    """Splice q_/d_ tokenizer outputs into BERT pair format ``[CLS] q [SEP] d [SEP]``.

    Inputs already present in the graph (shape ``[1, L]``):
      ``q_input_ids``, ``q_attention_mask``,
      ``d_input_ids``, ``d_attention_mask``.

    Outputs added (shape ``[1, L_q + L_d - 1]`` — variable; the transformer
    body accepts dynamic sequence length):
      ``input_ids``, ``attention_mask``, ``token_type_ids``.

    The exported BertForSequenceClassification graph accepts a dynamic sequence
    dimension, so no padding to a fixed ``max_length`` is needed — we just
    drop the document's leading ``[CLS]`` (since the query already starts with
    one), concat, and build the segment-id row 0...0 1...1 from the q and d
    lengths.
    """
    inits = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="splice_drop_cls_starts"),
        numpy_helper.from_array(
            np.array([np.iinfo(np.int64).max], dtype=np.int64), name="splice_drop_cls_ends"
        ),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="splice_drop_cls_axes"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="splice_drop_cls_steps"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="splice_axis_0"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="splice_one_scalar"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="splice_one_1d"),
    ]
    graph.initializer.extend(inits)

    nodes = [
        # BertTokenizer wraps each single-text input with [CLS]...[SEP], so the
        # raw q_* = [CLS] q [SEP] and raw d_* = [CLS] d [SEP]. The cross-encoder
        # expects [CLS] q [SEP] d [SEP]. We achieve that by dropping the leading
        # [CLS] from the document side only, and concatenating the rest.
        helper.make_node(
            "Slice",
            ["d_input_ids",
             "splice_drop_cls_starts",
             "splice_drop_cls_ends",
             "splice_drop_cls_axes",
             "splice_drop_cls_steps"],
            ["d_input_ids_tail"],
        ),
        helper.make_node(
            "Slice",
            ["d_attention_mask",
             "splice_drop_cls_starts",
             "splice_drop_cls_ends",
             "splice_drop_cls_axes",
             "splice_drop_cls_steps"],
            ["d_attention_mask_tail"],
        ),
        helper.make_node(
            "Concat", ["q_input_ids", "d_input_ids_tail"], ["input_ids"], axis=1
        ),
        helper.make_node(
            "Concat", ["q_attention_mask", "d_attention_mask_tail"], ["attention_mask"], axis=1
        ),
        # token_type_ids: zeros for the q span (length = q seq_len), ones for the d-tail.
        helper.make_node("Shape", ["q_input_ids"], ["q_shape"]),
        helper.make_node("Gather", ["q_shape", "splice_one_scalar"], ["q_len_scalar"], axis=0),
        helper.make_node("Unsqueeze", ["q_len_scalar", "splice_axis_0"], ["q_len_1d"]),
        helper.make_node(
            "Concat", ["splice_one_1d", "q_len_1d"], ["q_shape_1xL"], axis=0
        ),
        helper.make_node(
            "ConstantOfShape",
            ["q_shape_1xL"],
            ["q_segment_zeros"],
            value=helper.make_tensor(
                name="cos_q_seg", data_type=TensorProto.INT64, dims=[1], vals=[0]
            ),
        ),
        helper.make_node("Shape", ["d_input_ids_tail"], ["d_tail_shape"]),
        helper.make_node(
            "Gather", ["d_tail_shape", "splice_one_scalar"], ["d_tail_len_scalar"], axis=0
        ),
        helper.make_node(
            "Unsqueeze", ["d_tail_len_scalar", "splice_axis_0"], ["d_tail_len_1d"]
        ),
        helper.make_node(
            "Concat", ["splice_one_1d", "d_tail_len_1d"], ["d_shape_1xL"], axis=0
        ),
        helper.make_node(
            "ConstantOfShape",
            ["d_shape_1xL"],
            ["d_segment_ones"],
            value=helper.make_tensor(
                name="cos_d_seg", data_type=TensorProto.INT64, dims=[1], vals=[1]
            ),
        ),
        helper.make_node(
            "Concat", ["q_segment_zeros", "d_segment_ones"], ["token_type_ids"], axis=1
        ),
    ]
    graph.node.extend(nodes)


def build_reranker(spec: RerankerSpec, cache_dir: Path | None = None, use_cache: bool = True) -> bytes:
    """Build a cross-encoder reranker ONNX graph and return its bytes.

    Graph shape: ``(pre_text_1: string[1], pre_text_2: string[1])`` →
    tokenizer → splice → BertForSequenceClassification → scalar ``logits``.

    The resulting graph is loaded into Oracle with ``function:"regression"``
    metadata (see ``loader.build_metadata_json("reranker")``) and queried via
    ``PREDICTION(model USING q AS DATA1, d AS DATA2)``.

    *cache_dir* is the HuggingFace download cache. *use_cache* controls the separate cache of
    finished graphs (see the module docstring): when ``False`` the graph is always rebuilt and
    that cache is neither read nor written.
    """
    if spec.task != "reranker":
        raise ValueError(
            f"build_reranker requires spec.task == 'reranker', got {spec.task!r}"
        )

    return _build_or_load(spec, cache_dir, use_cache, _build_reranker_uncached)


def _build_reranker_uncached(spec: RerankerSpec, cache_dir: Path | None) -> bytes:
    from transformers import AutoTokenizer, BertTokenizerFast

    cache_kwargs = _hf_cache_kwargs(cache_dir)

    # 1) Export the cross-encoder body from PyTorch.
    with tempfile.TemporaryDirectory(prefix="onnx2oracle_rerank_") as export_dir:
        core_path = Path(export_dir) / "model.onnx"
        _export_cross_encoder_onnx(spec.hf_repo, cache_dir, core_path)
        # onnx.load inlines any external data, so the export directory can go afterwards.
        core_model = onnx.load(str(core_path))

    # 2) Tokenizer — require WordPiece (cross-encoder/ms-marco-MiniLM family is BERT-based).
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, **cache_kwargs)
    if "[UNK]" not in tokenizer.get_vocab():
        # SentencePiece tokenizers (XLM-R, e.g. BAAI/bge-reranker-base) can't be expressed
        # as Oracle-compatible BertTokenizer ONNX — same constraint as the embedding path.
        raise NotImplementedError(
            f"{spec.hf_repo} uses a SentencePiece/Unigram tokenizer "
            f"({type(tokenizer).__name__}) which cannot be represented as a "
            f"BertTokenizer ONNX graph compatible with Oracle's DBMS_VECTOR. "
            f"Use a WordPiece-based cross-encoder (e.g. ms-marco-MiniLM family)."
        )

    # Some cross-encoders ship a FastTokenizer subclass that gen_processing_models doesn't
    # recognise; fall through to BertTokenizerFast on the same vocab as the embedding path does.
    try:
        q_pre, q_outs = _make_tokenizer_subgraph(
            tokenizer, prefix="q_", input_name="pre_text_1"
        )
        d_pre, d_outs = _make_tokenizer_subgraph(
            tokenizer, prefix="d_", input_name="pre_text_2"
        )
    except ValueError as _tok_err:
        logger.warning(
            "Tokenizer %s not directly supported by gen_processing_models (%s); "
            "falling back to BertTokenizerFast.",
            type(tokenizer).__name__,
            _tok_err,
        )
        tokenizer = BertTokenizerFast.from_pretrained(spec.hf_repo, **cache_kwargs)
        q_pre, q_outs = _make_tokenizer_subgraph(
            tokenizer, prefix="q_", input_name="pre_text_1"
        )
        d_pre, d_outs = _make_tokenizer_subgraph(
            tokenizer, prefix="d_", input_name="pre_text_2"
        )

    # 3) Truncate each side, then unsqueeze to add the batch dim.
    # Cross-encoders (ms-marco family) are trained with asymmetric q/d budgets:
    # queries are short, documents are long. spec.query_max_length defaults to
    # 64; the doc side gets max_length - query_max_length (one of which is the
    # [CLS] the splice drops from the d-side). Each side keeps its closing [SEP]
    # when truncated, so the pair is still [CLS] q [SEP] d [SEP].
    q_len = max(1, min(spec.query_max_length, spec.max_length - 1))
    d_len = max(1, spec.max_length - q_len)
    _truncate_and_unsqueeze_tokenizer_outputs(q_pre, q_outs, q_len, init_prefix="q_")
    _truncate_and_unsqueeze_tokenizer_outputs(d_pre, d_outs, d_len, init_prefix="d_")

    # Merge q + d tokenizers into one graph.
    pre_merged = compose.merge_models(
        q_pre, d_pre, io_map=[], prefix1="", prefix2=""
    )

    # 4) Splice into [CLS] q [SEP] d [SEP] format with proper segments.
    # Validate that the tokenizer has all three special tokens for clarity, even
    # though the splice itself relies on the tokenizer subgraph having already
    # inserted [CLS]/[SEP] around each input.
    _bert_special_ids(tokenizer)
    _splice_query_doc_subgraph(pre_merged.graph)

    # Re-expose only the spliced BERT inputs (sequence dim is dynamic).
    expose_dynamic_int64_sequence_outputs(
        pre_merged.graph,
        ("input_ids", "attention_mask", "token_type_ids"),
        "pair_sequence_length",
    )

    # 5) Align opsets — bump core to 18, sync ir_version, copy custom domains.
    core_model = version_converter.convert_version(core_model, 18)
    core_model.ir_version = pre_merged.ir_version
    copy_missing_opset_domains(core_model, pre_merged)

    # Deduplicate the pre_merged opset list (compose.merge_models concatenates
    # both sub-graph opset_imports, producing duplicates) and align the default
    # ai.onnx version with the core. Without this the second compose.merge_models
    # call rejects the mismatched domains.
    dedupe_opset_imports(pre_merged, pin={"": 18, "ai.onnx": 18})

    # 6) Merge tokenizer subgraph + transformer.
    core_input_names = {i.name for i in core_model.graph.input}
    io_map = [
        (n, n) for n in ("input_ids", "attention_mask", "token_type_ids")
        if n in core_input_names
    ]
    merged = compose.merge_models(
        pre_merged, core_model, io_map=io_map, prefix1="", prefix2="core_"
    )

    # 7) Squeeze logits[1, 1] → scalar. The classifier head outputs core_logits.
    clear_outputs(merged.graph)

    expose_squeezed_float_output(
        merged.graph,
        "core_logits",
        "logits",
        axes=[0, 1],
        shape=[],
        axes_initializer_name="rerank_squeeze_axes",
    )

    # Pin any leftover dynamic batch dims to 1.
    pin_dynamic_batch_to_one(merged.graph)

    data = merged.SerializeToString()
    _validate_reranker_graph(data, spec)
    return data
