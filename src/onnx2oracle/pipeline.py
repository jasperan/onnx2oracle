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
"""

from __future__ import annotations

import logging
import tempfile
from importlib.util import find_spec
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper, version_converter

from onnx2oracle.presets import ModelSpec

logger = logging.getLogger(__name__)


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


def _truncate_and_unsqueeze_tokenizer_outputs(
    pre_model: onnx.ModelProto,
    names: list[str],
    max_length: int,
    init_prefix: str = "",
) -> None:
    """Truncate 1-D tokenizer outputs, then add the transformer batch dimension.

    ``init_prefix`` lets the helper run twice in one graph (q-side and d-side
    of a reranker pipeline) without colliding on shared initializer names.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")

    starts_n = f"{init_prefix}truncate_starts"
    ends_n = f"{init_prefix}truncate_ends"
    axes_n = f"{init_prefix}truncate_axes"
    steps_n = f"{init_prefix}truncate_steps"
    unsq_n = f"{init_prefix}unsqueeze_axes_0"

    slice_starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name=starts_n)
    slice_ends = numpy_helper.from_array(np.array([max_length], dtype=np.int64), name=ends_n)
    slice_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name=axes_n)
    slice_steps = numpy_helper.from_array(np.array([1], dtype=np.int64), name=steps_n)
    axes_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name=unsq_n)
    pre_model.graph.initializer.extend([slice_starts, slice_ends, slice_axes, slice_steps, axes_0])

    for name in names:
        raw_name = f"{name}_raw"
        flat_name = f"{name}_flat"
        for node in pre_model.graph.node:
            for i, out in enumerate(node.output):
                if out == name:
                    node.output[i] = raw_name
        pre_model.graph.node.append(
            helper.make_node(
                "Slice",
                [raw_name, starts_n, ends_n, axes_n, steps_n],
                [flat_name],
            )
        )
        pre_model.graph.node.append(
            helper.make_node("Unsqueeze", [flat_name, unsq_n], [name])
        )
        for out in pre_model.graph.output:
            if out.name == name:
                shape = out.type.tensor_type.shape
                while len(shape.dim) > 0:
                    shape.dim.pop()
                shape.dim.add().dim_value = 1
                shape.dim.add().dim_param = "sequence_length"


def build_augmented(spec: ModelSpec, cache_dir: Path | None = None) -> bytes:
    """Build the augmented ONNX pipeline for *spec* and return it as bytes.

    Graph shape: string -> tokenizer -> transformer -> pool -> l2-normalize -> [dims] float32.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    from onnxruntime_extensions import gen_processing_models
    from transformers import AutoTokenizer

    cache_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        cache_kwargs["cache_dir"] = str(cache_dir)

    # 1) Download core transformer ONNX (prefer pre-exported onnx/model.onnx)
    try:
        core_path = hf_hub_download(spec.hf_repo, "onnx/model.onnx", **cache_kwargs)
    except Exception as exc:
        if not _should_use_export_fallback(exc):
            raise
        _require_torch_for_export_fallback()
        # Fallback: export from PyTorch (slower but universal)
        from transformers import AutoModel
        from transformers.onnx.convert import export as hf_onnx_export
        from transformers.onnx.features import FeaturesManager

        snap = Path(snapshot_download(spec.hf_repo, **cache_kwargs))
        out_dir = Path(tempfile.mkdtemp(prefix="onnx2oracle_export_"))
        tokenizer_pt = AutoTokenizer.from_pretrained(snap)
        model_pt = AutoModel.from_pretrained(snap)
        _model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model_pt, feature="default"
        )
        config = model_onnx_config(model_pt.config)
        core_path = out_dir / "model.onnx"
        hf_onnx_export(
            preprocessor=tokenizer_pt, model=model_pt, config=config, opset=14, output=core_path
        )
        core_path = str(core_path)
    else:
        # Some repos store large tensors beside model.onnx as external data.
        # Probe without loading sidecars, then fetch the exact files referenced.
        core_probe = onnx.load(core_path, load_external_data=False)
        for location in _external_data_locations(core_probe):
            repo_path = _external_data_repo_path(location)
            logger.info("Downloading ONNX external data sidecar %s", repo_path)
            hf_hub_download(spec.hf_repo, repo_path, **cache_kwargs)

    core_model = onnx.load(core_path)

    # 2) Generate tokenizer ONNX
    # Some tokenizer classes (MPNetTokenizer, XLMRobertaTokenizer) are not
    # supported by gen_processing_models, or produce non-BERT output names
    # (tokens/instance_indices/token_indices instead of input_ids/...).
    # Fallback: BertTokenizerFast shares the WordPiece vocabulary and produces
    # the standard input_ids/token_type_ids/attention_mask outputs that the
    # core transformer expects.
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, **cache_kwargs)
    try:
        pre_model_raw, _ = gen_processing_models(
            tokenizer, pre_kwargs={}, post_kwargs=cast(Any, None), opset=14
        )
        pre_model = cast(onnx.ModelProto, pre_model_raw)
        pre_output_names = {o.name for o in pre_model.graph.output}
        if "input_ids" not in pre_output_names:
            raise ValueError(
                f"gen_processing_models produced unexpected outputs: {pre_output_names}"
            )
    except Exception as _tok_err:
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
        pre_model_raw, _ = gen_processing_models(
            tokenizer, pre_kwargs={}, post_kwargs=cast(Any, None), opset=14
        )
        pre_model = cast(onnx.ModelProto, pre_model_raw)

    # 3) Align opsets — bump core to 18 + copy custom domains from pre
    core_model = version_converter.convert_version(core_model, 18)
    core_model.ir_version = pre_model.ir_version

    core_domains = {o.domain for o in core_model.opset_import}
    for o in pre_model.opset_import:
        if o.domain not in core_domains:
            new_o = core_model.opset_import.add()
            new_o.domain = o.domain
            new_o.version = o.version

    # 4) Unsqueeze tokenizer outputs: [seq_len] -> [1, seq_len]
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

    while len(merged.graph.output) > 0:
        merged.graph.output.pop()

    # 6) Pool
    if spec.pooling == "mean":
        pool_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="pool_axes_1")
        merged.graph.initializer.append(pool_axes)
        merged.graph.node.append(
            helper.make_node(
                "ReduceMean",
                ["core_last_hidden_state", "pool_axes_1"],
                ["pooled"],
                keepdims=0,
            )
        )
        pool_out = "pooled"
    elif spec.pooling == "cls":
        # Gather the first token's hidden state; keepdims = False via squeeze
        cls_indices = numpy_helper.from_array(np.array([0], dtype=np.int64), name="cls_indices")
        merged.graph.initializer.append(cls_indices)
        merged.graph.node.append(
            helper.make_node(
                "Gather",
                ["core_last_hidden_state", "cls_indices"],
                ["pooled_2d"],
                axis=1,
            )
        )
        # Gather keeps dim 1 -> shape [1, 1, hidden]; squeeze dim 1 to [1, hidden]
        squeeze_cls = numpy_helper.from_array(np.array([1], dtype=np.int64), name="squeeze_cls_axes")
        merged.graph.initializer.append(squeeze_cls)
        merged.graph.node.append(
            helper.make_node("Squeeze", ["pooled_2d", "squeeze_cls_axes"], ["pooled"])
        )
        pool_out = "pooled"
    else:
        raise ValueError(f"Unsupported pooling: {spec.pooling!r}")

    # 7) L2 normalize (optional)
    if spec.normalize:
        pow_exp = numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="pow_exp")
        merged.graph.initializer.append(pow_exp)
        merged.graph.node.append(helper.make_node("Pow", [pool_out, "pow_exp"], ["squared"]))

        axes_neg1 = numpy_helper.from_array(np.array([-1], dtype=np.int64), name="axes_neg1")
        merged.graph.initializer.append(axes_neg1)
        merged.graph.node.append(
            helper.make_node("ReduceSum", ["squared", "axes_neg1"], ["sum_sq"], keepdims=1)
        )
        merged.graph.node.append(helper.make_node("Sqrt", ["sum_sq"], ["l2_norm"]))

        eps = numpy_helper.from_array(np.array(1e-12, dtype=np.float32), name="eps_val")
        merged.graph.initializer.append(eps)
        merged.graph.node.append(helper.make_node("Max", ["l2_norm", "eps_val"], ["l2_safe"]))
        merged.graph.node.append(helper.make_node("Div", [pool_out, "l2_safe"], ["emb_2d"]))
        final_2d = "emb_2d"
    else:
        final_2d = pool_out

    # 8) Squeeze batch dim: [1, dims] -> [dims]
    sq_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="squeeze_axes_final")
    merged.graph.initializer.append(sq_axes)
    merged.graph.node.append(
        helper.make_node("Squeeze", [final_2d, "squeeze_axes_final"], ["embedding"])
    )

    merged.graph.output.append(
        helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [spec.dims])
    )

    # Fix leftover dynamic batch dims in value_info / inputs
    for vi in list(merged.graph.value_info) + list(merged.graph.input):
        if vi.type.tensor_type.shape:
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_param == "batch_size":
                    dim.ClearField("dim_param")
                    dim.dim_value = 1

    # Serialize
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx.save(merged, tmp.name)
        data = Path(tmp.name).read_bytes()
        Path(tmp.name).unlink()
    return data


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
    from onnxruntime_extensions import gen_processing_models

    raw, _ = gen_processing_models(
        tokenizer, pre_kwargs={}, post_kwargs=cast(Any, None), opset=14
    )
    model = cast(onnx.ModelProto, raw)
    output_names = {o.name for o in model.graph.output}
    if "input_ids" not in output_names:
        raise ValueError(
            f"gen_processing_models produced unexpected outputs: {output_names}"
        )

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


def build_reranker(spec: ModelSpec, cache_dir: Path | None = None) -> bytes:
    """Build a cross-encoder reranker ONNX graph and return its bytes.

    Graph shape: ``(pre_text_1: string[1], pre_text_2: string[1])`` →
    tokenizer → splice → BertForSequenceClassification → scalar ``logits``.

    The resulting graph is loaded into Oracle with ``function:"regression"``
    metadata (see ``loader.build_metadata_json("reranker")``) and queried via
    ``PREDICTION(model USING q AS DATA1, d AS DATA2)``.
    """
    if spec.task != "reranker":
        raise ValueError(
            f"build_reranker requires spec.task == 'reranker', got {spec.task!r}"
        )

    from transformers import AutoTokenizer, BertTokenizerFast

    cache_kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        cache_kwargs["cache_dir"] = str(cache_dir)

    # 1) Export the cross-encoder body from PyTorch.
    out_dir = Path(tempfile.mkdtemp(prefix="onnx2oracle_rerank_"))
    core_path = out_dir / "model.onnx"
    _export_cross_encoder_onnx(spec.hf_repo, cache_dir, core_path)
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
    except Exception as _tok_err:
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
    # 64; the doc side gets max_length - query_max_length (minus the [CLS] the
    # splice drops from the d-side).
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
    while len(pre_merged.graph.output) > 0:
        pre_merged.graph.output.pop()
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        vi = helper.make_tensor_value_info(name, TensorProto.INT64, [1, None])
        # ``None`` becomes an unset dim — clear it so the consumer accepts dynamic shape.
        seq_dim = vi.type.tensor_type.shape.dim[1]
        seq_dim.ClearField("dim_value")
        seq_dim.dim_param = "pair_sequence_length"
        pre_merged.graph.output.append(vi)

    # 5) Align opsets — bump core to 18, sync ir_version, copy custom domains.
    core_model = version_converter.convert_version(core_model, 18)
    core_model.ir_version = pre_merged.ir_version
    core_domains = {o.domain for o in core_model.opset_import}
    for o in pre_merged.opset_import:
        if o.domain not in core_domains:
            new_o = core_model.opset_import.add()
            new_o.domain = o.domain
            new_o.version = o.version
            core_domains.add(o.domain)

    # Deduplicate the pre_merged opset list (compose.merge_models concatenates
    # both sub-graph opset_imports, producing duplicates) and align the default
    # ai.onnx version with the core. Without this the second compose.merge_models
    # call rejects the mismatched domains.
    seen: dict[str, int] = {}
    for o in pre_merged.opset_import:
        if o.domain not in seen:
            seen[o.domain] = o.version
        else:
            seen[o.domain] = max(seen[o.domain], o.version)
    seen[""] = 18
    seen["ai.onnx"] = 18
    while len(pre_merged.opset_import) > 0:
        pre_merged.opset_import.pop()
    for domain, version in seen.items():
        new_o = pre_merged.opset_import.add()
        new_o.domain = domain
        new_o.version = version

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
    while len(merged.graph.output) > 0:
        merged.graph.output.pop()

    sq_axes = numpy_helper.from_array(
        np.array([0, 1], dtype=np.int64), name="rerank_squeeze_axes"
    )
    merged.graph.initializer.append(sq_axes)
    merged.graph.node.append(
        helper.make_node("Squeeze", ["core_logits", "rerank_squeeze_axes"], ["logits"])
    )
    merged.graph.output.append(
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [])
    )

    # Pin any leftover dynamic batch dims to 1.
    for vi in list(merged.graph.value_info) + list(merged.graph.input):
        if vi.type.tensor_type.shape:
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_param == "batch_size":
                    dim.ClearField("dim_param")
                    dim.dim_value = 1

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx.save(merged, tmp.name)
        data = Path(tmp.name).read_bytes()
        Path(tmp.name).unlink()
    return data
