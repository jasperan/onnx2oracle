"""Augmented ONNX pipeline builder.

Downloads a sentence-transformer HF repo, wraps the tokenizer as ONNX ops,
composes it with the transformer body, appends pooling + L2-normalization,
and emits a single ONNX graph that accepts a string and returns a
dims-sized normalized float32 vector.

The resulting ONNX can be loaded into Oracle via DBMS_VECTOR.LOAD_ONNX_MODEL
and queried via VECTOR_EMBEDDING(... USING :text AS DATA).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper, version_converter

from onnx2oracle.presets import ModelSpec

logger = logging.getLogger(__name__)


def build_augmented(spec: ModelSpec, cache_dir: Path | None = None) -> bytes:
    """Build the augmented ONNX pipeline for *spec* and return it as bytes.

    Graph shape: string -> tokenizer -> transformer -> pool -> l2-normalize -> [dims] float32.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    from onnxruntime_extensions import gen_processing_models
    from transformers import AutoTokenizer

    cache_kwargs: dict[str, str] = {}
    if cache_dir is not None:
        cache_kwargs["cache_dir"] = str(cache_dir)

    # 1) Download core transformer ONNX (prefer pre-exported onnx/model.onnx)
    try:
        core_path = hf_hub_download(spec.hf_repo, "onnx/model.onnx", **cache_kwargs)
    except Exception:
        # Fallback: export from PyTorch (slower but universal)
        from transformers import AutoModel
        from transformers.onnx import export as hf_onnx_export
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
        pre_model, _ = gen_processing_models(
            tokenizer, pre_kwargs={}, post_kwargs=None, opset=14
        )
        pre_output_names = {o.name for o in pre_model.graph.output}
        if "input_ids" not in pre_output_names:
            raise ValueError(
                f"gen_processing_models produced unexpected outputs: {pre_output_names}"
            )
    except (ValueError, Exception) as _tok_err:
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
            )
        pre_model, _ = gen_processing_models(
            tokenizer, pre_kwargs={}, post_kwargs=None, opset=14
        )

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

    axes_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsqueeze_axes_0")
    pre_model.graph.initializer.append(axes_0)

    for name in bert_tensor_names:
        flat_name = f"{name}_flat"
        for node in pre_model.graph.node:
            for i, out in enumerate(node.output):
                if out == name:
                    node.output[i] = flat_name
        pre_model.graph.node.append(
            helper.make_node("Unsqueeze", [flat_name, "unsqueeze_axes_0"], [name])
        )
        for out in pre_model.graph.output:
            if out.name == name:
                shape = out.type.tensor_type.shape
                while len(shape.dim) > 0:
                    shape.dim.pop()
                shape.dim.add().dim_value = 1
                shape.dim.add().dim_param = "sequence_length"

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
