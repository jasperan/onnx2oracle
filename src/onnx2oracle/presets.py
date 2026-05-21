"""Curated preset registry for popular embedding and reranker models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Task = Literal["embedding", "reranker"]


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an ONNX model to be loaded into Oracle.

    For ``task == "embedding"``: ``dims``, ``pooling`` and ``normalize`` describe
    the post-transformer head appended by ``pipeline.build_augmented``.

    For ``task == "reranker"``: those fields are unused (the cross-encoder head
    is taken from the source repo's ``BertForSequenceClassification``). Keep
    them at safe defaults so callers that don't care can ignore them.
    """

    hf_repo: str
    dims: int
    pooling: Literal["mean", "cls"]
    normalize: bool
    oracle_name: str
    max_length: int = 512
    approx_size_mb: int = 100  # informational
    task: Task = "embedding"
    # Reranker only: how much of ``max_length`` is reserved for the query.
    # The doc gets the remainder. 64 matches MS MARCO / ms-marco-MiniLM
    # training where queries are short and documents are long.
    query_max_length: int = 64


PRESETS: dict[str, ModelSpec] = {
    "all-MiniLM-L6-v2": ModelSpec(
        hf_repo="sentence-transformers/all-MiniLM-L6-v2",
        dims=384,
        pooling="mean",
        normalize=True,
        oracle_name="ALL_MINILM_L6_V2",
        approx_size_mb=90,
    ),
    "all-MiniLM-L12-v2": ModelSpec(
        hf_repo="sentence-transformers/all-MiniLM-L12-v2",
        dims=384,
        pooling="mean",
        normalize=True,
        oracle_name="ALL_MINILM_L12_V2",
        approx_size_mb=130,
    ),
    "all-mpnet-base-v2": ModelSpec(
        hf_repo="sentence-transformers/all-mpnet-base-v2",
        dims=768,
        pooling="mean",
        normalize=True,
        oracle_name="ALL_MPNET_BASE_V2",
        approx_size_mb=420,
    ),
    "bge-small-en-v1.5": ModelSpec(
        hf_repo="BAAI/bge-small-en-v1.5",
        dims=384,
        pooling="cls",
        normalize=True,
        oracle_name="BGE_SMALL_EN_V1_5",
        approx_size_mb=130,
    ),
    "nomic-embed-text-v1": ModelSpec(
        hf_repo="nomic-ai/nomic-embed-text-v1",
        dims=768,
        pooling="mean",
        normalize=True,
        oracle_name="NOMIC_EMBED_TEXT_V1",
        approx_size_mb=540,
    ),
    "ms-marco-MiniLM-L-6-v2": ModelSpec(
        hf_repo="cross-encoder/ms-marco-MiniLM-L-6-v2",
        dims=1,
        pooling="cls",
        normalize=False,
        oracle_name="MS_MARCO_MINILM_L_6_V2",
        approx_size_mb=90,
        task="reranker",
    ),
    "ms-marco-MiniLM-L-12-v2": ModelSpec(
        hf_repo="cross-encoder/ms-marco-MiniLM-L-12-v2",
        dims=1,
        pooling="cls",
        normalize=False,
        oracle_name="MS_MARCO_MINILM_L_12_V2",
        approx_size_mb=130,
        task="reranker",
    ),
}


def get_preset(name: str) -> ModelSpec:
    """Look up a preset by its CLI name. Raises KeyError if unknown."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name!r}. Available: {sorted(PRESETS.keys())}")
    return PRESETS[name]
