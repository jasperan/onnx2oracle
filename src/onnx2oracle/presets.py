"""Curated preset registry for popular embedding and reranker models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Task = Literal["embedding", "reranker"]
Pooling = Literal["mean", "cls"]


@dataclass(frozen=True)
class EmbeddingSpec:
    """Specification for a sentence-transformer embedding pipeline."""

    hf_repo: str
    dims: int
    pooling: Pooling
    normalize: bool
    oracle_name: str
    max_length: int = 512
    approx_size_mb: int = 100  # informational
    task: Literal["embedding"] = "embedding"


@dataclass(frozen=True)
class RerankerSpec:
    """Specification for a cross-encoder reranker pipeline."""

    hf_repo: str
    oracle_name: str
    max_length: int = 512
    approx_size_mb: int = 100  # informational
    task: Literal["reranker"] = "reranker"
    # How much of ``max_length`` is reserved for the query. The doc gets the
    # remainder. 64 matches MS MARCO / ms-marco-MiniLM training where queries
    # are short and documents are long.
    query_max_length: int = 64


ModelSpec: TypeAlias = EmbeddingSpec | RerankerSpec


PRESETS: dict[str, ModelSpec] = {
    "all-MiniLM-L6-v2": EmbeddingSpec(
        hf_repo="sentence-transformers/all-MiniLM-L6-v2",
        dims=384,
        pooling="mean",
        normalize=True,
        oracle_name="ALL_MINILM_L6_V2",
        approx_size_mb=90,
    ),
    "all-MiniLM-L12-v2": EmbeddingSpec(
        hf_repo="sentence-transformers/all-MiniLM-L12-v2",
        dims=384,
        pooling="mean",
        normalize=True,
        oracle_name="ALL_MINILM_L12_V2",
        approx_size_mb=130,
    ),
    "all-mpnet-base-v2": EmbeddingSpec(
        hf_repo="sentence-transformers/all-mpnet-base-v2",
        dims=768,
        pooling="mean",
        normalize=True,
        oracle_name="ALL_MPNET_BASE_V2",
        approx_size_mb=420,
    ),
    "bge-small-en-v1.5": EmbeddingSpec(
        hf_repo="BAAI/bge-small-en-v1.5",
        dims=384,
        pooling="cls",
        normalize=True,
        oracle_name="BGE_SMALL_EN_V1_5",
        approx_size_mb=130,
    ),
    "nomic-embed-text-v1": EmbeddingSpec(
        hf_repo="nomic-ai/nomic-embed-text-v1",
        dims=768,
        pooling="mean",
        normalize=True,
        oracle_name="NOMIC_EMBED_TEXT_V1",
        approx_size_mb=540,
    ),
    "ms-marco-MiniLM-L-6-v2": RerankerSpec(
        hf_repo="cross-encoder/ms-marco-MiniLM-L-6-v2",
        oracle_name="MS_MARCO_MINILM_L_6_V2",
        approx_size_mb=90,
    ),
    "ms-marco-MiniLM-L-12-v2": RerankerSpec(
        hf_repo="cross-encoder/ms-marco-MiniLM-L-12-v2",
        oracle_name="MS_MARCO_MINILM_L_12_V2",
        approx_size_mb=130,
    ),
}


def get_preset(name: str) -> ModelSpec:
    """Look up a preset by its CLI name. Raises KeyError if unknown."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name!r}. Available: {sorted(PRESETS.keys())}")
    return PRESETS[name]
