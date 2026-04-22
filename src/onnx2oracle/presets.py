"""Curated preset registry for popular embedding models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an embedding model to be loaded into Oracle."""

    hf_repo: str
    dims: int
    pooling: Literal["mean", "cls"]
    normalize: bool
    oracle_name: str
    max_length: int = 512
    approx_size_mb: int = 100  # informational


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
    "multilingual-e5-small": ModelSpec(
        hf_repo="intfloat/multilingual-e5-small",
        dims=384,
        pooling="mean",
        normalize=True,
        oracle_name="MULTILINGUAL_E5_SMALL",
        approx_size_mb=470,
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
}


def get_preset(name: str) -> ModelSpec:
    """Look up a preset by its CLI name. Raises KeyError if unknown."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name!r}. Available: {sorted(PRESETS.keys())}")
    return PRESETS[name]
