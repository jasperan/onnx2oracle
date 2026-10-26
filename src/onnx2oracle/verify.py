"""End-to-end health check for a loaded ONNX model.

For ``function:"embedding"`` models, runs ``VECTOR_EMBEDDING`` against a live
Oracle connection, asserts the output dimension and L2 normalization, and
performs a cosine sanity check (king/queen should be more similar than
king/banana).

For ``function:"regression"`` (reranker) models, runs ``PREDICTION`` with two
text inputs and asserts that a relevant query/document pair scores strictly
higher than an irrelevant one.

``smoke_test`` does not raise for database problems: every failure comes back
as ``VerifyResult.error``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import oracledb

from onnx2oracle._ident import validate_oracle_name
from onnx2oracle.connection import DSN
from onnx2oracle.loader import registered_task

logger = logging.getLogger(__name__)

NORM_TOLERANCE = 1e-3
"""How far an L2-normalized embedding's norm may drift from 1.0 and still pass."""


@dataclass
class VerifyResult:
    connected: bool
    model_registered: bool
    sample_embedding_dims: int | None
    sample_embedding_norm: float | None
    similarity_sane: bool
    elapsed_ms: int
    error: str | None = None
    task: str | None = None
    sample_scores: tuple[float, float] | None = None
    # What the caller expects of an embedding model. ``expected_dims=None`` means "any positive dimension";
    # ``expect_normalized=False`` waives the L2-norm check for models loaded with --no-normalize.
    expected_dims: int | None = None
    expect_normalized: bool = True
    # (cos(king, queen), cos(king, banana)) for embedding models.
    similarities: tuple[float, float] | None = None

    @property
    def dims_ok(self) -> bool:
        """Dimension is positive and, when an expectation is known, equal to it."""
        dims = self.sample_embedding_dims
        if dims is None or dims <= 0:
            return False
        return self.expected_dims is None or dims == self.expected_dims

    @property
    def norm_ok(self) -> bool:
        """Sample norm is finite and, for normalized models, within ``NORM_TOLERANCE`` of 1.0."""
        norm = self.sample_embedding_norm
        if norm is None or not math.isfinite(norm):
            return False
        return not self.expect_normalized or abs(norm - 1.0) <= NORM_TOLERANCE

    @property
    def scores_ok(self) -> bool:
        """Reranker sample scores exist and are finite."""
        return self.sample_scores is not None and all(math.isfinite(s) for s in self.sample_scores)

    @property
    def ok(self) -> bool:
        """True only when every task-appropriate check passed."""
        if self.error is not None or not self.connected or not self.model_registered:
            return False
        if self.task == "embedding":
            return self.dims_ok and self.norm_ok and self.similarity_sane
        if self.task == "reranker":
            return self.scores_ok and self.similarity_sane
        return False


def _to_floats(vec: Any) -> list[float]:
    """Turn what python-oracledb returns for a VECTOR value (array, list, or JSON text) into floats."""
    if vec is None:
        return []
    if isinstance(vec, str):
        return [float(x) for x in json.loads(vec)]
    return [float(x) for x in vec]


def embed_texts(conn: oracledb.Connection, model_name: str, texts: Sequence[str]) -> list[list[float]]:
    """Embed *texts* with the registered model *model_name* in a single query.

    Returns one vector per input, in input order. An input Oracle yields NULL for
    (an empty string, for instance) comes back as ``[]``.
    """
    safe = validate_oracle_name(model_name)
    if not texts:
        return []
    # One row per text; the explicit idx keeps the result order independent of the optimizer.
    rows = " UNION ALL ".join(f"SELECT {i} AS idx, :t{i} AS t FROM dual" for i in range(len(texts)))
    sql = f"SELECT idx, VECTOR_EMBEDDING({safe} USING t AS DATA) FROM ({rows}) ORDER BY idx"
    cur = conn.cursor()
    cur.execute(sql, {f"t{i}": text for i, text in enumerate(texts)})
    by_idx = {int(idx): _to_floats(vec) for idx, vec in cur.fetchall()}
    return [by_idx.get(i, []) for i in range(len(texts))]


def _rerank_score(conn: oracledb.Connection, model_name: str, query: str, doc: str) -> float | None:
    safe = validate_oracle_name(model_name)
    cur = conn.cursor()
    cur.execute(
        f"SELECT PREDICTION({safe} USING :q AS DATA1, :d AS DATA2) FROM dual",
        {"q": query, "d": doc},
    )
    row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Cannot compare embeddings of different lengths ({len(a)} vs {len(b)}).")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _describe(exc: BaseException) -> str:
    """One-line-ish, never-empty description of *exc* (``VerifyResult.error`` must be truthy)."""
    text = str(exc).strip()
    if isinstance(exc, oracledb.Error):
        return text or type(exc).__name__
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _close_quietly(conn: oracledb.Connection) -> None:
    try:
        conn.close()
    except Exception:
        logger.debug("Closing the Oracle connection failed", exc_info=True)


def smoke_test(
    dsn: DSN,
    oracle_name: str,
    *,
    expected_dims: int | None = None,
    expect_normalized: bool = True,
) -> VerifyResult:
    """Connect, confirm registration, and run a task-appropriate sanity check.

    ``expected_dims`` pins the embedding dimension and ``expect_normalized`` says whether the
    embedding should have an L2 norm of ~1 (see ``VerifyResult.ok``). Database problems are
    reported through ``VerifyResult.error`` rather than raised.
    """
    t0 = time.perf_counter()

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    try:
        validate_oracle_name(oracle_name)
    except ValueError as e:
        return VerifyResult(False, False, None, None, False, _elapsed_ms(), str(e))

    try:
        conn = dsn.connect()
    except Exception as e:
        return VerifyResult(False, False, None, None, False, _elapsed_ms(), _describe(e))

    registered = False
    task: str | None = None
    try:
        task = registered_task(conn, oracle_name)
        if task is None:
            return VerifyResult(
                True, False, None, None, False,
                _elapsed_ms(),
                f"Model {oracle_name} not registered",
            )
        registered = True

        if task == "reranker":
            query = "How many people live in Berlin?"
            relevant = "Berlin has a population of 3.7 million inhabitants."
            irrelevant = "Bananas are a popular tropical fruit."
            r_score = _rerank_score(conn, oracle_name, query, relevant)
            i_score = _rerank_score(conn, oracle_name, query, irrelevant)
            if r_score is None or i_score is None:
                return VerifyResult(
                    True, True, None, None, False,
                    _elapsed_ms(),
                    "PREDICTION returned null",
                    task=task,
                )
            return VerifyResult(
                connected=True,
                model_registered=True,
                sample_embedding_dims=None,
                sample_embedding_norm=None,
                similarity_sane=r_score > i_score,
                elapsed_ms=_elapsed_ms(),
                task=task,
                sample_scores=(r_score, i_score),
            )

        texts = ["hello world", "king", "queen", "banana"]
        vectors = embed_texts(conn, oracle_name, texts)
        missing = [text for text, vec in zip(texts, vectors, strict=True) if not vec]
        if missing:
            return VerifyResult(
                True, True, None, None, False,
                _elapsed_ms(),
                f"VECTOR_EMBEDDING returned null for {missing!r}",
                task=task,
                expected_dims=expected_dims,
                expect_normalized=expect_normalized,
            )

        sample, king, queen, banana = vectors
        dims = len(sample)
        norm = math.sqrt(sum(x * x for x in sample))
        lengths = [len(vec) for vec in vectors]
        if len(set(lengths)) != 1:
            return VerifyResult(
                True, True, dims, norm, False,
                _elapsed_ms(),
                f"VECTOR_EMBEDDING returned vectors of different lengths {lengths} for {texts!r}",
                task=task,
                expected_dims=expected_dims,
                expect_normalized=expect_normalized,
            )

        cos_king_queen = _cosine(king, queen)
        cos_king_banana = _cosine(king, banana)
        return VerifyResult(
            True, True, dims, norm,
            cos_king_queen > cos_king_banana,
            _elapsed_ms(),
            task=task,
            expected_dims=expected_dims,
            expect_normalized=expect_normalized,
            similarities=(cos_king_queen, cos_king_banana),
        )
    except Exception as e:
        logger.debug("Smoke test of %s failed", oracle_name, exc_info=True)
        return VerifyResult(
            True, registered, None, None, False,
            _elapsed_ms(),
            _describe(e),
            task=task,
            expected_dims=expected_dims,
            expect_normalized=expect_normalized,
        )
    finally:
        _close_quietly(conn)
