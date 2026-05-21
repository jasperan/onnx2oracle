"""End-to-end health check for a loaded ONNX model.

For ``function:"embedding"`` models, runs ``VECTOR_EMBEDDING`` against a live
Oracle connection, asserts shape and L2 normalization, and performs a cosine
sanity check (king/queen should be more similar than king/banana).

For ``function:"regression"`` (reranker) models, runs ``PREDICTION`` with two
text inputs and asserts that a relevant query/document pair scores strictly
higher than an irrelevant one.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass

import oracledb

from onnx2oracle._ident import validate_oracle_name
from onnx2oracle.connection import DSN
from onnx2oracle.loader import registered_task

logger = logging.getLogger(__name__)


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


def _embed(conn: oracledb.Connection, model_name: str, text: str) -> list[float]:
    safe = validate_oracle_name(model_name)
    cur = conn.cursor()
    cur.execute(
        f"SELECT VECTOR_EMBEDDING({safe} USING :t AS DATA) FROM dual",
        {"t": text},
    )
    row = cur.fetchone()
    if row is None or row[0] is None:
        return []
    vec = row[0]
    if isinstance(vec, str):
        return json.loads(vec)
    return list(vec)


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
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def smoke_test(dsn: DSN, oracle_name: str) -> VerifyResult:
    """Connect, confirm registration, and run a task-appropriate sanity check."""
    t0 = time.perf_counter()
    try:
        validate_oracle_name(oracle_name)
    except ValueError as e:
        return VerifyResult(False, False, None, None, False, 0, str(e))

    try:
        conn = oracledb.connect(
            user=dsn.user,
            password=dsn.password,
            dsn=dsn.to_oracle_dsn(),
            tcp_connect_timeout=30,
        )
    except Exception as e:
        return VerifyResult(False, False, None, None, False, 0, str(e))

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM user_mining_models WHERE model_name = :n",
            {"n": oracle_name},
        )
        registered = cur.fetchone()[0] > 0
        if not registered:
            return VerifyResult(
                True, False, None, None, False,
                int((time.perf_counter() - t0) * 1000),
                f"Model {oracle_name} not registered",
            )

        task = registered_task(conn, oracle_name) or "embedding"

        if task == "reranker":
            query = "How many people live in Berlin?"
            relevant = "Berlin has a population of 3.7 million inhabitants."
            irrelevant = "Bananas are a popular tropical fruit."
            r_score = _rerank_score(conn, oracle_name, query, relevant)
            i_score = _rerank_score(conn, oracle_name, query, irrelevant)
            if r_score is None or i_score is None:
                return VerifyResult(
                    True, True, None, None, False,
                    int((time.perf_counter() - t0) * 1000),
                    "PREDICTION returned null",
                    task=task,
                )
            sane = r_score > i_score
            return VerifyResult(
                connected=True,
                model_registered=True,
                sample_embedding_dims=None,
                sample_embedding_norm=None,
                similarity_sane=sane,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                task=task,
                sample_scores=(r_score, i_score),
            )

        vec = _embed(conn, oracle_name, "hello world")
        if not vec:
            return VerifyResult(
                True, True, None, None, False,
                int((time.perf_counter() - t0) * 1000),
                "VECTOR_EMBEDDING returned null",
                task=task,
            )

        dims = len(vec)
        norm = math.sqrt(sum(x * x for x in vec))

        v_king = _embed(conn, oracle_name, "king")
        v_queen = _embed(conn, oracle_name, "queen")
        v_banana = _embed(conn, oracle_name, "banana")
        sane = _cosine(v_king, v_queen) > _cosine(v_king, v_banana)

        return VerifyResult(
            True, True, dims, norm, sane,
            int((time.perf_counter() - t0) * 1000),
            task=task,
        )
    finally:
        conn.close()
