"""End-to-end health check for a loaded ONNX embedding model.

Runs VECTOR_EMBEDDING against a live Oracle connection, asserts shape and
L2 normalization, and performs a cosine-sanity check (king/queen should be
more similar than king/banana).
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


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def smoke_test(dsn: DSN, oracle_name: str) -> VerifyResult:
    """Connect, confirm registration, embed a sample, check cosine sanity."""
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

        vec = _embed(conn, oracle_name, "hello world")
        if not vec:
            return VerifyResult(
                True, True, None, None, False,
                int((time.perf_counter() - t0) * 1000),
                "VECTOR_EMBEDDING returned null",
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
        )
    finally:
        conn.close()
