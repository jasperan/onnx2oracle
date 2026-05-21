"""Oracle-side ONNX model registration via DBMS_VECTOR.LOAD_ONNX_MODEL."""

from __future__ import annotations

import json
import logging

import oracledb

from onnx2oracle._ident import validate_oracle_name
from onnx2oracle.connection import DSN

logger = logging.getLogger(__name__)


def build_metadata_json(task: str = "embedding") -> str:
    """Metadata descriptor for Oracle's DBMS_VECTOR.LOAD_ONNX_MODEL.

    Embedding pipeline (pipeline.build_augmented) emits one ``pre_text`` input
    and one ``embedding`` output. Reranker pipeline (pipeline.build_reranker)
    emits two text inputs (``pre_text_1``, ``pre_text_2``) and one ``logits``
    scalar output, which Oracle treats as a regression score.
    """
    if task == "embedding":
        body = {
            "function": "embedding",
            "embeddingOutput": "embedding",
            "input": {"pre_text": ["DATA"]},
        }
    elif task == "reranker":
        body = {
            "function": "regression",
            "regressionOutput": "logits",
            "input": {
                "pre_text_1": ["DATA1"],
                "pre_text_2": ["DATA2"],
            },
        }
    else:
        raise ValueError(f"Unknown task: {task!r} (expected 'embedding' or 'reranker')")
    return json.dumps(body)


def model_exists(conn: oracledb.Connection, oracle_name: str) -> bool:
    validate_oracle_name(oracle_name)
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM user_mining_models WHERE model_name = :name",
        {"name": oracle_name},
    )
    (count,) = cur.fetchone()
    return count > 0


def registered_task(conn: oracledb.Connection, oracle_name: str) -> str | None:
    """Return ``"reranker"``, ``"embedding"``, or ``None`` if not registered.

    Reads ``user_mining_models.mining_function``: REGRESSION → reranker,
    anything else → embedding. Used by ``rerank``/``verify`` to dispatch the
    right SQL surface without trusting the caller to remember the task.
    """
    validate_oracle_name(oracle_name)
    cur = conn.cursor()
    cur.execute(
        "SELECT mining_function FROM user_mining_models WHERE model_name = :n",
        {"n": oracle_name},
    )
    row = cur.fetchone()
    if not row:
        return None
    fn = str(row[0] or "").upper()
    return "reranker" if fn == "REGRESSION" else "embedding"


def drop_model(conn: oracledb.Connection, oracle_name: str) -> None:
    """Drop a previously-registered ONNX model. Uses DBMS_VECTOR.DROP_ONNX_MODEL."""
    validate_oracle_name(oracle_name)
    cur = conn.cursor()
    cur.execute(
        "BEGIN DBMS_VECTOR.DROP_ONNX_MODEL(model_name => :n, force => TRUE); END;",
        {"n": oracle_name},
    )
    conn.commit()


def upload_model(
    dsn: DSN,
    model_bytes: bytes,
    oracle_name: str,
    force: bool = False,
    task: str = "embedding",
) -> None:
    """Connect to Oracle and register *model_bytes* as *oracle_name*.

    ``task`` controls the metadata JSON: ``"embedding"`` for vector models,
    ``"reranker"`` for two-input regression models queried via PREDICTION.

    If a model with that name already exists:
      - force=False: log and return (idempotent no-op).
      - force=True: drop and re-upload.
    """
    validate_oracle_name(oracle_name)
    logger.info("Connecting to %s ...", dsn.display())
    conn = oracledb.connect(
        user=dsn.user,
        password=dsn.password,
        dsn=dsn.to_oracle_dsn(),
        tcp_connect_timeout=30,
    )
    try:
        if model_exists(conn, oracle_name):
            if not force:
                logger.info(
                    "Model %s already registered — skipping (use --force to replace).",
                    oracle_name,
                )
                return
            logger.info("Dropping existing model %s ...", oracle_name)
            drop_model(conn, oracle_name)

        logger.info("Uploading %d bytes as %s (task=%s) ...", len(model_bytes), oracle_name, task)
        cur = conn.cursor()
        cur.execute(
            """
            BEGIN
                DBMS_VECTOR.LOAD_ONNX_MODEL(
                    model_name => :model_name,
                    model_data => :model_data,
                    metadata   => JSON(:metadata)
                );
            END;
            """,
            {
                "model_name": oracle_name,
                "model_data": model_bytes,
                "metadata": build_metadata_json(task),
            },
        )
        conn.commit()
        logger.info("Model %s registered successfully.", oracle_name)
    finally:
        conn.close()
