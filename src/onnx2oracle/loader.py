"""Oracle-side ONNX model registration via DBMS_VECTOR.LOAD_ONNX_MODEL."""

from __future__ import annotations

import json
import logging

import oracledb

from onnx2oracle.connection import DSN

logger = logging.getLogger(__name__)


def build_metadata_json() -> str:
    """Metadata descriptor for Oracle's DBMS_VECTOR.LOAD_ONNX_MODEL.

    The input tensor is named 'pre_text' because pipeline.build_augmented uses
    prefix1="pre_" when merging the tokenizer onto the transformer.
    """
    return json.dumps(
        {
            "function": "embedding",
            "embeddingOutput": "embedding",
            "input": {"pre_text": ["DATA"]},
        }
    )


def model_exists(conn: oracledb.Connection, oracle_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM user_mining_models WHERE model_name = :name",
        {"name": oracle_name},
    )
    (count,) = cur.fetchone()
    return count > 0


def drop_model(conn: oracledb.Connection, oracle_name: str) -> None:
    """Drop a previously-registered ONNX model. Uses DBMS_VECTOR.DROP_ONNX_MODEL."""
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
) -> None:
    """Connect to Oracle and register *model_bytes* as *oracle_name*.

    If a model with that name already exists:
      - force=False: log and return (idempotent no-op).
      - force=True: drop and re-upload.
    """
    logger.info("Connecting to %s ...", dsn.display())
    conn = oracledb.connect(user=dsn.user, password=dsn.password, dsn=dsn.to_oracle_dsn())
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

        logger.info("Uploading %d bytes as %s ...", len(model_bytes), oracle_name)
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
                "metadata": build_metadata_json(),
            },
        )
        conn.commit()
        logger.info("Model %s registered successfully.", oracle_name)
    finally:
        conn.close()
