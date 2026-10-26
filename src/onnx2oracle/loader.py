"""Oracle-side ONNX model registration via DBMS_VECTOR.LOAD_ONNX_MODEL."""

from __future__ import annotations

import json
import logging
from enum import Enum

import oracledb

from onnx2oracle._ident import validate_oracle_name
from onnx2oracle.connection import DSN

logger = logging.getLogger(__name__)


class UploadStatus(str, Enum):
    """Outcome of :func:`upload_model`."""

    REGISTERED = "registered"
    SKIPPED = "skipped"


class TaskMismatchError(ValueError):
    """A model of a different task already holds the requested name (and ``force`` is off)."""


class ModelUploadError(RuntimeError):
    """``LOAD_ONNX_MODEL`` failed after the previously registered model had been dropped."""


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


def lookup_registered_task(dsn: DSN, oracle_name: str) -> str | None:
    """Open a short-lived connection and return the task *oracle_name* is registered as, or ``None``."""
    validate_oracle_name(oracle_name)
    conn = dsn.connect()
    try:
        return registered_task(conn, oracle_name)
    finally:
        conn.close()


def check_existing(existing: str | None, oracle_name: str, task: str, force: bool = False) -> UploadStatus | None:
    """Decide what an already-registered model means for a requested upload.

    *existing* is the task *oracle_name* is currently registered as (``None`` if free). Returns
    ``UploadStatus.SKIPPED`` when nothing has to happen (same task, no *force*) and ``None`` when the
    upload should go ahead (name free, or *force* replaces the model). Raises
    :class:`TaskMismatchError` when a model of a different task holds the name and *force* is off.
    """
    if existing is None or force:
        return None
    if existing != task:
        raise TaskMismatchError(
            f"Model {oracle_name} is already registered as {existing!r}, but {task!r} was requested; "
            "use --force to replace it."
        )
    return UploadStatus.SKIPPED


def upload_model(
    dsn: DSN,
    model_bytes: bytes,
    oracle_name: str,
    force: bool = False,
    task: str = "embedding",
) -> UploadStatus:
    """Connect to Oracle and register *model_bytes* as *oracle_name*.

    ``task`` controls the metadata JSON: ``"embedding"`` for vector models,
    ``"reranker"`` for two-input regression models queried via PREDICTION.

    If a model with that name already exists:
      - force=False, same task: log and return ``UploadStatus.SKIPPED`` (idempotent no-op).
      - force=False, different task: raise :class:`TaskMismatchError` (a ``ValueError``).
      - force=True: drop it, then upload. The drop is not rolled back if the upload fails; that case
        raises :class:`ModelUploadError` saying the old model is gone.

    Returns ``UploadStatus.REGISTERED`` once the model is registered.
    """
    validate_oracle_name(oracle_name)
    # Reject an unknown task before anything is dropped.
    metadata = build_metadata_json(task)
    logger.info("Connecting to %s ...", dsn.display())
    conn = dsn.connect()
    try:
        existing = registered_task(conn, oracle_name)
        if check_existing(existing, oracle_name, task, force) is UploadStatus.SKIPPED:
            logger.info(
                "Model %s already registered (task=%s) — skipping (use --force to replace).",
                oracle_name,
                existing,
            )
            return UploadStatus.SKIPPED

        dropped = False
        if existing is not None:
            logger.warning(
                "Dropping existing model %s (task=%s) before uploading its replacement; "
                "if the upload fails the old model is NOT restored.",
                oracle_name,
                existing,
            )
            drop_model(conn, oracle_name)
            dropped = True

        logger.info("Uploading %d bytes as %s (task=%s) ...", len(model_bytes), oracle_name, task)
        try:
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
                    "metadata": metadata,
                },
            )
            conn.commit()
        except Exception as exc:
            if dropped:
                raise ModelUploadError(
                    f"Uploading {oracle_name} failed after the existing model was dropped, "
                    f"so {oracle_name} is no longer registered. Cause: {exc}"
                ) from exc
            raise
        logger.info("Model %s registered successfully.", oracle_name)
        return UploadStatus.REGISTERED
    finally:
        conn.close()
