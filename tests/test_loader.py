"""Unit tests for loader.py, using an in-memory fake of Oracle (no DB needed).

Real-database coverage is in tests/test_loader_integration.py.
"""

import json
import logging

import oracledb
import pytest

from onnx2oracle.connection import DSN
from onnx2oracle.loader import (
    ModelUploadError,
    TaskMismatchError,
    UploadStatus,
    build_metadata_json,
    check_existing,
    lookup_registered_task,
    upload_model,
)

from ._fake_oracle import FakeOracle

DSN_UNDER_TEST = DSN.parse("system/pw@localhost:1521/FREEPDB1")


@pytest.fixture
def db(monkeypatch):
    return FakeOracle().install(monkeypatch)


def test_metadata_json_embedding_shape():
    js = build_metadata_json()
    parsed = json.loads(js)
    assert parsed["function"] == "embedding"
    assert parsed["embeddingOutput"] == "embedding"
    assert parsed["input"] == {"pre_text": ["DATA"]}


def test_metadata_json_embedding_explicit_task():
    parsed = json.loads(build_metadata_json("embedding"))
    assert parsed["function"] == "embedding"


def test_metadata_json_reranker_shape():
    parsed = json.loads(build_metadata_json("reranker"))
    assert parsed["function"] == "regression"
    assert parsed["regressionOutput"] == "logits"
    assert parsed["input"] == {
        "pre_text_1": ["DATA1"],
        "pre_text_2": ["DATA2"],
    }


def test_metadata_json_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        build_metadata_json("classification")


def test_metadata_is_valid_json_string():
    json.loads(build_metadata_json())
    json.loads(build_metadata_json("reranker"))


def test_upload_registers_a_new_embedding_model(db):
    status = upload_model(DSN_UNDER_TEST, b"onnx-bytes", "MY_MODEL")

    assert status is UploadStatus.REGISTERED
    assert status == "registered"
    assert db.models == {"MY_MODEL": "EMBEDDING"}
    (_, name, data, metadata) = [e for e in db.events if e[0] == "load"][0]
    assert (name, data) == ("MY_MODEL", b"onnx-bytes")
    assert metadata["function"] == "embedding"
    assert db.commits >= 1
    assert db.open_connections == 0


def test_upload_registers_a_reranker_with_regression_metadata(db):
    assert upload_model(DSN_UNDER_TEST, b"x", "MY_RERANKER", task="reranker") is UploadStatus.REGISTERED

    assert db.models == {"MY_RERANKER": "REGRESSION"}


def test_upload_skips_when_same_task_already_registered(db):
    db.models["MY_MODEL"] = "EMBEDDING"

    status = upload_model(DSN_UNDER_TEST, b"new-bytes", "MY_MODEL")

    assert status is UploadStatus.SKIPPED
    assert status == "skipped"
    assert db.kinds("drop", "load") == []
    assert db.open_connections == 0


def test_upload_refuses_when_a_model_of_another_task_holds_the_name(db):
    db.models["MY_MODEL"] = "REGRESSION"

    with pytest.raises(TaskMismatchError, match="reranker.*embedding.*--force") as exc_info:
        upload_model(DSN_UNDER_TEST, b"x", "MY_MODEL", task="embedding")

    assert isinstance(exc_info.value, ValueError)
    assert db.models == {"MY_MODEL": "REGRESSION"}
    assert db.kinds("drop", "load") == []
    assert db.open_connections == 0


def test_force_replaces_a_model_even_when_the_task_differs(db):
    db.models["MY_MODEL"] = "REGRESSION"

    status = upload_model(DSN_UNDER_TEST, b"x", "MY_MODEL", force=True, task="embedding")

    assert status is UploadStatus.REGISTERED
    assert db.kinds("drop", "load") == ["drop", "load"]
    assert db.models == {"MY_MODEL": "EMBEDDING"}


def test_force_logs_a_warning_before_dropping(db):
    db.models["MY_MODEL"] = "EMBEDDING"

    class _Recorder(logging.Handler):
        def emit(self, record):
            db.events.append(("log", record.levelname, record.getMessage()))

    logger = logging.getLogger("onnx2oracle.loader")
    handler = _Recorder()
    logger.addHandler(handler)
    try:
        upload_model(DSN_UNDER_TEST, b"x", "MY_MODEL", force=True)
    finally:
        logger.removeHandler(handler)

    warning_at = next(i for i, e in enumerate(db.events) if e[:2] == ("log", "WARNING"))
    drop_at = next(i for i, e in enumerate(db.events) if e[0] == "drop")
    assert warning_at < drop_at
    assert "MY_MODEL" in db.events[warning_at][2]
    assert "NOT restored" in db.events[warning_at][2]


def test_failed_load_after_a_forced_drop_says_the_old_model_is_gone(db):
    db.models["MY_MODEL"] = "EMBEDDING"
    db.errors["LOAD_ONNX_MODEL"] = oracledb.DatabaseError("ORA-54426: invalid ONNX graph")

    with pytest.raises(ModelUploadError) as exc_info:
        upload_model(DSN_UNDER_TEST, b"x", "MY_MODEL", force=True)

    message = str(exc_info.value)
    assert "dropped" in message
    assert "no longer registered" in message
    assert "ORA-54426" in message
    assert isinstance(exc_info.value.__cause__, oracledb.DatabaseError)
    assert db.models == {}
    assert db.open_connections == 0


def test_failed_load_without_a_drop_propagates_the_database_error(db):
    db.errors["LOAD_ONNX_MODEL"] = oracledb.DatabaseError("ORA-54426: invalid ONNX graph")

    with pytest.raises(oracledb.DatabaseError, match="ORA-54426"):
        upload_model(DSN_UNDER_TEST, b"x", "MY_MODEL")

    assert db.open_connections == 0


def test_unknown_task_with_force_fails_before_anything_is_dropped(db):
    db.models["MY_MODEL"] = "EMBEDDING"

    with pytest.raises(ValueError, match="Unknown task"):
        upload_model(DSN_UNDER_TEST, b"x", "MY_MODEL", force=True, task="classification")

    assert db.events == []
    assert db.models == {"MY_MODEL": "EMBEDDING"}


def test_unsafe_model_name_is_rejected_before_connecting(db):
    with pytest.raises(ValueError, match="Invalid Oracle model name"):
        upload_model(DSN_UNDER_TEST, b"x", "bad-name; DROP TABLE x")

    assert db.events == []


@pytest.mark.parametrize(
    ("existing", "task", "force", "expected"),
    [
        (None, "embedding", False, None),
        (None, "embedding", True, None),
        ("embedding", "embedding", False, UploadStatus.SKIPPED),
        ("embedding", "embedding", True, None),
        ("reranker", "embedding", True, None),
    ],
)
def test_check_existing_decisions(existing, task, force, expected):
    assert check_existing(existing, "M", task, force) is expected


def test_check_existing_raises_on_task_mismatch_without_force():
    with pytest.raises(TaskMismatchError):
        check_existing("embedding", "M", "reranker", False)


def test_lookup_registered_task_reports_the_task_and_closes_the_connection(db):
    db.models.update({"EMB": "EMBEDDING", "RER": "REGRESSION"})

    assert lookup_registered_task(DSN_UNDER_TEST, "EMB") == "embedding"
    assert lookup_registered_task(DSN_UNDER_TEST, "RER") == "reranker"
    assert lookup_registered_task(DSN_UNDER_TEST, "MISSING") is None
    assert db.open_connections == 0
