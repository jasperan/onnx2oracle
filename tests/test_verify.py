"""Unit tests for verify.py, using an in-memory fake of Oracle (no DB needed)."""

import math

import oracledb
import pytest

from onnx2oracle.connection import DSN
from onnx2oracle.verify import VerifyResult, embed_texts, smoke_test

from ._fake_oracle import FakeOracle, make_embedder

DSN_UNDER_TEST = DSN.parse("system/pw@localhost:1521/FREEPDB1")


def _healthy(**overrides) -> VerifyResult:
    fields = {
        "connected": True,
        "model_registered": True,
        "sample_embedding_dims": 384,
        "sample_embedding_norm": 1.0,
        "similarity_sane": True,
        "elapsed_ms": 5,
        "task": "embedding",
    }
    fields.update(overrides)
    return VerifyResult(**fields)


# --- VerifyResult.ok -------------------------------------------------------------------------------------------


def test_healthy_embedding_result_is_ok():
    assert _healthy().ok


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_embedding_norm": 0.5},
        {"sample_embedding_norm": 1.002},
        {"sample_embedding_norm": float("nan")},
        {"sample_embedding_norm": float("inf")},
        {"sample_embedding_norm": None},
        {"sample_embedding_dims": 0},
        {"sample_embedding_dims": None},
        {"expected_dims": 768},
        {"similarity_sane": False},
        {"error": "boom"},
        {"connected": False},
        {"model_registered": False},
    ],
)
def test_embedding_result_is_not_ok(overrides):
    assert not _healthy(**overrides).ok


def test_norm_within_tolerance_is_ok():
    assert _healthy(sample_embedding_norm=1.0005).ok
    assert _healthy(sample_embedding_norm=0.9995).ok


def test_norm_is_not_enforced_when_the_model_is_not_normalized():
    assert _healthy(sample_embedding_norm=5.0, expect_normalized=False).ok
    assert not _healthy(sample_embedding_norm=float("nan"), expect_normalized=False).ok


def test_matching_expected_dims_is_ok():
    assert _healthy(expected_dims=384).ok


def test_reranker_result_needs_finite_scores_and_sane_ordering():
    def reranker(sane, scores):
        return VerifyResult(True, True, None, None, sane, 1, task="reranker", sample_scores=scores)

    assert reranker(True, (2.0, -1.0)).ok
    assert not reranker(False, (-1.0, 2.0)).ok
    assert not reranker(True, None).ok
    assert not reranker(True, (float("inf"), 0.0)).ok


# --- smoke_test: embedding --------------------------------------------------------------------------------------


@pytest.fixture
def db(monkeypatch):
    return FakeOracle({"MY_MODEL": "EMBEDDING"}).install(monkeypatch)


def test_smoke_test_passes_for_a_healthy_embedding_model(db):
    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert result.ok, result
    assert (result.connected, result.model_registered, result.task) == (True, True, "embedding")
    assert result.sample_embedding_dims == 8
    assert result.sample_embedding_norm == pytest.approx(1.0, abs=1e-6)
    king_queen, king_banana = result.similarities
    assert king_queen > king_banana
    assert db.open_connections == 0


def test_smoke_test_embeds_all_probe_texts_in_one_query(db):
    smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert [e for e in db.events if e[0] == "embed"] == [("embed", 4)]
    assert sum("VECTOR_EMBEDDING" in s for s in db.statements) == 1


def test_smoke_test_fails_when_the_output_is_not_normalized(db):
    db.embedder = make_embedder(dims=8, scale=3.0)

    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert result.error is None
    assert result.sample_embedding_norm == pytest.approx(3.0, abs=1e-5)
    assert not result.ok


def test_smoke_test_accepts_unnormalized_output_when_told_to(db):
    db.embedder = make_embedder(dims=8, scale=3.0)

    assert smoke_test(DSN_UNDER_TEST, "MY_MODEL", expect_normalized=False).ok


def test_smoke_test_checks_the_expected_dims(db):
    assert smoke_test(DSN_UNDER_TEST, "MY_MODEL", expected_dims=8).ok

    mismatch = smoke_test(DSN_UNDER_TEST, "MY_MODEL", expected_dims=384)
    assert mismatch.sample_embedding_dims == 8
    assert mismatch.expected_dims == 384
    assert not mismatch.ok


def test_smoke_test_fails_when_embeddings_are_not_finite(db):
    base = make_embedder(dims=8)
    db.embedder = lambda text: [math.nan] + [0.0] * 7 if text == "hello world" else base(text)

    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert result.error is None
    assert math.isnan(result.sample_embedding_norm)
    assert not result.ok


def test_smoke_test_reports_a_null_embedding_as_an_error(db):
    base = make_embedder()
    db.embedder = lambda text: None if text == "queen" else base(text)

    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert "returned null" in result.error
    assert "queen" in result.error
    assert not result.ok


def test_smoke_test_reports_inconsistent_dimensions_instead_of_crashing(db):
    base = make_embedder(dims=8)
    db.embedder = lambda text: base(text)[:5] if text == "banana" else base(text)

    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert "different lengths" in result.error
    assert not result.ok


def test_smoke_test_turns_a_failing_inference_query_into_a_result_error(db):
    db.errors["VECTOR_EMBEDDING"] = oracledb.DatabaseError('ORA-00904: "X": invalid identifier')

    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert "ORA-00904" in result.error
    assert (result.connected, result.model_registered, result.task) == (True, True, "embedding")
    assert not result.ok
    assert db.open_connections == 0


def test_smoke_test_turns_a_failing_catalog_query_into_a_result_error(db):
    db.errors["user_mining_models"] = oracledb.DatabaseError("ORA-00942: table or view does not exist")

    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert "ORA-00942" in result.error
    assert result.connected
    assert not result.model_registered
    assert not result.ok
    assert db.open_connections == 0


def test_smoke_test_reports_unexpected_exceptions_with_their_type(db):
    db.errors["VECTOR_EMBEDDING"] = RuntimeError()

    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert result.error == "RuntimeError"
    assert not result.ok


def test_smoke_test_reports_connection_failure(db):
    db.connect_error = oracledb.OperationalError("DPY-6005: cannot connect to database")

    result = smoke_test(DSN_UNDER_TEST, "MY_MODEL")

    assert "DPY-6005" in result.error
    assert not result.connected
    assert not result.ok


def test_smoke_test_reports_an_unregistered_model(db):
    result = smoke_test(DSN_UNDER_TEST, "MISSING_MODEL")

    assert result.connected
    assert not result.model_registered
    assert "not registered" in result.error
    assert not result.ok


def test_smoke_test_rejects_an_unsafe_model_name_without_connecting(db):
    result = smoke_test(DSN_UNDER_TEST, "bad name; DROP TABLE x")

    assert "Invalid Oracle model name" in result.error
    assert not result.connected
    assert db.events == []


# --- smoke_test: reranker ---------------------------------------------------------------------------------------


def test_smoke_test_passes_for_a_healthy_reranker(monkeypatch):
    db = FakeOracle({"MY_RERANKER": "REGRESSION"}).install(monkeypatch)

    result = smoke_test(DSN_UNDER_TEST, "MY_RERANKER")

    assert result.ok, result
    assert result.task == "reranker"
    relevant, irrelevant = result.sample_scores
    assert relevant > irrelevant
    assert db.open_connections == 0


def test_smoke_test_fails_a_reranker_that_orders_documents_wrongly(monkeypatch):
    FakeOracle({"MY_RERANKER": "REGRESSION"}, scorer=lambda q, d: -len(d)).install(monkeypatch)

    result = smoke_test(DSN_UNDER_TEST, "MY_RERANKER")

    assert result.error is None
    assert not result.similarity_sane
    assert not result.ok


def test_smoke_test_reports_a_null_reranker_score(monkeypatch):
    FakeOracle({"MY_RERANKER": "REGRESSION"}, scorer=lambda q, d: None).install(monkeypatch)

    result = smoke_test(DSN_UNDER_TEST, "MY_RERANKER")

    assert "PREDICTION returned null" in result.error
    assert not result.ok


def test_smoke_test_turns_a_failing_reranker_query_into_a_result_error(monkeypatch):
    db = FakeOracle({"MY_RERANKER": "REGRESSION"}).install(monkeypatch)
    db.errors["PREDICTION"] = oracledb.DatabaseError("ORA-40284: model does not exist")

    result = smoke_test(DSN_UNDER_TEST, "MY_RERANKER")

    assert "ORA-40284" in result.error
    assert result.task == "reranker"
    assert not result.ok


# --- embed_texts ------------------------------------------------------------------------------------------------


def test_embed_texts_returns_vectors_in_input_order_whatever_order_rows_arrive_in(monkeypatch):
    db = FakeOracle({"MY_MODEL": "EMBEDDING"}, shuffle_rows=True).install(monkeypatch)
    conn = DSN_UNDER_TEST.connect()
    expected = make_embedder()
    texts = ["king", "banana", "queen"]

    vectors = embed_texts(conn, "MY_MODEL", texts)

    for text, vec in zip(texts, vectors, strict=True):
        assert vec == pytest.approx(expected(text), abs=1e-6)
    assert db.kinds("embed") == ["embed"]


def test_embed_texts_returns_an_empty_vector_for_a_null_result(db):
    conn = DSN_UNDER_TEST.connect()

    vectors = embed_texts(conn, "MY_MODEL", ["king", ""])

    assert len(vectors[0]) == 8
    assert vectors[1] == []


def test_embed_texts_with_no_texts_runs_no_query(db):
    conn = DSN_UNDER_TEST.connect()

    assert embed_texts(conn, "MY_MODEL", []) == []
    assert db.statements == []


def test_embed_texts_rejects_an_unsafe_model_name_before_querying(db):
    conn = DSN_UNDER_TEST.connect()

    with pytest.raises(ValueError, match="Invalid Oracle model name"):
        embed_texts(conn, "x USING 1 AS DATA) FROM dual --", ["king"])

    assert db.statements == []
