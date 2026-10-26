"""`onnx2oracle verify`, `embed` and `rerank` against a fake Oracle."""

import json
import re
from importlib.metadata import version

import oracledb
import pytest
from typer.testing import CliRunner

import onnx2oracle.cli as cli

from ._fake_oracle import DSN_ARGS, FakeOracle, isolate_cli, make_embedder

runner = CliRunner()

# Click < 8.2 folds stderr into result.stdout, so a stdout-only assertion needs a newer Click.
CLICK_SPLITS_STDERR = tuple(int(part) for part in version("click").split(".")[:2]) >= (8, 2)


@pytest.fixture(autouse=True)
def _cli_env(monkeypatch, tmp_path):
    isolate_cli(monkeypatch, tmp_path)


@pytest.fixture
def db(monkeypatch):
    return FakeOracle({"MY_MODEL": "EMBEDDING", "MY_RERANKER": "REGRESSION"}).install(monkeypatch)


def _verify(*args):
    return runner.invoke(cli.app, ["verify", *args, *DSN_ARGS])


def _embed(*args):
    return runner.invoke(cli.app, ["embed", *args, *DSN_ARGS])


def _rerank(*args):
    return runner.invoke(cli.app, ["rerank", *args, *DSN_ARGS])


# --- verify -------------------------------------------------------------------------------------------------------


def test_verify_passes_for_a_preset_model_and_prints_every_check(monkeypatch):
    FakeOracle({"ALL_MINILM_L6_V2": "EMBEDDING"}, embedder=make_embedder(dims=384)).install(monkeypatch)

    result = _verify()

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "✓ Connected" in out
    assert "✓ Model ALL_MINILM_L6_V2 registered" in out
    assert "✓ Sample embedding: 384 dims (norm=1.0000)" in out
    similarity = r"✓ Similarity sanity \(king/queen > king/banana\): king/queen=0\.99\d\d king/banana=0\.06\d\d"
    assert re.search(similarity, out)


def test_verify_fails_when_the_dimension_differs_from_the_presets(monkeypatch):
    FakeOracle({"ALL_MINILM_L6_V2": "EMBEDDING"}, embedder=make_embedder(dims=8)).install(monkeypatch)

    result = _verify()

    assert result.exit_code == 1
    assert "✗ Sample embedding: 8 dims, expected 384" in result.stdout


def test_verify_expect_dims_overrides_the_preset_default(monkeypatch):
    FakeOracle({"ALL_MINILM_L6_V2": "EMBEDDING"}, embedder=make_embedder(dims=8)).install(monkeypatch)

    assert _verify("--expect-dims", "8").exit_code == 0


def test_verify_checks_dims_only_when_it_has_an_expectation(db):
    assert _verify("--name", "MY_MODEL").exit_code == 0

    result = _verify("--name", "MY_MODEL", "--expect-dims", "384")

    assert result.exit_code == 1
    assert "8 dims, expected 384" in result.stdout


def test_verify_fails_on_unnormalized_output_unless_told_it_is_expected(db):
    db.embedder = make_embedder(dims=8, scale=3.0)

    result = _verify("--name", "MY_MODEL")

    assert result.exit_code == 1
    assert "norm=3.0000; expected 1.0000" in result.stdout
    assert _verify("--name", "MY_MODEL", "--no-normalize").exit_code == 0


def test_verify_reports_a_database_error_without_a_traceback(db):
    db.errors["VECTOR_EMBEDDING"] = oracledb.DatabaseError('ORA-00904: "X": invalid identifier')

    result = _verify("--name", "MY_MODEL")

    assert result.exit_code == 1
    assert 'Error: ORA-00904: "X": invalid identifier' in result.stdout
    assert "✓ Model MY_MODEL registered" in result.stdout
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_verify_prints_error_text_that_looks_like_markup_literally(db):
    db.errors["VECTOR_EMBEDDING"] = oracledb.DatabaseError("ORA-00001: [/bold] oops [red]")

    result = _verify("--name", "MY_MODEL")

    assert result.exit_code == 1
    assert "Error: ORA-00001: [/bold] oops [red]" in result.stdout
    assert isinstance(result.exception, SystemExit)


def test_verify_reports_an_unregistered_model(db):
    result = _verify("--name", "NOT_LOADED")

    assert result.exit_code == 1
    assert "✗ Model NOT_LOADED registered" in result.stdout
    assert "not registered" in result.stdout


def test_verify_passes_a_healthy_reranker(db):
    result = _verify("--name", "MY_RERANKER")

    assert result.exit_code == 0, result.output
    assert "Task: reranker" in result.stdout
    assert "✓ Sample scores: relevant=" in result.stdout
    assert "✓ Relevance sanity (relevant > irrelevant)" in result.stdout


def test_verify_fails_a_reranker_that_scores_the_wrong_way_round(db):
    db.scorer = lambda query, doc: -len(doc)

    result = _verify("--name", "MY_RERANKER")

    assert result.exit_code == 1
    assert "✗ Relevance sanity" in result.stdout


# --- embed --------------------------------------------------------------------------------------------------------


def test_embed_prints_dims_and_norm_per_text_from_one_query(db):
    result = _embed("hello", "second text", "--name", "MY_MODEL")

    assert result.exit_code == 0, result.output
    assert "hello" in result.stdout
    assert "second text" in result.stdout
    assert result.stdout.count("1.0000") == 2
    assert re.search(r"\b8\b", result.stdout)
    assert [e for e in db.events if e[0] == "embed"] == [("embed", 2)]


def test_embed_defaults_to_the_minilm_preset_model(monkeypatch):
    FakeOracle({"ALL_MINILM_L6_V2": "EMBEDDING"}).install(monkeypatch)

    assert _embed("hello").exit_code == 0


def test_embed_json_prints_the_vector_of_every_text(db):
    texts = ["king", "queen", "some other text"]

    result = _embed(*texts, "--name", "MY_MODEL", "--json")

    assert result.exit_code == 0, result.output
    records = json.loads(result.stdout)
    assert [record["text"] for record in records] == texts
    for record, text in zip(records, texts, strict=True):
        assert record["dims"] == 8
        assert record["norm"] == pytest.approx(1.0, abs=1e-6)
        assert record["embedding"] == pytest.approx(make_embedder()(text), abs=1e-6)
    assert [e for e in db.events if e[0] == "embed"] == [("embed", 3)]


@pytest.mark.skipif(not CLICK_SPLITS_STDERR, reason="needs Click >= 8.2 to separate stderr from stdout")
def test_embed_json_failure_leaves_stdout_empty_and_reports_on_stderr(db):
    result = _embed("king", "--name", "NOT_LOADED", "--json")

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "not registered" in result.stderr


@pytest.mark.skipif(not CLICK_SPLITS_STDERR, reason="needs Click >= 8.2 to separate stderr from stdout")
@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--target", "prod"], "--target must be 'local'"),
        (["--dsn", "user/no-host"], "Malformed DSN"),
        (["--dsn", *DSN_ARGS[1:], "--name", "bad name"], "Invalid Oracle model name"),
    ],
)
def test_embed_json_configuration_errors_also_go_to_stderr(db, flags, message):
    result = runner.invoke(cli.app, ["embed", "king", "--json", *flags])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_embed_json_stdout_is_pure_json_even_with_zero_config(monkeypatch):
    FakeOracle({"ALL_MINILM_L6_V2": "EMBEDDING"}).install(monkeypatch)

    result = runner.invoke(cli.app, ["embed", "king", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["text"] == "king"


def test_embed_fails_for_an_unregistered_model(db):
    result = _embed("king", "--name", "NOT_LOADED")

    assert result.exit_code == 1
    assert "'NOT_LOADED' is not registered" in result.stdout
    assert isinstance(result.exception, SystemExit)


def test_embed_reports_an_unreachable_database_without_a_traceback(db):
    db.connect_error = oracledb.OperationalError("DPY-6005: cannot connect to database")

    result = _embed("king", "--name", "MY_MODEL")

    assert result.exit_code == 1
    assert "DPY-6005" in result.stdout
    assert isinstance(result.exception, SystemExit)


def test_embed_refuses_a_reranker_model(db):
    result = _embed("king", "--name", "MY_RERANKER")

    assert result.exit_code == 1
    assert "registered as 'reranker', not an embedding model" in result.stdout
    assert db.kinds("embed") == []


def test_embed_rejects_an_unsafe_model_name_before_connecting(db):
    result = _embed("king", "--name", "x) FROM dual; --")

    assert result.exit_code == 2
    assert "Invalid Oracle model name" in result.stdout
    assert db.events == []


def test_embed_reports_oracle_errors_without_a_traceback(db):
    db.errors["VECTOR_EMBEDDING"] = oracledb.DatabaseError("ORA-12345: boom")

    result = _embed("king", "--name", "MY_MODEL")

    assert result.exit_code == 1
    assert "ORA-12345: boom" in result.stdout
    assert isinstance(result.exception, SystemExit)
    assert db.open_connections == 0


def test_embed_fails_when_oracle_returns_null_for_a_text(db):
    result = _embed("king", "", "--name", "MY_MODEL")

    assert result.exit_code == 1
    assert "NULL" in result.stdout


def test_embed_json_marks_a_null_embedding_as_null(db):
    result = _embed("king", "", "--name", "MY_MODEL", "--json")

    assert result.exit_code == 1
    records = json.loads(result.stdout)
    assert records[0]["dims"] == 8
    assert records[1] == {"text": "", "dims": None, "norm": None, "embedding": None}


def test_embed_shows_text_that_looks_like_markup_literally(db):
    result = _embed("[/bold] [red]x", "--name", "MY_MODEL")

    assert result.exit_code == 0, result.output
    assert "[/bold] [red]x" in result.stdout


def test_embed_needs_at_least_one_text(db):
    assert runner.invoke(cli.app, ["embed", *DSN_ARGS]).exit_code == 2


# --- rerank -------------------------------------------------------------------------------------------------------


def test_rerank_lists_the_best_document_first(db):
    result = _rerank(
        "--name", "MY_RERANKER",
        "--query", "How many people live in Berlin?",
        "--doc", "Bananas are a tropical fruit.",
        "--doc", "Berlin has 3.7 million people.",
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.index("Berlin has") < result.stdout.index("Bananas")


def test_rerank_unregistered_model_is_a_clean_error(db):
    result = _rerank("--name", "NOT_LOADED", "--query", "q", "--doc", "d")

    assert result.exit_code == 1
    assert "'NOT_LOADED' is not registered" in result.stdout
    assert isinstance(result.exception, SystemExit)


def test_rerank_refuses_an_embedding_model_cleanly(db):
    result = _rerank("--name", "MY_MODEL", "--query", "q", "--doc", "d")

    assert result.exit_code == 1
    assert "registered as 'embedding', not a reranker" in result.stdout
    assert isinstance(result.exception, SystemExit)


def test_rerank_shows_query_and_documents_that_look_like_markup_literally(db):
    result = _rerank("--name", "MY_RERANKER", "--query", "[/x] query", "--doc", "[/bold] [red]doc")

    assert result.exit_code == 0, result.output
    assert "[/x] query" in result.stdout
    assert "[/bold] [red]doc" in result.stdout
