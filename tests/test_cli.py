import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

import onnx2oracle.cli as cli
from onnx2oracle.preflight import PreflightCheck, PreflightResult
from onnx2oracle.presets import PRESETS
from onnx2oracle.verify import VerifyResult

from ._fake_oracle import FakeOracle, isolate_cli

runner = CliRunner()


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Detach the CLI from the developer's ORACLE_* variables, config file and terminal width."""
    return isolate_cli(monkeypatch, tmp_path)


def test_cli_help_succeeds():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "onnx2oracle" in result.stdout.lower()


def test_presets_command_lists_all(monkeypatch):
    monkeypatch.setattr(cli, "console", Console(width=200))
    result = runner.invoke(cli.app, ["presets"])
    assert result.exit_code == 0
    for name in [
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "all-mpnet-base-v2",
        "bge-small-en-v1.5",
        "nomic-embed-text-v1",
        "ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2",
    ]:
        assert name in result.stdout
    assert "reranker" in result.stdout
    assert "embedding" in result.stdout


def test_rerank_command_requires_doc():
    result = runner.invoke(
        cli.app, ["rerank", "--name", "MS_MARCO_MINILM_L_6_V2", "--query", "hi"]
    )
    assert result.exit_code != 0


def test_rerank_command_rejects_unsafe_model_name(monkeypatch):
    result = runner.invoke(
        cli.app,
        [
            "rerank",
            "--name", "bad-name!",
            "--query", "hi",
            "--doc", "doc1",
            "--dsn", "system/pw@localhost:1521/FREEPDB1",
        ],
    )
    assert result.exit_code != 0


def test_rerank_command_orders_by_score(monkeypatch):
    captured: dict[str, object] = {}

    def fake_score_pairs(_dsn, model_name, query, docs):
        captured["model_name"] = model_name
        captured["query"] = query
        captured["docs"] = list(docs)
        return [(2.5, docs[1]), (-1.0, docs[0])]

    monkeypatch.setattr(cli, "_score_pairs", fake_score_pairs)

    result = runner.invoke(
        cli.app,
        [
            "rerank",
            "--name", "MS_MARCO_MINILM_L_6_V2",
            "--query", "what is panda?",
            "--doc", "pandas eat bamboo",
            "--doc", "the eiffel tower",
            "--dsn", "system/pw@localhost:1521/FREEPDB1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["model_name"] == "MS_MARCO_MINILM_L_6_V2"
    assert captured["query"] == "what is panda?"
    assert captured["docs"] == ["pandas eat bamboo", "the eiffel tower"]


def test_load_rejects_invalid_task():
    result = runner.invoke(
        cli.app,
        [
            "load",
            "--from-huggingface", "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "--task", "summarization",
            "--dsn", "system/pw@localhost:1521/FREEPDB1",
        ],
    )
    assert result.exit_code != 0
    assert "embedding" in result.stdout
    assert "reranker" in result.stdout


def test_load_requires_preset_or_from_hf():
    result = runner.invoke(cli.app, ["load"])
    assert result.exit_code != 0


def test_preflight_command_prints_checks(monkeypatch):
    def fake_preflight(_dsn):
        return PreflightResult(
            checks=[
                PreflightCheck("connect", True, "system@localhost:1521/FREEPDB1"),
                PreflightCheck("DBMS_VECTOR package", True, "SYS.DBMS_VECTOR:VALID"),
            ],
            elapsed_ms=12,
        )

    monkeypatch.setattr(cli, "run_preflight", fake_preflight)

    result = runner.invoke(cli.app, ["preflight", "--dsn", "system/pw@localhost:1521/FREEPDB1"])

    assert result.exit_code == 0
    assert "connect" in result.stdout
    assert "SYS.DBMS_VECTOR:VALID" in result.stdout


def test_preflight_command_fails_when_a_check_fails(monkeypatch):
    def fake_preflight(_dsn):
        return PreflightResult(
            checks=[PreflightCheck("VECTOR_EMBEDDING function", False, "not found")],
            elapsed_ms=12,
        )

    monkeypatch.setattr(cli, "run_preflight", fake_preflight)

    result = runner.invoke(cli.app, ["preflight", "--dsn", "system/pw@localhost:1521/FREEPDB1"])

    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_verify_command_fails_when_similarity_sanity_fails(monkeypatch):
    def fake_smoke_test(_dsn, _model_name, **_kwargs):
        return VerifyResult(
            connected=True,
            model_registered=True,
            sample_embedding_dims=384,
            sample_embedding_norm=1.0,
            similarity_sane=False,
            elapsed_ms=12,
        )

    monkeypatch.setattr(cli, "smoke_test", fake_smoke_test)

    result = runner.invoke(cli.app, ["verify", "--dsn", "system/pw@localhost:1521/FREEPDB1"])

    assert result.exit_code == 1
    assert "Similarity sanity" in result.stdout


def test_docker_up_no_wait_only_starts_compose(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_compose(*args: str) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "_compose", fake_compose)

    result = runner.invoke(cli.app, ["docker", "up", "--no-wait"])

    assert result.exit_code == 0
    assert calls == [("up", "-d")]


def test_docker_up_wait_uses_bounded_sql_probe(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_compose(*args: str) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "_compose", fake_compose)

    result = runner.invoke(cli.app, ["docker", "up", "--wait-timeout", "7", "--wait-interval", "2"])

    assert result.exit_code == 0
    assert calls[0] == ("up", "-d")
    assert calls[1][:4] == ("exec", "-T", "oracle", "bash")
    probe = calls[1][-1]
    assert "SECONDS + 7" in probe
    assert "sleep 2" in probe
    assert "${ORACLE_PWD:-onnx2oracle}" in probe
    assert "sqlplus -L -S" in probe


def test_docker_down_can_remove_volumes(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_compose(*args: str) -> int:
        calls.append(args)
        return 0

    monkeypatch.setattr(cli, "_compose", fake_compose)

    result = runner.invoke(cli.app, ["docker", "down", "--volumes"])

    assert result.exit_code == 0
    assert calls == [("down", "--volumes")]


# --- presets --json -----------------------------------------------------------------------------------------------


def test_presets_json_is_a_clean_array_with_the_documented_fields():
    result = runner.invoke(cli.app, ["presets", "--json"])

    assert result.exit_code == 0, result.output
    records = json.loads(result.stdout)
    assert [record["name"] for record in records] == list(PRESETS)
    expected_keys = {
        "name", "hf_repo", "task", "pooling", "dims", "max_length", "normalize", "oracle_name", "approx_size_mb",
    }
    assert all(set(record) == expected_keys for record in records)


def test_presets_json_describes_an_embedding_preset():
    records = {r["name"]: r for r in json.loads(runner.invoke(cli.app, ["presets", "--json"]).stdout)}

    assert records["all-MiniLM-L6-v2"] == {
        "name": "all-MiniLM-L6-v2",
        "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "task": "embedding",
        "pooling": "mean",
        "dims": 384,
        "max_length": 512,
        "normalize": True,
        "oracle_name": "ALL_MINILM_L6_V2",
        "approx_size_mb": 90,
    }
    assert records["bge-small-en-v1.5"]["pooling"] == "cls"


def test_presets_json_has_null_embedding_fields_for_rerankers():
    records = {r["name"]: r for r in json.loads(runner.invoke(cli.app, ["presets", "--json"]).stdout)}

    assert records["ms-marco-MiniLM-L-6-v2"] == {
        "name": "ms-marco-MiniLM-L-6-v2",
        "hf_repo": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "task": "reranker",
        "pooling": None,
        "dims": None,
        "max_length": 512,
        "normalize": None,
        "oracle_name": "MS_MARCO_MINILM_L_6_V2",
        "approx_size_mb": 90,
    }


# --- --target / DSN handling -------------------------------------------------------------------------------------

ORACLE_DSN_ENV = "envuser/envpw@remote.example.com:1522/ENVSVC"

DB_COMMANDS = {
    "verify": ["verify"],
    "embed": ["embed", "hello"],
    "rerank": ["rerank", "--query", "q", "--doc", "d", "--name", "MY_RERANKER"],
    "load": ["load", "all-MiniLM-L6-v2"],
}


def _db_with_models(monkeypatch):
    models = {"ALL_MINILM_L6_V2": "EMBEDDING", "MY_RERANKER": "REGRESSION"}
    return FakeOracle(models).install(monkeypatch)


@pytest.mark.parametrize("argv", DB_COMMANDS.values(), ids=DB_COMMANDS.keys())
def test_explicit_target_local_wins_over_the_oracle_dsn_env(monkeypatch, cli_env, argv):
    monkeypatch.setenv("ORACLE_DSN", ORACLE_DSN_ENV)
    db = _db_with_models(monkeypatch)

    runner.invoke(cli.app, [*argv, "--target", "local"])

    assert {(d.host, d.port) for d in db.dsns} == {("localhost", 1521)}


@pytest.mark.parametrize("argv", DB_COMMANDS.values(), ids=DB_COMMANDS.keys())
def test_without_target_the_oracle_dsn_env_is_used(monkeypatch, cli_env, argv):
    monkeypatch.setenv("ORACLE_DSN", ORACLE_DSN_ENV)
    db = _db_with_models(monkeypatch)

    runner.invoke(cli.app, argv)

    assert {(d.host, d.port) for d in db.dsns} == {("remote.example.com", 1522)}


def test_preflight_explicit_target_local_wins_over_the_oracle_dsn_env(monkeypatch, cli_env):
    monkeypatch.setenv("ORACLE_DSN", ORACLE_DSN_ENV)
    seen = []

    def fake_preflight(dsn):
        seen.append(dsn)
        return PreflightResult(checks=[PreflightCheck("connect", True, "ok")], elapsed_ms=1)

    monkeypatch.setattr(cli, "run_preflight", fake_preflight)

    assert runner.invoke(cli.app, ["preflight", "--target", "local"]).exit_code == 0
    assert runner.invoke(cli.app, ["preflight"]).exit_code == 0

    assert [(d.host, d.port) for d in seen] == [("localhost", 1521), ("remote.example.com", 1522)]


def test_zero_config_still_defaults_to_the_local_database(monkeypatch, cli_env):
    db = FakeOracle().install(monkeypatch)

    result = runner.invoke(cli.app, ["verify"])

    assert "defaulting to --target local" in result.stdout
    assert {(d.user, d.host, d.port) for d in db.dsns} == {("system", "localhost", 1521)}


def test_unknown_target_is_rejected_instead_of_silently_ignored(monkeypatch, cli_env):
    monkeypatch.setenv("ORACLE_DSN", ORACLE_DSN_ENV)
    db = _db_with_models(monkeypatch)

    result = runner.invoke(cli.app, ["verify", "--target", "prod"])

    assert result.exit_code == 2
    assert "--target must be 'local'" in result.stdout
    assert db.events == []


def test_malformed_dsn_is_a_clean_error_that_does_not_echo_the_secret(cli_env):
    result = runner.invoke(cli.app, ["verify", "--dsn", "user/S3cr3t-no-host"])

    assert result.exit_code == 2
    assert "Malformed DSN" in result.stdout
    assert "S3cr3t" not in result.output
    assert isinstance(result.exception, SystemExit)
