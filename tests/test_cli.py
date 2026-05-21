from rich.console import Console
from typer.testing import CliRunner

import onnx2oracle.cli as cli
from onnx2oracle.preflight import PreflightCheck, PreflightResult
from onnx2oracle.verify import VerifyResult

runner = CliRunner()


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
    def fake_smoke_test(_dsn, _model_name):
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
