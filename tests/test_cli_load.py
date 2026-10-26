"""`onnx2oracle load`: what happens before, during and after the (slow) build."""

import oracledb
import pytest
from typer.testing import CliRunner

import onnx2oracle.cli as cli
from onnx2oracle.presets import EmbeddingSpec, RerankerSpec

from ._fake_oracle import DSN_ARGS, FakeOracle, isolate_cli

runner = CliRunner()

EMBEDDING_PRESET = "all-MiniLM-L6-v2"
EMBEDDING_NAME = "ALL_MINILM_L6_V2"
RERANKER_PRESET = "ms-marco-MiniLM-L-6-v2"
RERANKER_NAME = "MS_MARCO_MINILM_L_6_V2"


@pytest.fixture(autouse=True)
def _cli_env(monkeypatch, tmp_path):
    isolate_cli(monkeypatch, tmp_path)


@pytest.fixture
def db(monkeypatch):
    return FakeOracle().install(monkeypatch)


class Builds:
    """Stands in for the ONNX builders and records, in ``db.events``, when each build happened."""

    def __init__(self, db: FakeOracle) -> None:
        self._db = db
        self.calls: list[tuple[str, object, dict]] = []

    def embedding(self, spec, **kwargs):
        self.calls.append(("embedding", spec, kwargs))
        self._db.events.append(("build", spec.oracle_name))
        return b"EMBEDDING-ONNX"

    def reranker(self, spec, **kwargs):
        self.calls.append(("reranker", spec, kwargs))
        self._db.events.append(("build", spec.oracle_name))
        return b"RERANKER-ONNX"


@pytest.fixture
def builds(monkeypatch, db):
    recorder = Builds(db)
    monkeypatch.setattr(cli, "build_augmented", recorder.embedding)
    monkeypatch.setattr(cli, "build_reranker", recorder.reranker)
    return recorder


def _load(*args):
    return runner.invoke(cli.app, ["load", *args, *DSN_ARGS])


# --- the registration check happens before the build --------------------------------------------------------------


def test_already_registered_model_is_skipped_without_building(db, builds):
    db.models[EMBEDDING_NAME] = "EMBEDDING"

    result = _load(EMBEDDING_PRESET)

    assert result.exit_code == 0, result.output
    assert "already registered (task embedding), skipping; use --force" in result.stdout
    assert "✓" not in result.stdout  # nothing was registered, so no success line
    assert builds.calls == []
    assert db.kinds("drop", "load") == []
    assert db.open_connections == 0


def test_already_registered_reranker_is_skipped_without_building(db, builds):
    db.models[RERANKER_NAME] = "REGRESSION"

    result = _load(RERANKER_PRESET)

    assert result.exit_code == 0, result.output
    assert "already registered (task reranker), skipping" in result.stdout
    assert builds.calls == []


@pytest.mark.parametrize(
    ("preset", "name", "existing", "requested"),
    [
        (EMBEDDING_PRESET, EMBEDDING_NAME, "REGRESSION", "embedding"),
        (RERANKER_PRESET, RERANKER_NAME, "EMBEDDING", "reranker"),
    ],
)
def test_a_model_of_another_task_holding_the_name_is_an_error_before_building(
    db, builds, preset, name, existing, requested
):
    db.models[name] = existing

    result = _load(preset)

    assert result.exit_code == 1
    assert f"already registered as {'reranker' if existing == 'REGRESSION' else 'embedding'!r}" in result.stdout
    assert f"{requested!r} was requested" in result.stdout
    assert "--force" in result.stdout
    assert builds.calls == []
    assert db.models == {name: existing}
    assert isinstance(result.exception, SystemExit)


def test_new_model_is_checked_then_built_then_uploaded(db, builds):
    result = _load(EMBEDDING_PRESET)

    assert result.exit_code == 0, result.output
    assert f"✓ {EMBEDDING_NAME} registered (embedding)" in result.stdout
    assert db.kinds("connect", "build", "load") == ["connect", "build", "connect", "load"]
    (_, name, data, metadata) = next(e for e in db.events if e[0] == "load")
    assert (name, data, metadata["function"]) == (EMBEDDING_NAME, b"EMBEDDING-ONNX", "embedding")
    assert db.models == {EMBEDDING_NAME: "EMBEDDING"}


def test_reranker_preset_is_built_with_the_reranker_builder(db, builds):
    result = _load(RERANKER_PRESET)

    assert result.exit_code == 0, result.output
    assert [kind for kind, _, _ in builds.calls] == ["reranker"]
    assert db.models == {RERANKER_NAME: "REGRESSION"}


def test_unreachable_database_fails_before_building(db, builds):
    db.connect_error = oracledb.OperationalError("DPY-6005: cannot connect to database")

    result = _load(EMBEDDING_PRESET)

    assert result.exit_code == 1
    assert "DPY-6005" in result.stdout
    assert builds.calls == []
    assert isinstance(result.exception, SystemExit)


def test_oracle_error_text_is_printed_literally_not_as_markup(db, builds):
    db.errors["user_mining_models"] = oracledb.DatabaseError("ORA-00942: [/bold] [red]view missing")

    result = _load(EMBEDDING_PRESET)

    assert result.exit_code == 1
    assert "ORA-00942: [/bold] [red]view missing" in result.stdout
    assert isinstance(result.exception, SystemExit)


def test_dsn_text_that_looks_like_markup_does_not_crash_the_status_lines(db, builds):
    db.models[EMBEDDING_NAME] = "EMBEDDING"
    # `[/x]` would raise MarkupError and `[bold]` would be swallowed if the DSN were printed as markup.
    dsn = "us[bold]er/pw@localhost:1521/FREE[/x]"

    result = runner.invoke(cli.app, ["load", EMBEDDING_PRESET, "--dsn", dsn])

    assert result.exit_code == 0, result.output
    assert "us[bold]er@localhost:1521/FREE[/x]" in result.stdout


# --- --force ------------------------------------------------------------------------------------------------------


def test_force_builds_before_it_drops_and_says_so(db, builds):
    db.models[EMBEDDING_NAME] = "EMBEDDING"

    result = _load(EMBEDDING_PRESET, "--force")

    assert result.exit_code == 0, result.output
    assert db.kinds("build", "drop", "load") == ["build", "drop", "load"]
    assert "dropped only after the new model has been built" in result.stdout
    assert f"✓ {EMBEDDING_NAME} registered (embedding)" in result.stdout


def test_force_may_replace_a_model_registered_for_another_task(db, builds):
    db.models[EMBEDDING_NAME] = "REGRESSION"

    result = _load(EMBEDDING_PRESET, "--force")

    assert result.exit_code == 0, result.output
    assert db.models == {EMBEDDING_NAME: "EMBEDDING"}


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("graph surgery failed"),
        ValueError("core hidden size 768 != --dims 384"),
        NotImplementedError("SentencePiece tokenizers are not supported"),
        OSError("401 Client Error: repository not found"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_force_with_a_failed_build_leaves_the_registered_model_alone(db, monkeypatch, error):
    db.models[EMBEDDING_NAME] = "EMBEDDING"

    def exploding_build(spec, **kwargs):
        raise error

    monkeypatch.setattr(cli, "build_augmented", exploding_build)

    result = _load(EMBEDDING_PRESET, "--force")

    assert result.exit_code == 1
    assert f"Could not build {EMBEDDING_NAME}" in result.stdout
    assert str(error) in result.stdout
    assert isinstance(result.exception, SystemExit)
    assert db.kinds("drop", "load") == []
    assert db.models == {EMBEDDING_NAME: "EMBEDDING"}


def test_failed_reranker_build_is_a_clean_error_too(db, monkeypatch):
    def exploding_build(spec, **kwargs):
        raise RuntimeError("torch is required to export cross-encoders")

    monkeypatch.setattr(cli, "build_reranker", exploding_build)

    result = _load(RERANKER_PRESET)

    assert result.exit_code == 1
    assert "torch is required" in result.stdout
    assert db.kinds("load") == []


def test_force_with_a_failed_upload_tells_the_user_the_old_model_was_dropped(db, builds):
    db.models[EMBEDDING_NAME] = "EMBEDDING"
    db.errors["LOAD_ONNX_MODEL"] = oracledb.DatabaseError("ORA-54426: invalid ONNX graph")

    result = _load(EMBEDDING_PRESET, "--force")

    assert result.exit_code == 1
    assert "dropped" in result.stdout
    assert "no longer registered" in result.stdout
    assert "ORA-54426" in result.stdout
    assert db.models == {}
    assert isinstance(result.exception, SystemExit)  # a clean exit, not a traceback


def test_upload_failure_without_force_is_a_clean_error(db, builds):
    db.errors["LOAD_ONNX_MODEL"] = oracledb.DatabaseError("ORA-54426: invalid ONNX graph")

    result = _load(EMBEDDING_PRESET)

    assert result.exit_code == 1
    assert "ORA-54426" in result.stdout
    assert "dropped" not in result.stdout
    assert isinstance(result.exception, SystemExit)


def test_model_registered_during_the_build_is_reported_as_skipped_not_registered(db, builds, monkeypatch):
    def build_while_someone_else_registers(spec, **kwargs):
        db.models[spec.oracle_name] = "EMBEDDING"
        return b"EMBEDDING-ONNX"

    monkeypatch.setattr(cli, "build_augmented", build_while_someone_else_registers)

    result = _load(EMBEDDING_PRESET)

    assert result.exit_code == 0, result.output
    assert "nothing was uploaded" in result.stdout
    assert "✓" not in result.stdout
    assert db.kinds("load") == []


# --- --no-cache ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", [EMBEDDING_PRESET, RERANKER_PRESET])
def test_builds_use_the_cache_by_default(db, builds, preset):
    assert _load(preset).exit_code == 0

    assert [kwargs["use_cache"] for _, _, kwargs in builds.calls] == [True]


@pytest.mark.parametrize("preset", [EMBEDDING_PRESET, RERANKER_PRESET])
def test_no_cache_flag_disables_the_build_cache(db, builds, preset):
    assert _load(preset, "--no-cache").exit_code == 0

    assert [kwargs["use_cache"] for _, _, kwargs in builds.calls] == [False]


# --- presets cannot be tweaked ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "named"),
    [
        (["--pooling", "cls"], "--pooling"),
        (["--dims", "128"], "--dims"),
        (["--max-length", "128"], "--max-length"),
        (["--normalize"], "--normalize"),
        (["--no-normalize"], "--no-normalize"),
        (["--from-huggingface", "org/other-model"], "--from-huggingface"),
    ],
)
def test_preset_combined_with_custom_model_flags_is_an_error(db, builds, flags, named):
    result = _load(EMBEDDING_PRESET, *flags)

    assert result.exit_code == 2
    assert named in result.stdout
    assert "cannot be combined with preset" in result.stdout
    assert builds.calls == []
    assert db.events == []


def test_all_conflicting_flags_are_named_together(db, builds):
    result = _load(EMBEDDING_PRESET, "--pooling", "cls", "--dims", "128")

    assert result.exit_code == 2
    assert "--pooling, --dims" in result.stdout


def test_preset_may_still_be_renamed_and_forced(db, builds):
    db.models["MY_COPY"] = "EMBEDDING"

    result = _load(EMBEDDING_PRESET, "--name", "MY_COPY", "--force")

    assert result.exit_code == 0, result.output
    assert builds.calls[0][1].oracle_name == "MY_COPY"
    assert db.models == {"MY_COPY": "EMBEDDING"}


def test_unknown_preset_is_a_clean_error_listing_the_choices(db, builds):
    result = _load("no-such-preset")

    assert result.exit_code == 2
    assert "Unknown preset" in result.stdout
    assert EMBEDDING_PRESET in result.stdout
    assert builds.calls == []
    assert isinstance(result.exception, SystemExit)


# --- custom --from-huggingface models -----------------------------------------------------------------------------


def test_custom_embedding_model_defaults(db, builds):
    result = _load("--from-huggingface", "org/My-Model.v2", "--dims", "16")

    assert result.exit_code == 0, result.output
    spec = builds.calls[0][1]
    assert isinstance(spec, EmbeddingSpec)
    assert (spec.hf_repo, spec.dims, spec.pooling, spec.normalize, spec.max_length) == (
        "org/My-Model.v2", 16, "mean", True, 512
    )
    assert spec.oracle_name == "ORG_MY_MODEL_V2"
    assert db.models == {"ORG_MY_MODEL_V2": "EMBEDDING"}


def test_custom_embedding_model_honours_every_override(db, builds):
    result = _load(
        "--from-huggingface", "org/model", "--dims", "32", "--pooling", "cls",
        "--no-normalize", "--max-length", "128", "--name", "CUSTOM",
    )

    assert result.exit_code == 0, result.output
    spec = builds.calls[0][1]
    assert (spec.dims, spec.pooling, spec.normalize, spec.max_length, spec.oracle_name) == (
        32, "cls", False, 128, "CUSTOM"
    )


def test_custom_reranker_honours_max_length(db, builds):
    result = _load(
        "--from-huggingface", "org/cross-encoder", "--task", "reranker", "--max-length", "256", "--name", "MY_RERANKER"
    )

    assert result.exit_code == 0, result.output
    kind, spec, _ = builds.calls[0]
    assert (kind, spec.max_length, spec.oracle_name) == ("reranker", 256, "MY_RERANKER")
    assert isinstance(spec, RerankerSpec)
    assert db.models == {"MY_RERANKER": "REGRESSION"}


def test_custom_embedding_model_requires_dims(db, builds):
    result = _load("--from-huggingface", "org/model")

    assert result.exit_code == 2
    assert "--dims is required" in result.stdout
    assert builds.calls == []


def test_custom_embedding_model_rejects_an_unknown_pooling(db, builds):
    result = _load("--from-huggingface", "org/model", "--dims", "8", "--pooling", "max")

    assert result.exit_code == 2
    assert "--pooling must be 'mean' or 'cls'" in result.stdout


# --- bad names never cost a download ------------------------------------------------------------------------------


def test_invalid_name_is_rejected_before_the_database_or_the_build(db, builds):
    result = _load("--from-huggingface", "org/model", "--dims", "8", "--name", "not-a-valid-name")

    assert result.exit_code == 2
    assert "Invalid Oracle model name" in result.stdout
    assert builds.calls == []
    assert db.events == []


def test_invalid_derived_name_points_at_the_name_option(db, builds):
    result = _load("--from-huggingface", "org/model+x", "--dims", "8")

    assert result.exit_code == 2
    assert "Invalid Oracle model name" in result.stdout
    assert "pass --name" in result.stdout
    assert builds.calls == []
    assert db.events == []


def test_option_values_that_look_like_markup_do_not_crash(db, builds):
    result = _load("--from-huggingface", "org/model", "--dims", "8", "--task", "[/red]")

    assert result.exit_code == 2
    assert "[/red]" in result.stdout
