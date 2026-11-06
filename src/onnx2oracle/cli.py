"""Typer-based CLI for onnx2oracle."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import date, datetime, time
from importlib.resources import files
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from onnx2oracle import __version__
from onnx2oracle._ident import validate_oracle_name
from onnx2oracle.connection import DEFAULT_CONFIG_PATH, DSN, resolve_dsn
from onnx2oracle.loader import (
    TaskMismatchError,
    UploadStatus,
    check_existing,
    lookup_registered_task,
    registered_task,
    upload_model,
)
from onnx2oracle.pipeline import build_augmented, build_reranker
from onnx2oracle.preflight import run_preflight
from onnx2oracle.presets import PRESETS, EmbeddingSpec, ModelSpec, Pooling, RerankerSpec, Task, get_preset
from onnx2oracle.verify import embed_texts, smoke_test

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

app = typer.Typer(
    name="onnx2oracle",
    help="Load ONNX embedding and reranker models into Oracle AI Database.",
    no_args_is_help=True,
    # A traceback must never print locals: they can hold a DSN with its password.
    pretty_exceptions_show_locals=False,
)
docker_app = typer.Typer(help="Manage a local Oracle AI Database Free container.")
config_app = typer.Typer(help="Manage ~/.onnx2oracle/config.toml.")
app.add_typer(docker_app, name="docker")
app.add_typer(config_app, name="config")

console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)

DEFAULT_MODEL_NAME = "ALL_MINILM_L6_V2"
LOCAL_TARGET = "local"


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _inert(text: str) -> str:
    """Drop C0/C1 control bytes (ESC, CSI, OSC...) so server text cannot drive the terminal."""
    return _CONTROL_CHARS.sub("", text)


def _mark(ok: bool) -> Text:
    return Text("✓", style="green") if ok else Text("✗", style="red")


def _print_error(message: str, *, to_stderr: bool = False) -> None:
    """Print ``Error: <message>``. *message* is external text and is never parsed as Rich markup."""
    out = Console(stderr=True) if to_stderr else console
    out.print(Text.assemble(("Error: ", "red"), _inert(message)), soft_wrap=True)


def _fail(message: str, code: int = 1, *, to_stderr: bool = False) -> NoReturn:
    _print_error(message, to_stderr=to_stderr)
    raise typer.Exit(code)


def _parse_task(value: str) -> Task:
    if value not in ("embedding", "reranker"):
        _fail(f"--task must be 'embedding' or 'reranker', got {value!r}", 2)
    return value


def _parse_pooling(value: str) -> Pooling:
    if value not in ("mean", "cls"):
        _fail(f"--pooling must be 'mean' or 'cls', got {value!r}", 2)
    return value


def _parse_target(value: str | None, *, to_stderr: bool = False) -> str | None:
    if value is not None and value != LOCAL_TARGET:
        _fail(
            f"--target must be '{LOCAL_TARGET}' (use --dsn for any other database), got {value!r}",
            2,
            to_stderr=to_stderr,
        )
    return value


def _resolve_target_dsn(
    dsn: str | None, target: str | None, announce: bool = True, *, to_stderr: bool = False
) -> DSN:
    """Resolve a DSN, defaulting to the local docker-compose target when nothing
    else is configured. ``announce`` prints a note when the local fallback fires;
    ``to_stderr`` sends error messages to stderr (for commands whose stdout is machine-read).

    ``target`` reaches ``resolve_dsn`` only when the user passed ``--target``; an explicit
    target beats ORACLE_DSN and the config file, the implicit default does not.
    """
    target = _parse_target(target, to_stderr=to_stderr)
    if (
        announce
        and target is None
        and dsn is None
        and "ORACLE_DSN" not in os.environ
        and not DEFAULT_CONFIG_PATH.exists()
    ):
        console.print("[dim]No DSN configured — defaulting to --target local (docker-compose).[/dim]")
    try:
        return resolve_dsn(cli_dsn=dsn, target=target, config_path=DEFAULT_CONFIG_PATH, default_local=True)
    except ValueError as exc:
        _fail(str(exc), 2, to_stderr=to_stderr)


def _preset_for_oracle_name(oracle_name: str) -> ModelSpec | None:
    return next((spec for spec in PRESETS.values() if spec.oracle_name == oracle_name), None)


def _preset_record(name: str, spec: ModelSpec) -> dict[str, object]:
    """One entry of ``presets --json``. Reranker specs have no pooling/dims/normalize: those are null."""
    if isinstance(spec, EmbeddingSpec):
        pooling: str | None = spec.pooling
        dims: int | None = spec.dims
        normalize: bool | None = spec.normalize
    else:
        pooling, dims, normalize = None, None, None
    return {
        "name": name,
        "hf_repo": spec.hf_repo,
        "task": spec.task,
        "pooling": pooling,
        "dims": dims,
        "max_length": spec.max_length,
        "normalize": normalize,
        "oracle_name": spec.oracle_name,
        "approx_size_mb": spec.approx_size_mb,
    }


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"onnx2oracle {__version__}")


@app.command()
def presets(
    as_json: bool = typer.Option(False, "--json", help="Print the presets as a JSON array."),
) -> None:
    """List curated preset embedding and reranker models."""
    if as_json:
        typer.echo(json.dumps([_preset_record(name, spec) for name, spec in PRESETS.items()], indent=2))
        return
    table = Table(title="onnx2oracle presets")
    table.add_column("Name", style="bold cyan")
    table.add_column("Task")
    table.add_column("Dims", justify="right")
    table.add_column("Pooling")
    table.add_column("~Size (MB)", justify="right")
    table.add_column("Oracle model name", style="dim")
    for name, spec in PRESETS.items():
        table.add_row(
            name,
            spec.task,
            str(spec.dims) if spec.task == "embedding" else "—",
            spec.pooling if spec.task == "embedding" else "—",
            str(spec.approx_size_mb),
            spec.oracle_name,
        )
    console.print(table)


@app.command()
def load(
    preset: str | None = typer.Argument(None, help="Preset name (see `presets`)."),
    from_huggingface: str | None = typer.Option(
        None, "--from-huggingface", help="Raw HF repo, e.g. BAAI/bge-base-en-v1.5."
    ),
    task: str = typer.Option(
        "embedding", "--task", help="Pipeline shape: 'embedding' or 'reranker'."
    ),
    pooling: str | None = typer.Option(
        None, help="Pooling for embedding --from-huggingface: mean (default) or cls."
    ),
    normalize: bool | None = typer.Option(
        None, "--normalize/--no-normalize", help="L2-normalize the output (default: on). --from-huggingface only."
    ),
    max_length: int | None = typer.Option(
        None, help="Max tokenizer sequence length (default: 512). --from-huggingface only."
    ),
    dims: int | None = typer.Option(
        None, help="Expected output dims (required for embedding --from-huggingface)."
    ),
    name: str | None = typer.Option(None, help="Oracle model name override."),
    target: str | None = typer.Option(
        None, help="'local' for Docker shortcut. When given, it wins over ORACLE_DSN and config.toml."
    ),
    dsn: str | None = typer.Option(None, help="Full DSN: user/pw@host:port/service."),
    force: bool = typer.Option(False, help="Replace if already registered."),
    cache_dir: Path | None = typer.Option(None, help="HuggingFace cache dir."),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Rebuild the ONNX graph from scratch; neither read nor write the build cache (~/.onnx2oracle/cache).",
    ),
) -> None:
    """Build the augmented ONNX pipeline (embedding or reranker) and register it in Oracle."""
    if not preset and not from_huggingface:
        _fail("Provide a preset name or --from-huggingface", 2)

    parsed_task = _parse_task(task)
    name_derived = False

    if preset:
        conflicting = [
            flag
            for flag, value in (
                ("--from-huggingface", from_huggingface),
                ("--pooling", pooling),
                ("--dims", dims),
                ("--max-length", max_length),
                ("--normalize" if normalize else "--no-normalize", normalize),
            )
            if value is not None
        ]
        if conflicting:
            _fail(
                f"{', '.join(conflicting)} cannot be combined with preset {preset!r}: a preset fixes the "
                "model and its settings. Use --from-huggingface (without a preset) for a custom model.",
                2,
            )
        try:
            spec = get_preset(preset)
        except KeyError as exc:
            _fail(str(exc.args[0]), 2)
        if parsed_task != "embedding" and parsed_task != spec.task:
            console.print(
                f"[yellow]--task {escape(repr(parsed_task))} ignored: preset {escape(repr(preset))} is "
                f"registered as {escape(repr(spec.task))}.[/yellow]"
            )
    else:
        assert from_huggingface is not None
        if parsed_task == "embedding" and dims is None:
            _fail("--dims is required with --from-huggingface for embedding models", 2)
        if not name:
            name = from_huggingface.replace("/", "_").replace("-", "_").replace(".", "_").upper()
            name_derived = True
        if parsed_task == "embedding":
            assert dims is not None
            spec = EmbeddingSpec(
                hf_repo=from_huggingface,
                dims=dims,
                pooling=_parse_pooling(pooling or "mean"),
                normalize=True if normalize is None else normalize,
                oracle_name=name,
            )
        else:
            spec = RerankerSpec(hf_repo=from_huggingface, oracle_name=name)
        if max_length is not None:
            spec = replace(spec, max_length=max_length)

    if name and spec.oracle_name != name:
        spec = replace(spec, oracle_name=name)

    # Cheap local check first: a bad name must not cost a model download.
    try:
        validate_oracle_name(spec.oracle_name)
    except ValueError as exc:
        hint = " It was derived from --from-huggingface; pass --name to choose one." if name_derived else ""
        _fail(f"{exc}{hint}", 2)

    dsn_resolved = _resolve_target_dsn(dsn, target)
    console.print(f"[green]Target:[/green] {escape(dsn_resolved.display())}")
    console.print(f"[green]Model:[/green] {escape(spec.hf_repo)} ({spec.task}) -> {escape(spec.oracle_name)}")

    # Look at the database before the (slow) build: skip, refuse or fail fast now, not after minutes of work.
    try:
        with console.status("Checking Oracle for an existing model..."):
            existing = lookup_registered_task(dsn_resolved, spec.oracle_name)
    except Exception as exc:
        _fail(f"Could not check whether {spec.oracle_name} is registered on {dsn_resolved.display()}: {exc}")
    try:
        already_registered = check_existing(existing, spec.oracle_name, spec.task, force) is UploadStatus.SKIPPED
    except TaskMismatchError as exc:
        _fail(str(exc))
    if already_registered:
        console.print(
            Text(
                f"{spec.oracle_name} already registered (task {spec.task}), skipping; use --force to replace it.",
                style="yellow",
            ),
            soft_wrap=True,
        )
        raise typer.Exit(0)
    if existing is not None:
        console.print(
            Text(
                f"--force: {spec.oracle_name} is registered (task {existing}). It is dropped only after the "
                "new model has been built, and is not restored if the upload then fails.",
                style="yellow",
            ),
            soft_wrap=True,
        )

    try:
        if spec.task == "reranker":
            with console.status("Building reranker ONNX pipeline..."):
                data = build_reranker(spec, cache_dir=cache_dir, use_cache=not no_cache)
        else:
            with console.status("Building augmented ONNX pipeline..."):
                data = build_augmented(spec, cache_dir=cache_dir, use_cache=not no_cache)
    except (ValueError, RuntimeError, OSError) as exc:
        # The builders raise these with actionable messages (unsupported tokenizer, dims mismatch, failed local
        # validation, missing torch extra) and HF/network failures surface as OSError. Nothing was changed in Oracle.
        _fail(f"Could not build {spec.oracle_name} from {spec.hf_repo}: {exc}")
    console.print(f"[green]ONNX built:[/green] {len(data):,} bytes")

    try:
        with console.status("Uploading to Oracle..."):
            status = upload_model(dsn_resolved, data, spec.oracle_name, force=force, task=spec.task)
    except Exception as exc:
        _fail(str(exc))
    if status is UploadStatus.SKIPPED:
        console.print(
            Text(
                f"{spec.oracle_name} was registered (task {spec.task}) while the model was building, "
                "so nothing was uploaded; use --force to replace it.",
                style="yellow",
            ),
            soft_wrap=True,
        )
    else:
        console.print(f"[bold green]✓ {escape(spec.oracle_name)} registered ({spec.task}).[/bold green]")


@app.command()
def preflight(
    target: str | None = typer.Option(
        None, help="'local' for Docker shortcut. When given, it wins over ORACLE_DSN and config.toml."
    ),
    dsn: str | None = typer.Option(None, help="Full DSN."),
) -> None:
    """Check whether the target database is ready for ONNX model loading."""
    dsn_resolved = _resolve_target_dsn(dsn, target)
    console.print(f"[green]Target:[/green] {escape(dsn_resolved.display())}")

    result = run_preflight(dsn_resolved)

    for check in result.checks:
        console.print(Text.assemble(_mark(check.ok), f" {check.name}: {_inert(check.detail)}"))
    console.print(f"[dim]Elapsed: {result.elapsed_ms} ms[/dim]")
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def verify(
    name: str | None = typer.Option(None, help=f"Oracle model name (defaults to {DEFAULT_MODEL_NAME})."),
    expect_dims: int | None = typer.Option(
        None,
        "--expect-dims",
        min=1,
        help="Expected embedding dimension. Defaults to the preset's dims when --name is a preset's model name.",
    ),
    normalize: bool | None = typer.Option(
        None,
        "--normalize/--no-normalize",
        help="Whether the embedding is L2-normalized (norm ~ 1). Defaults to the preset's setting, else on.",
    ),
    target: str | None = typer.Option(
        None, help="'local' for Docker shortcut. When given, it wins over ORACLE_DSN and config.toml."
    ),
    dsn: str | None = typer.Option(None, help="Full DSN."),
) -> None:
    """Run an end-to-end smoke test against a registered model."""
    dsn_resolved = _resolve_target_dsn(dsn, target)
    model_name = name or DEFAULT_MODEL_NAME
    console.print(f"[green]Target:[/green] {escape(dsn_resolved.display())}")

    expected_dims = expect_dims
    expect_normalized = normalize
    preset_spec = _preset_for_oracle_name(model_name)
    if isinstance(preset_spec, EmbeddingSpec):
        if expected_dims is None:
            expected_dims = preset_spec.dims
        if expect_normalized is None:
            expect_normalized = preset_spec.normalize
    if expect_normalized is None:
        expect_normalized = True

    result = smoke_test(
        dsn_resolved, model_name, expected_dims=expected_dims, expect_normalized=expect_normalized
    )

    console.print(Text.assemble(_mark(result.connected), " Connected"))
    console.print(Text.assemble(_mark(result.model_registered), f" Model {model_name} registered"))
    if result.task:
        console.print(Text(f"Task: {result.task}", style="dim"))
    if result.sample_embedding_dims is not None:
        dims_text = f"{result.sample_embedding_dims} dims"
        if result.expected_dims is not None and result.sample_embedding_dims != result.expected_dims:
            dims_text += f", expected {result.expected_dims}"
        norm = result.sample_embedding_norm if result.sample_embedding_norm is not None else math.nan
        norm_text = f"norm={norm:.4f}"
        if result.expect_normalized and not result.norm_ok:
            norm_text += "; expected 1.0000 ± 0.001"
        console.print(
            Text.assemble(_mark(result.dims_ok and result.norm_ok), f" Sample embedding: {dims_text} ({norm_text})")
        )
    if result.sample_scores is not None:
        r_score, i_score = result.sample_scores
        console.print(
            Text.assemble(
                _mark(result.scores_ok), f" Sample scores: relevant={r_score:.4f} irrelevant={i_score:.4f}"
            )
        )
    if result.task == "reranker":
        sanity = "Relevance sanity (relevant > irrelevant)"
    else:
        sanity = "Similarity sanity (king/queen > king/banana)"
        if result.similarities is not None:
            king_queen, king_banana = result.similarities
            sanity += f": king/queen={king_queen:.4f} king/banana={king_banana:.4f}"
    console.print(Text.assemble(_mark(result.similarity_sane), f" {sanity}"))
    console.print(f"[dim]Elapsed: {result.elapsed_ms} ms[/dim]")
    if result.error:
        _print_error(result.error)
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def embed(
    texts: list[str] = typer.Argument(..., help="Text(s) to embed. All of them go to Oracle in one query."),
    name: str = typer.Option(
        DEFAULT_MODEL_NAME, "--name", help="Registered Oracle embedding model name."
    ),
    target: str | None = typer.Option(
        None, help="'local' for Docker shortcut. When given, it wins over ORACLE_DSN and config.toml."
    ),
    dsn: str | None = typer.Option(None, help="Full DSN."),
    as_json: bool = typer.Option(
        False, "--json", help="Print a JSON array of {text, dims, norm, embedding} instead of a table."
    ),
) -> None:
    """Embed text(s) with a registered embedding model, in a single round trip."""
    try:
        validate_oracle_name(name)
    except ValueError as exc:
        _fail(str(exc), 2, to_stderr=as_json)

    dsn_resolved = _resolve_target_dsn(dsn, target, announce=not as_json, to_stderr=as_json)
    if not as_json:
        console.print(f"[green]Target:[/green] {escape(dsn_resolved.display())}")

    try:
        vectors = _embed_via_db(dsn_resolved, name, texts)
    except Exception as exc:
        _fail(str(exc), to_stderr=as_json)

    norms = [math.sqrt(sum(x * x for x in vec)) for vec in vectors]
    if as_json:
        records = [
            {"text": text, "dims": len(vec) or None, "norm": norm if vec else None, "embedding": vec or None}
            for text, vec, norm in zip(texts, vectors, norms, strict=True)
        ]
        typer.echo(json.dumps(records))
    else:
        table = Table(title=Text(f"Embeddings from {name}"))
        table.add_column("#", justify="right")
        table.add_column("Dims", justify="right")
        table.add_column("Norm", justify="right")
        table.add_column("Text")
        for i, (text, vec, norm) in enumerate(zip(texts, vectors, norms, strict=True)):
            table.add_row(str(i), str(len(vec)) if vec else "NULL", f"{norm:.4f}" if vec else "NULL", Text(text))
        console.print(table)

    null_inputs = [text for text, vec in zip(texts, vectors, strict=True) if not vec]
    if null_inputs:
        _fail(f"Oracle returned NULL for {null_inputs!r} (empty text?).", to_stderr=as_json)


def _embed_via_db(dsn_resolved: DSN, model_name: str, texts: list[str]) -> list[list[float]]:
    validate_oracle_name(model_name)
    conn = dsn_resolved.connect()
    try:
        task = registered_task(conn, model_name)
        if task is None:
            raise ValueError(f"Model {model_name!r} is not registered in Oracle.")
        if task != "embedding":
            raise ValueError(
                f"Model {model_name!r} is registered as {task!r}, not an embedding model. "
                "Use `onnx2oracle rerank` for rerankers."
            )
        return embed_texts(conn, model_name, texts)
    finally:
        conn.close()


@app.command()
def rerank(
    query: str = typer.Option(..., "--query", "-q", help="The query text."),
    doc: list[str] = typer.Option(..., "--doc", "-d", help="Document to score (repeat for many)."),
    name: str = typer.Option(..., "--name", help="Registered Oracle reranker model name."),
    target: str | None = typer.Option(
        None, help="'local' for Docker shortcut. When given, it wins over ORACLE_DSN and config.toml."
    ),
    dsn: str | None = typer.Option(None, help="Full DSN."),
) -> None:
    """Score (query, document) pairs against a registered reranker model."""
    if not doc:
        _fail("At least one --doc is required", 2)
    try:
        validate_oracle_name(name)
    except ValueError as exc:
        _fail(str(exc), 2)

    dsn_resolved = _resolve_target_dsn(dsn, target)

    try:
        scores = _score_pairs(dsn_resolved, name, query, doc)
    except Exception as exc:
        _fail(str(exc))

    table = Table(title=Text(f"Reranker scores for: {query!r}"))
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Document")
    for rank, (score, text) in enumerate(scores, start=1):
        table.add_row(str(rank), f"{score:.4f}", Text(text))
    console.print(table)


def _score_pairs(
    dsn_resolved: DSN, model_name: str, query: str, docs: list[str]
) -> list[tuple[float, str]]:
    safe = validate_oracle_name(model_name)
    conn = dsn_resolved.connect()
    try:
        task = registered_task(conn, model_name)
        if task is None:
            raise ValueError(f"Model {model_name!r} is not registered in Oracle.")
        if task != "reranker":
            raise ValueError(
                f"Model {model_name!r} is registered as {task!r}, not a reranker. "
                f"Use `onnx2oracle verify` for embedding models, or load the model "
                f"with --task reranker."
            )
        cur = conn.cursor()
        results: list[tuple[float, str]] = []
        # A single prepared statement reused per document keeps the parse tree cached.
        sql = f"SELECT PREDICTION({safe} USING :q AS DATA1, :d AS DATA2) FROM dual"
        for d in docs:
            cur.execute(sql, {"q": query, "d": d})
            row = cur.fetchone()
            score = float(row[0]) if row and row[0] is not None else float("-inf")
            results.append((score, d))
    finally:
        conn.close()
    results.sort(key=lambda r: r[0], reverse=True)
    return results


# Docker subcommands

DOCKER_COMPOSE = files("onnx2oracle") / "data" / "docker-compose.yml"
DEFAULT_DOCKER_WAIT_TIMEOUT_SECONDS = 600
DEFAULT_DOCKER_WAIT_INTERVAL_SECONDS = 5


def _compose(*args: str) -> int:
    return subprocess.call(["docker", "compose", "-f", str(DOCKER_COMPOSE), *args])


def _wait_for_oracle(timeout_seconds: int, interval_seconds: int) -> int:
    probe = f"""
set -eu
dsn="system/${{ORACLE_PWD:-onnx2oracle}}@localhost:1521/FREEPDB1"
deadline=$((SECONDS + {timeout_seconds}))
while [ "$SECONDS" -lt "$deadline" ]; do
  if echo 'select 1 from dual;' | sqlplus -L -S "$dsn" >/dev/null 2>&1; then
    echo ready
    exit 0
  fi
  sleep {interval_seconds}
done
echo "timed out after {timeout_seconds}s waiting for Oracle at localhost:1521/FREEPDB1" >&2
exit 1
""".strip()
    return _compose("exec", "-T", "oracle", "bash", "-lc", probe)


@docker_app.command("up")
def docker_up(
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for a SQL readiness probe to pass."),
    wait_timeout: int = typer.Option(
        DEFAULT_DOCKER_WAIT_TIMEOUT_SECONDS,
        "--wait-timeout",
        min=1,
        help="Seconds to wait before failing the SQL readiness probe.",
    ),
    wait_interval: int = typer.Option(
        DEFAULT_DOCKER_WAIT_INTERVAL_SECONDS,
        "--wait-interval",
        min=1,
        help="Seconds between SQL readiness probes.",
    ),
) -> None:
    """Start the Oracle AI Database Free container."""
    rc = _compose("up", "-d")
    if rc != 0:
        raise typer.Exit(rc)
    if wait:
        console.print(
            "[yellow]Waiting for Oracle SQL readiness "
            f"(timeout={wait_timeout}s, interval={wait_interval}s)...[/yellow]"
        )
        rc = _wait_for_oracle(timeout_seconds=wait_timeout, interval_seconds=wait_interval)
        if rc != 0:
            raise typer.Exit(rc)
    console.print("[bold green]✓ Oracle up.[/bold green]")


@docker_app.command("down")
def docker_down(volumes: bool = typer.Option(False, "--volumes", "-v", help="Remove the database volume too.")) -> None:
    """Stop the Oracle container."""
    args = ["down"]
    if volumes:
        args.append("--volumes")
    rc = _compose(*args)
    if rc != 0:
        raise typer.Exit(rc)


@docker_app.command("logs")
def docker_logs(follow: bool = typer.Option(False, "-f", "--follow")) -> None:
    """Tail Oracle container logs."""
    args = ["logs"]
    if follow:
        args.append("-f")
    rc = _compose(*args)
    if rc != 0:
        raise typer.Exit(rc)


# Config subcommands

_CONFIG_KEY = re.compile(r"[A-Za-z0-9_-]+")
_KEY_LINE = re.compile(r"\s*(?P<key>[A-Za-z0-9_-]+|\"[^\"]*\"|'[^']*')\s*=")
_TABLE_HEADER = re.compile(r"\s*(?P<open>\[\[?)\s*(?P<name>[^\[\]#]*?)\s*\]\]?\s*(?:#.*)?")
_SECRET_KEY = re.compile(r"pass|secret|token|credential|api[_-]?key", re.IGNORECASE)
_TOML_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}


def _toml_string(value: str) -> str:
    """Encode *value* as a TOML basic string: quotes, backslashes and control characters escaped."""
    out = []
    for ch in value:
        if ch in _TOML_ESCAPES:
            out.append(_TOML_ESCAPES[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _table_name(line: str) -> str | None:
    """The table a ``[table]`` header line opens, ``"[name]"`` for ``[[array-of-tables]]``, else ``None``."""
    m = _TABLE_HEADER.fullmatch(line)
    if m is None:
        return None
    name = m["name"].strip("\"' ")
    return name if m["open"] == "[" else f"[{name}]"


def _line_key(line: str) -> str | None:
    m = _KEY_LINE.match(line)
    return m["key"].strip("\"'") if m else None


def _set_default_key(text: str, key: str, value: str) -> str:
    """Return *text* with ``key = "value"`` set in its ``[default]`` table.

    An existing line for *key* in that table is replaced whatever its spacing; otherwise the key is added at the
    end of the table. A file without a ``[default]`` table gets one appended. Everything else is left as is.
    """
    new_line = f"{key} = {_toml_string(value)}"
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()

    start = next((i for i, line in enumerate(lines) if _table_name(line) == "default"), None)
    if start is None:
        lines += ["", "[default]", new_line] if lines else ["[default]", new_line]
        return "\n".join(lines) + "\n"

    end = next((i for i in range(start + 1, len(lines)) if _table_name(lines[i]) is not None), len(lines))
    matches = [i for i in range(start + 1, end) if _line_key(lines[i]) == key]
    if matches:
        lines[matches[0]] = new_line
        for i in reversed(matches[1:]):
            del lines[i]
    else:
        insert_at = end
        while insert_at > start + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines.insert(insert_at, new_line)
    return "\n".join(lines) + "\n"


def _toml_key(key: str) -> str:
    return key if _CONFIG_KEY.fullmatch(key) else _toml_string(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()) + " }"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return repr(value)  # int or float; repr gives TOML's spelling of nan/inf too


def _render_config(data: dict[str, Any]) -> str:
    """Render parsed config *data* as TOML: a table's own values first, then its sub-tables as ``[headers]``."""
    lines: list[str] = []

    def emit(table: dict[str, Any], path: tuple[str, ...]) -> None:
        if path:
            if lines:
                lines.append("")
            lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        for key, value in table.items():
            if isinstance(value, dict):
                emit(value, (*path, key))

    emit(data, ())
    return "\n".join(lines)


def _mask_dsn(raw: str) -> str:
    """``user/password@target`` -> ``user/***@target`` (just ``***`` if *raw* is not a parseable DSN)."""
    try:
        parsed = DSN.parse(raw)
    except ValueError:
        return "***"
    return f"{parsed.user}/***@{parsed.to_oracle_dsn()}"


def _mask(key: str, value: Any, secret: bool = False) -> Any:
    """*value* as ``config show`` displays it: DSN passwords, and everything under a password/token/secret-like
    key, replaced by ``***``. Works on parsed TOML, so every spelling of a key (dotted, inline table, multi-line
    string) is covered.
    """
    secret = secret or bool(_SECRET_KEY.search(key))
    if isinstance(value, dict):
        return {k: _mask(k, v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(key, v, secret) for v in value]
    if secret:
        return "***"
    if key == "dsn" and isinstance(value, str):
        return _mask_dsn(value)
    return value


def _write_private(path: Path, text: str) -> None:
    """Write *text* to *path* atomically with mode 0600 (the directory is created 0700 if missing)."""
    real = Path(os.path.realpath(path))  # keep a symlinked config pointing where it did
    real.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = tempfile.mkstemp(dir=real.parent, prefix=".config.", suffix=".tmp")  # created 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp_name, real)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


@config_app.command("show")
def config_show() -> None:
    """Print the config file's contents (passwords and tokens masked, comments omitted)."""
    path = DEFAULT_CONFIG_PATH
    if not path.exists():
        console.print(f"[dim]No config at {escape(str(path))}[/dim]")
        return
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        _fail(f"{path} is not valid TOML ({exc}). Fix it by hand, or rewrite a key with `onnx2oracle config set`.")
    console.print(Text(f"# {path} (passwords and tokens masked, comments omitted)", style="dim"), soft_wrap=True)
    console.print(_render_config(_mask("", data)), markup=False, highlight=False, emoji=False, soft_wrap=True)


@config_app.command("set")
def config_set(kv: str = typer.Argument(..., help="key=value (stored in the [default] table)")) -> None:
    """Set a key in ~/.onnx2oracle/config.toml (file mode 0600)."""
    key, sep, value = kv.partition("=")
    if not sep:
        _fail("Expected key=value", 2)
    key = key.strip()
    if not _CONFIG_KEY.fullmatch(key):
        _fail(f"Invalid config key {key!r}: use letters, digits, '_' and '-' only.", 2)

    path = DEFAULT_CONFIG_PATH
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    updated = _set_default_key(existing, key, value)
    # Never leave behind a file that does not parse or does not hold exactly what was asked for.
    try:
        stored = tomllib.loads(updated).get("default", {}).get(key)
    except tomllib.TOMLDecodeError as exc:
        _fail(f"Refusing to write {path}: the result would not be valid TOML ({exc}). Fix or remove that file.")
    if stored != value:
        _fail(f"Refusing to write {path}: could not store {key!r} in its [default] table. Fix or remove that file.")

    try:
        _write_private(path, updated)
    except (OSError, UnicodeError) as exc:
        _fail(f"Could not write {path}: {exc}")
    console.print(f"[green]Wrote {escape(key)} to {escape(str(path))}[/green]")


if __name__ == "__main__":
    app()
