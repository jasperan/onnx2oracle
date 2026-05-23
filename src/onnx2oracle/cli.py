"""Typer-based CLI for onnx2oracle."""

from __future__ import annotations

import logging
import os
import subprocess
from importlib.resources import files
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from onnx2oracle import __version__
from onnx2oracle._ident import validate_oracle_name
from onnx2oracle.connection import DEFAULT_CONFIG_PATH, DSN, resolve_dsn
from onnx2oracle.loader import registered_task, upload_model
from onnx2oracle.pipeline import build_augmented, build_reranker
from onnx2oracle.preflight import run_preflight
from onnx2oracle.presets import PRESETS, EmbeddingSpec, Pooling, RerankerSpec, Task, get_preset
from onnx2oracle.verify import smoke_test

app = typer.Typer(
    name="onnx2oracle",
    help="Load ONNX embedding and reranker models into Oracle AI Database.",
    no_args_is_help=True,
)
docker_app = typer.Typer(help="Manage a local Oracle AI Database Free container.")
config_app = typer.Typer(help="Manage ~/.onnx2oracle/config.toml.")
app.add_typer(docker_app, name="docker")
app.add_typer(config_app, name="config")

console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


def _parse_task(value: str) -> Task:
    if value not in ("embedding", "reranker"):
        console.print(f"[red]--task must be 'embedding' or 'reranker', got {value!r}[/red]")
        raise typer.Exit(2)
    return value


def _parse_pooling(value: str) -> Pooling:
    if value not in ("mean", "cls"):
        console.print(f"[red]--pooling must be 'mean' or 'cls', got {value!r}[/red]")
        raise typer.Exit(2)
    return value


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"onnx2oracle {__version__}")


@app.command()
def presets() -> None:
    """List curated preset embedding and reranker models."""
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
    pooling: str = typer.Option("mean", help="Pooling for embedding --from-huggingface: mean or cls."),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize"),
    max_length: int = typer.Option(512, help="Max tokenizer sequence length."),
    dims: int | None = typer.Option(
        None, help="Expected output dims (required for embedding --from-huggingface)."
    ),
    name: str | None = typer.Option(None, help="Oracle model name override."),
    target: str | None = typer.Option(None, help="'local' for Docker shortcut."),
    dsn: str | None = typer.Option(None, help="Full DSN: user/pw@host:port/service."),
    force: bool = typer.Option(False, help="Replace if already registered."),
    cache_dir: Path | None = typer.Option(None, help="HuggingFace cache dir."),
) -> None:
    """Build the augmented ONNX pipeline (embedding or reranker) and register it in Oracle."""
    if not preset and not from_huggingface:
        console.print("[red]Provide a preset name or --from-huggingface[/red]")
        raise typer.Exit(2)

    parsed_task = _parse_task(task)

    if (
        target is None
        and dsn is None
        and "ORACLE_DSN" not in os.environ
        and not DEFAULT_CONFIG_PATH.exists()
    ):
        console.print("[dim]No DSN configured — defaulting to --target local (docker-compose).[/dim]")
        target = "local"

    if preset:
        spec = get_preset(preset)
        if parsed_task != "embedding" and parsed_task != spec.task:
            console.print(
                f"[yellow]--task {parsed_task!r} ignored: preset {preset!r} is registered as "
                f"{spec.task!r}.[/yellow]"
            )
    else:
        assert from_huggingface is not None
        if parsed_task == "embedding" and dims is None:
            console.print("[red]--dims is required with --from-huggingface for embedding models[/red]")
            raise typer.Exit(2)
        if not name:
            name = from_huggingface.replace("/", "_").replace("-", "_").replace(".", "_").upper()
        if parsed_task == "embedding":
            parsed_pooling = _parse_pooling(pooling)
            assert dims is not None
            spec = EmbeddingSpec(
                hf_repo=from_huggingface,
                dims=dims,
                pooling=parsed_pooling,
                normalize=normalize,
                oracle_name=name,
                max_length=max_length,
            )
        else:
            spec = RerankerSpec(
                hf_repo=from_huggingface,
                oracle_name=name,
                max_length=max_length,
            )

    effective_name = name or spec.oracle_name
    if spec.oracle_name != effective_name:
        from dataclasses import replace as _replace
        spec = _replace(spec, oracle_name=effective_name)

    dsn_resolved = resolve_dsn(cli_dsn=dsn, target=target)
    console.print(f"[green]Target:[/green] {dsn_resolved.display()}")
    console.print(f"[green]Model:[/green] {spec.hf_repo} ({spec.task}) -> {spec.oracle_name}")

    if spec.task == "reranker":
        with console.status("Building reranker ONNX pipeline..."):
            data = build_reranker(spec, cache_dir=cache_dir)
    else:
        with console.status("Building augmented ONNX pipeline..."):
            data = build_augmented(spec, cache_dir=cache_dir)
    console.print(f"[green]ONNX built:[/green] {len(data):,} bytes")

    with console.status("Uploading to Oracle..."):
        upload_model(dsn_resolved, data, spec.oracle_name, force=force, task=spec.task)
    console.print(f"[bold green]✓ {spec.oracle_name} registered ({spec.task}).[/bold green]")


@app.command()
def preflight(
    target: str | None = typer.Option(None, help="'local' for Docker shortcut."),
    dsn: str | None = typer.Option(None, help="Full DSN."),
) -> None:
    """Check whether the target database is ready for ONNX model loading."""
    if (
        target is None
        and dsn is None
        and "ORACLE_DSN" not in os.environ
        and not DEFAULT_CONFIG_PATH.exists()
    ):
        console.print("[dim]No DSN configured — defaulting to --target local (docker-compose).[/dim]")
        target = "local"

    dsn_resolved = resolve_dsn(cli_dsn=dsn, target=target)
    console.print(f"[green]Target:[/green] {dsn_resolved.display()}")

    result = run_preflight(dsn_resolved)

    def _mark(ok: bool) -> str:
        return "[green]✓[/green]" if ok else "[red]✗[/red]"

    for check in result.checks:
        console.print(f"{_mark(check.ok)} {check.name}: {check.detail}")
    console.print(f"[dim]Elapsed: {result.elapsed_ms} ms[/dim]")
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def verify(
    name: str | None = typer.Option(None, help="Oracle model name (defaults to ALL_MINILM_L6_V2)."),
    target: str | None = typer.Option(None, help="'local' for Docker shortcut."),
    dsn: str | None = typer.Option(None, help="Full DSN."),
) -> None:
    """Run an end-to-end smoke test against a registered model."""
    if (
        target is None
        and dsn is None
        and "ORACLE_DSN" not in os.environ
        and not DEFAULT_CONFIG_PATH.exists()
    ):
        console.print("[dim]No DSN configured — defaulting to --target local (docker-compose).[/dim]")
        target = "local"

    dsn_resolved = resolve_dsn(cli_dsn=dsn, target=target)
    model_name = name or "ALL_MINILM_L6_V2"
    console.print(f"[green]Target:[/green] {dsn_resolved.display()}")

    result = smoke_test(dsn_resolved, model_name)

    def _mark(ok: bool) -> str:
        return "[green]✓[/green]" if ok else "[red]✗[/red]"

    console.print(f"{_mark(result.connected)} Connected")
    console.print(f"{_mark(result.model_registered)} Model {model_name} registered")
    if result.task:
        console.print(f"[dim]Task: {result.task}[/dim]")
    if result.sample_embedding_dims is not None:
        console.print(
            f"{_mark(True)} Sample embedding: {result.sample_embedding_dims} dims "
            f"(norm={result.sample_embedding_norm:.4f})"
        )
    if result.sample_scores is not None:
        r_score, i_score = result.sample_scores
        console.print(
            f"{_mark(True)} Sample scores: relevant={r_score:.4f} irrelevant={i_score:.4f}"
        )
    if result.task == "reranker":
        console.print(f"{_mark(result.similarity_sane)} Relevance sanity (relevant > irrelevant)")
    else:
        console.print(f"{_mark(result.similarity_sane)} Similarity sanity (king/queen > king/banana)")
    console.print(f"[dim]Elapsed: {result.elapsed_ms} ms[/dim]")
    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
    embedding_ok = result.task != "embedding" or result.sample_embedding_dims is not None
    reranker_ok = result.task != "reranker" or result.sample_scores is not None
    if (
        result.error
        or not result.connected
        or not result.model_registered
        or not embedding_ok
        or not reranker_ok
        or not result.similarity_sane
    ):
        raise typer.Exit(1)


@app.command()
def rerank(
    query: str = typer.Option(..., "--query", "-q", help="The query text."),
    doc: list[str] = typer.Option(..., "--doc", "-d", help="Document to score (repeat for many)."),
    name: str = typer.Option(..., "--name", help="Registered Oracle reranker model name."),
    target: str | None = typer.Option(None, help="'local' for Docker shortcut."),
    dsn: str | None = typer.Option(None, help="Full DSN."),
) -> None:
    """Score (query, document) pairs against a registered reranker model."""
    if not doc:
        console.print("[red]At least one --doc is required[/red]")
        raise typer.Exit(2)
    try:
        validate_oracle_name(name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    if (
        target is None
        and dsn is None
        and "ORACLE_DSN" not in os.environ
        and not DEFAULT_CONFIG_PATH.exists()
    ):
        target = "local"
    dsn_resolved = resolve_dsn(cli_dsn=dsn, target=target)

    scores = _score_pairs(dsn_resolved, name, query, doc)

    table = Table(title=f"Reranker scores for: {query!r}")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Document")
    for rank, (score, text) in enumerate(scores, start=1):
        table.add_row(str(rank), f"{score:.4f}", text)
    console.print(table)


def _score_pairs(
    dsn_resolved: DSN, model_name: str, query: str, docs: list[str]
) -> list[tuple[float, str]]:
    import oracledb

    safe = validate_oracle_name(model_name)
    conn = oracledb.connect(
        user=dsn_resolved.user,
        password=dsn_resolved.password,
        dsn=dsn_resolved.to_oracle_dsn(),
        tcp_connect_timeout=30,
    )
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
        # Batch all (query, doc) pairs in one round-trip via executemany.
        params = [{"q": query, "d": d} for d in docs]
        results: list[tuple[float, str]] = []
        # oracledb's executemany doesn't return row sets for SELECTs, so loop with
        # a single prepared statement (binds keep the parse tree cached).
        sql = f"SELECT PREDICTION({safe} USING :q AS DATA1, :d AS DATA2) FROM dual"
        for p, d in zip(params, docs, strict=True):
            cur.execute(sql, p)
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

@config_app.command("show")
def config_show() -> None:
    if not DEFAULT_CONFIG_PATH.exists():
        console.print(f"[dim]No config at {DEFAULT_CONFIG_PATH}[/dim]")
        return
    console.print(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


@config_app.command("set")
def config_set(kv: str = typer.Argument(..., help="key=value")) -> None:
    if "=" not in kv:
        console.print("[red]Expected key=value[/red]")
        raise typer.Exit(2)
    key, value = kv.split("=", 1)
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if DEFAULT_CONFIG_PATH.exists():
        existing = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    if "[default]" not in existing:
        existing = "[default]\n" + existing
    lines = [line for line in existing.splitlines() if not line.startswith(f"{key} ")]
    lines.append(f'{key} = "{value}"')
    DEFAULT_CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote {key} to {DEFAULT_CONFIG_PATH}[/green]")


if __name__ == "__main__":
    app()
