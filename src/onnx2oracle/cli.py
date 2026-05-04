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
from onnx2oracle.connection import DEFAULT_CONFIG_PATH, resolve_dsn
from onnx2oracle.loader import upload_model
from onnx2oracle.pipeline import build_augmented
from onnx2oracle.preflight import run_preflight
from onnx2oracle.presets import PRESETS, ModelSpec, get_preset
from onnx2oracle.verify import smoke_test

app = typer.Typer(
    name="onnx2oracle",
    help="Load ONNX embedding models into Oracle AI Database.",
    no_args_is_help=True,
)
docker_app = typer.Typer(help="Manage a local Oracle 26ai Free container.")
config_app = typer.Typer(help="Manage ~/.onnx2oracle/config.toml.")
app.add_typer(docker_app, name="docker")
app.add_typer(config_app, name="config")

console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"onnx2oracle {__version__}")


@app.command()
def presets() -> None:
    """List curated preset embedding models."""
    table = Table(title="onnx2oracle presets")
    table.add_column("Name", style="bold cyan")
    table.add_column("Dims", justify="right")
    table.add_column("Pooling")
    table.add_column("~Size (MB)", justify="right")
    table.add_column("Oracle model name", style="dim")
    for name, spec in PRESETS.items():
        table.add_row(
            name, str(spec.dims), spec.pooling, str(spec.approx_size_mb), spec.oracle_name,
        )
    console.print(table)


@app.command()
def load(
    preset: str | None = typer.Argument(None, help="Preset name (see `presets`)."),
    from_huggingface: str | None = typer.Option(
        None, "--from-huggingface", help="Raw HF repo, e.g. BAAI/bge-base-en-v1.5."
    ),
    pooling: str = typer.Option("mean", help="Pooling for --from-huggingface: mean or cls."),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize"),
    max_length: int = typer.Option(512, help="Max tokenizer sequence length."),
    dims: int | None = typer.Option(None, help="Expected output dims (required for --from-huggingface)."),
    name: str | None = typer.Option(None, help="Oracle model name override."),
    target: str | None = typer.Option(None, help="'local' for Docker shortcut."),
    dsn: str | None = typer.Option(None, help="Full DSN: user/pw@host:port/service."),
    force: bool = typer.Option(False, help="Replace if already registered."),
    cache_dir: Path | None = typer.Option(None, help="HuggingFace cache dir."),
) -> None:
    """Build the augmented ONNX pipeline and register it in Oracle."""
    if not preset and not from_huggingface:
        console.print("[red]Provide a preset name or --from-huggingface[/red]")
        raise typer.Exit(2)

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
    else:
        if dims is None:
            console.print("[red]--dims is required with --from-huggingface[/red]")
            raise typer.Exit(2)
        if not name:
            name = from_huggingface.replace("/", "_").replace("-", "_").replace(".", "_").upper()
        if pooling not in ("mean", "cls"):
            console.print(f"[red]--pooling must be 'mean' or 'cls', got {pooling!r}[/red]")
            raise typer.Exit(2)
        spec = ModelSpec(
            hf_repo=from_huggingface,
            dims=dims,
            pooling=pooling,  # type: ignore[arg-type]
            normalize=normalize,
            oracle_name=name,
            max_length=max_length,
        )

    effective_name = name or spec.oracle_name
    if spec.oracle_name != effective_name:
        spec = ModelSpec(
            hf_repo=spec.hf_repo,
            dims=spec.dims,
            pooling=spec.pooling,
            normalize=spec.normalize,
            oracle_name=effective_name,
            max_length=spec.max_length,
            approx_size_mb=spec.approx_size_mb,
        )

    dsn_resolved = resolve_dsn(cli_dsn=dsn, target=target)
    console.print(f"[green]Target:[/green] {dsn_resolved.display()}")
    console.print(f"[green]Model:[/green] {spec.hf_repo} -> {spec.oracle_name}")

    with console.status("Building augmented ONNX pipeline..."):
        data = build_augmented(spec, cache_dir=cache_dir)
    console.print(f"[green]Augmented ONNX built:[/green] {len(data):,} bytes")

    with console.status("Uploading to Oracle..."):
        upload_model(dsn_resolved, data, spec.oracle_name, force=force)
    console.print(f"[bold green]✓ {spec.oracle_name} registered.[/bold green]")


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
    if result.sample_embedding_dims is not None:
        console.print(
            f"{_mark(True)} Sample embedding: {result.sample_embedding_dims} dims "
            f"(norm={result.sample_embedding_norm:.4f})"
        )
    console.print(f"{_mark(result.similarity_sane)} Similarity sanity (king/queen > king/banana)")
    console.print(f"[dim]Elapsed: {result.elapsed_ms} ms[/dim]")
    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        raise typer.Exit(1)


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
    """Start the Oracle 26ai Free container."""
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
