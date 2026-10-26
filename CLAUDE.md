# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`onnx2oracle` ships a sentence-transformer HuggingFace repo into an Oracle AI Database instance as a single self-contained ONNX graph that Oracle's `DBMS_VECTOR.LOAD_ONNX_MODEL` can register. After registration, `VECTOR_EMBEDDING(MODEL USING :text AS DATA)` runs the full tokenizer + transformer + pooling + L2-normalize chain entirely in-database.

Published to PyPI as `onnx2oracle`. Python 3.10+. CLI entry point: `onnx2oracle` (Typer).

## Commands

```bash
# Env (uv; uv.lock and .python-version are committed)
uv sync --extra dev

# Lint / format-check (matches CI)
uv run ruff check src/ tests/ scripts/

# Type check (CI runs this on one Python version)
uv run pyright src/

# Fast tests (no network, no DB)
uv run pytest tests/ -v -m "not slow and not integration"

# Slow tests (real HuggingFace downloads, still no DB; reranker tests need torch: `uv sync --extra dev --extra export`)
uv run pytest tests/test_pipeline.py -v -m slow

# Integration test — needs a live Oracle AI Database Free container
ORACLE_PORT=1524 ORACLE_DSN='system/yourpw@localhost:1524/FREEPDB1' \
  uv run pytest tests/test_loader_integration.py --run-integration -v

# Single test
uv run pytest tests/test_pipeline.py::test_build_augmented_miniLM_L6_shape_and_valid_onnx -v

# Go TUI (gotui/) — what the CI `gotui` job runs
(cd gotui && gofmt -l . && go vet ./... && go test ./...)

# Local Oracle container (via the CLI, which uses the packaged compose file)
ORACLE_PORT=1524 uv run onnx2oracle docker up --wait --wait-timeout 900
ORACLE_PORT=1524 uv run onnx2oracle docker down

# End-to-end loop
uv run onnx2oracle preflight --target local
uv run onnx2oracle load all-MiniLM-L6-v2 --target local
uv run onnx2oracle verify --target local
uv run onnx2oracle embed "hello world" --target local

# Evidence loops
ORACLE_PORT=1524 scripts/run_real_db_integration.sh
ORACLE_PORT=1524 uv run scripts/check_model_compatibility.py all-MiniLM-L6-v2
uv run scripts/render_preset_tables.py --check

# Release (PyPI) — bump `__version__` in src/onnx2oracle/__init__.py only (pyproject reads it dynamically)
rm -rf dist/ build/ src/*.egg-info && uv run python -m build && uv run twine check dist/* && uv run twine upload dist/*
```

The `--run-integration` flag is a custom pytest option (see `tests/conftest.py`); without it, tests marked `integration` are skipped regardless of `-m`.

Known dependency debt: `transformers.onnx` (used by the PyTorch export fallback in `pipeline.py` and by the reranker export) was removed in transformers 5, which is why `pyproject.toml` pins `transformers<5`. Migrating the export to Optimum is the way to lift the pin.

## Architecture

Flow: HuggingFace repo → augmented ONNX bytes → Oracle `DBMS_VECTOR.LOAD_ONNX_MODEL` → SQL-callable `VECTOR_EMBEDDING`.

### The 6-stage ONNX augmentation (`src/onnx2oracle/pipeline.py`)

This is the non-obvious part. `build_augmented(spec)` returns raw bytes of a single ONNX graph with input `pre_text: string[1]` and output `embedding: float32[dims]`:

1. **Download core transformer** — prefers `onnx/model.onnx` from the HF repo, detects ONNX external-data sidecars, downloads those sidecars from the repo's `onnx/` folder, and falls back to PyTorch `AutoModel` + `transformers.onnx.export` at opset 14.
2. **Generate tokenizer ONNX** via `onnxruntime_extensions.gen_processing_models(tokenizer, opset=14)`. If the tokenizer class isn't directly supported (e.g. MPNet, XLM-R) or doesn't emit the BERT-style `input_ids`/`attention_mask`/`token_type_ids` outputs, it falls back to `BertTokenizerFast`. If the vocab lacks `[UNK]` (i.e. SentencePiece/Unigram), it raises `NotImplementedError` — Oracle's BertTokenizer op cannot represent SentencePiece models like XLM-R or multilingual-e5.
3. **Align opsets** — bumps core to opset 18 via `version_converter`, syncs `ir_version`, and copies any custom domains from the tokenizer subgraph.
4. **Truncate + unsqueeze tokenizer outputs** — tokenizer emits `[seq_len]`, core expects `[1, seq_len]`. Each tensor is truncated to `max_length` while keeping the closing `[SEP]` (`Slice` ×3 + `Concat`: `raw[:-1][:max_length-1] + raw[-1:]`, identical for inputs that fit), then `Unsqueeze`d. The reranker applies this to the query (`query_max_length`) and document sides separately.
5. **Merge** — `onnx.compose.merge_models(pre, core, io_map=[…], prefix1="pre_", prefix2="core_")`. The `pre_` prefix is why the Oracle metadata JSON names the input `pre_text` (see `loader.build_metadata_json`). Only connects tensors present in both graphs (MPNet has no `token_type_ids`).
6. **Pool + L2-normalize + squeeze** — pooling reads the core's hidden-state output, found dynamically (`last_hidden_state`, else the first float32 rank-3 output such as `token_embeddings`); its hidden size is checked against `dims` up front (`ValueError`). `ReduceMean(axis=1)` for mean pooling; `Gather(axis=1, [0]) + Squeeze` for CLS. L2 norm = `Pow(2) → ReduceSum(-1) → Sqrt → Max(eps=1e-12) → Div`. Final `Squeeze` drops the batch dim so the output is `float32[dims]`, not `float32[1, dims]`. Before the bytes are returned, the built graph is run once in onnxruntime + extensions on "hello world" (shape, finiteness, norm) and build fails if that check does.

If you touch any pooling/normalize node names, the Oracle-side metadata (`embeddingOutput: "embedding"`) has to match.

### Loader idempotency (`src/onnx2oracle/loader.py`)

`upload_model` returns an `UploadStatus` (`REGISTERED` or `SKIPPED`). Existing model, same task, no force → `SKIPPED`; different task, no force → `TaskMismatchError` (a `ValueError`); force → WARNING, drop, load, and if the load fails after the drop it raises `ModelUploadError` ("…no longer registered"). The `load` CLI calls `lookup_registered_task`/`check_existing` *before* building (skip exits 0, mismatch or unreachable DB exits 1, nothing is downloaded), and build errors print "Error: Could not build NAME from REPO: …" with exit 1. Drop-then-load is not atomic. The model bytes are passed as a bind variable to `LOAD_ONNX_MODEL`; metadata is `JSON(:metadata)` with `{"function": "embedding", "embeddingOutput": "embedding", "input": {"pre_text": ["DATA"]}}`.

### SQL-injection guard (`src/onnx2oracle/_ident.py`)

Oracle's `VECTOR_EMBEDDING(<model_name> USING :text AS DATA)` takes the model name as a **SQL token, not a bind variable**. Every code path that interpolates `oracle_name` into SQL goes through `validate_oracle_name()`, which enforces `/^[A-Z_][A-Z0-9_]{0,127}$/`. Anything else raises `ValueError` before the query runs. Do not add new SQL paths that bypass this.

### DSN resolution (`src/onnx2oracle/connection.py`)

Precedence: `--dsn` flag → an explicitly passed `--target local` → `ORACLE_DSN` env → `~/.onnx2oracle/config.toml` (`[default] dsn = "…"`) → interactive prompt (tty only). The `local` DSN is `system/${ORACLE_PWD:-onnx2oracle}@localhost:${ORACLE_PORT:-1521}/FREEPDB1`.

An explicit `--target local` beats ambient `ORACLE_DSN`/config.toml on purpose: `scripts/check_model_compatibility.py` defaults to it and force-drops models, so it must never land on a remote DB because of an env var. The CLI passes `target` to `resolve_dsn` only when the user typed `--target` (Typer default `None`); every Oracle-bound command still auto-defaults to the local DSN when nothing at all is set, so the zero-config docker-compose flow works. `DSN.__repr__`/`__str__` and `DSN.parse` errors never contain the password.

`DSN.parse` handles passwords containing `@` by scanning the `@host:port/service` suffix from the right.

### Verify (`src/onnx2oracle/verify.py`)

`smoke_test` never raises DB errors; they land in `VerifyResult.error`. `VerifyResult.ok` = connected + registered + no error + dims > 0 (== expected dims when known; `--expect-dims`, defaulting to the preset's dims when `--name` is a preset's Oracle name) + finite L2 norm within 1e-3 of 1.0 (waived by `verify --no-normalize`) + `cosine(king, queen) > cosine(king, banana)` (the cosines are printed). `verify.embed_texts()` is the one-round-trip multi-row `VECTOR_EMBEDDING` query shared by `embed` and `verify`. The CLI renders it with Rich checkmarks and exits 1 unless ok. `rerank` is still one `PREDICTION` query per document.

### Built-ONNX cache and new commands

The final augmented ONNX is cached in `~/.onnx2oracle/cache/` (`ONNX2ORACLE_CACHE_DIR` overrides; `load --no-cache` bypasses). The key covers the spec, the onnx2oracle version, pipeline source and dependency versions; entries are written only after a local onnxruntime self-check. `--cache-dir` is only the HuggingFace download cache. `onnx2oracle embed TEXT...` embeds many texts in one round trip; `onnx2oracle presets --json` is the stable interface the Go TUI consumes (do not scrape the Rich table).

### Docker compose

Two copies exist intentionally:
- `src/onnx2oracle/data/docker-compose.yml` — shipped inside the wheel; the CLI's `docker up/down/logs` subcommands point at this via `importlib.resources.files("onnx2oracle") / "data" / "docker-compose.yml"`.
- `docker/docker-compose.yml` — dev-only copy for git-clone workflow.

Both honor `ORACLE_PWD` (default `onnx2oracle`), `ORACLE_PORT` (default `1521`), and `ORACLE_IMAGE` (default `container-registry.oracle.com/database/free:latest`). The CLI's `docker up --wait` runs a bounded SQL readiness probe; tune it with `--wait-timeout` and `--wait-interval`.

## Testing notes

- `tests/conftest.py` adds the `--run-integration` flag. Integration tests are **skipped by default** even if you pass `-m integration` — you also need `--run-integration`.
- `test_pipeline.py::test_build_augmented_*` is marked `slow` because it actually downloads models from HuggingFace (~90 MB–540 MB depending on preset). The PR/push CI skips them; the `slow` CI job runs them weekly and on `workflow_dispatch`, with the HF cache.
- `test_loader_integration.py` expects a live Oracle and reads `ORACLE_DSN`, or `ORACLE_PWD`/`ORACLE_PORT` for the local shortcut defaults.
- Unit tests don't need network or DB; CI runs them on the 3.10 / 3.11 / 3.12 / 3.13 matrix.

## CI

`.github/workflows/ci.yml` jobs: `unit` (3.10–3.13 matrix: `ruff check src/ tests/ scripts/` → `render_preset_tables.py --check` → `pyright src/` on 3.12 → `pytest -m "not slow and not integration"`), `gotui` (`gofmt -l`, `go vet`, `go test` in `gotui/`), `build` (`python -m build`, `twine check`, wheel version == `__version__`), and `slow` (weekly cron + `workflow_dispatch`: `pytest tests/test_pipeline.py -m slow` with a cached HF dir and CPU torch). No tolerance for ruff warnings; the config selects `E, F, W, I, UP, B, SIM` and ignores only `B008` (Typer uses function-call defaults by design). Line length is 120.

`.github/workflows/pages.yml` — deploys `docs/` to GitHub Pages (presentation, for-agents guide, reference).

## Sister projects

This is one of the "OraClaw" family. Stay consistent with the others (PicoOraClaw Go, IronOraClaw Rust, ZeroOraClaw Rust, OracLaw TS+Python, TinyOraClaw TS): Oracle AI Database is the headline feature, the README positions it as such, and the core value is "zero network round-trips — everything happens inside Oracle."
