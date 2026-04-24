# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`onnx2oracle` ships a sentence-transformer HuggingFace repo into an Oracle AI Database 26ai instance as a single self-contained ONNX graph that Oracle's `DBMS_VECTOR.LOAD_ONNX_MODEL` can register. After registration, `VECTOR_EMBEDDING(MODEL USING :text AS DATA)` runs the full tokenizer + transformer + pooling + L2-normalize chain entirely in-database.

Published to PyPI as `onnx2oracle`. Python 3.10+. CLI entry point: `onnx2oracle` (Typer).

## Commands

```bash
# Env
conda create -n onnx2oracle python=3.12 -y && conda activate onnx2oracle
pip install -e ".[dev]"

# Lint / format-check (matches CI)
ruff check src/ tests/

# Fast tests (19 unit tests, no network, no DB)
pytest tests/ -v -m "not slow and not integration"

# Slow tests (real HuggingFace downloads, still no DB)
pytest tests/test_pipeline.py -v -m slow

# Integration test — needs a live Oracle 26ai Free container
ORACLE_DSN='system/yourpw@localhost:1521/FREEPDB1' \
  pytest tests/test_loader_integration.py --run-integration -v

# Single test
pytest tests/test_pipeline.py::test_build_augmented_minilm -v

# Local Oracle container (via the CLI, which uses the packaged compose file)
onnx2oracle docker up --wait     # first boot ~3-5 min
onnx2oracle docker down

# End-to-end loop
onnx2oracle load all-MiniLM-L6-v2 --target local
onnx2oracle verify --target local

# Release (PyPI) — after version bumps in pyproject.toml AND src/onnx2oracle/__init__.py
rm -rf dist/ build/ src/*.egg-info && python -m build && twine upload dist/*
```

The `--run-integration` flag is a custom pytest option (see `tests/conftest.py`); without it, tests marked `integration` are skipped regardless of `-m`.

## Architecture

Flow: HuggingFace repo → augmented ONNX bytes → Oracle `DBMS_VECTOR.LOAD_ONNX_MODEL` → SQL-callable `VECTOR_EMBEDDING`.

### The 6-stage ONNX augmentation (`src/onnx2oracle/pipeline.py`)

This is the non-obvious part. `build_augmented(spec)` returns raw bytes of a single ONNX graph with input `pre_text: string[1]` and output `embedding: float32[dims]`:

1. **Download core transformer** — prefers `onnx/model.onnx` from the HF repo; falls back to PyTorch `AutoModel` + `transformers.onnx.export` at opset 14.
2. **Generate tokenizer ONNX** via `onnxruntime_extensions.gen_processing_models(tokenizer, opset=14)`. If the tokenizer class isn't directly supported (e.g. MPNet, XLM-R) or doesn't emit the BERT-style `input_ids`/`attention_mask`/`token_type_ids` outputs, it falls back to `BertTokenizerFast`. If the vocab lacks `[UNK]` (i.e. SentencePiece/Unigram), it raises `NotImplementedError` — Oracle's BertTokenizer op cannot represent SentencePiece models like XLM-R or multilingual-e5.
3. **Align opsets** — bumps core to opset 18 via `version_converter`, syncs `ir_version`, and copies any custom domains from the tokenizer subgraph.
4. **Unsqueeze tokenizer outputs** — tokenizer emits `[seq_len]`, core expects `[1, seq_len]`. Rewires node outputs through `*_flat` and appends `Unsqueeze` nodes.
5. **Merge** — `onnx.compose.merge_models(pre, core, io_map=[…], prefix1="pre_", prefix2="core_")`. The `pre_` prefix is why the Oracle metadata JSON names the input `pre_text` (see `loader.build_metadata_json`). Only connects tensors present in both graphs (MPNet has no `token_type_ids`).
6. **Pool + L2-normalize + squeeze** — `ReduceMean(axis=1)` for mean pooling; `Gather(axis=1, [0]) + Squeeze` for CLS. L2 norm = `Pow(2) → ReduceSum(-1) → Sqrt → Max(eps=1e-12) → Div`. Final `Squeeze` drops the batch dim so the output is `float32[dims]`, not `float32[1, dims]`.

If you touch any pooling/normalize node names, the Oracle-side metadata (`embeddingOutput: "embedding"`) has to match.

### Loader idempotency (`src/onnx2oracle/loader.py`)

`upload_model` checks `user_mining_models` first. If the model exists and `force=False`, it's a no-op (returns, doesn't raise). With `force=True`, it calls `DBMS_VECTOR.DROP_ONNX_MODEL(force=>TRUE)` before re-uploading. The model bytes are passed as a bind variable to `LOAD_ONNX_MODEL`; metadata is `JSON(:metadata)` with `{"function": "embedding", "embeddingOutput": "embedding", "input": {"pre_text": ["DATA"]}}`.

### SQL-injection guard (`src/onnx2oracle/_ident.py`)

Oracle's `VECTOR_EMBEDDING(<model_name> USING :text AS DATA)` takes the model name as a **SQL token, not a bind variable**. Every code path that interpolates `oracle_name` into SQL goes through `validate_oracle_name()`, which enforces `/^[A-Z_][A-Z0-9_]{0,127}$/`. Anything else raises `ValueError` before the query runs. Do not add new SQL paths that bypass this.

### DSN resolution (`src/onnx2oracle/connection.py`)

Precedence: `--dsn` flag → `ORACLE_DSN` env → `~/.onnx2oracle/config.toml` (`[default] dsn = "…"`) → `--target local` shortcut (`system/onnx2oracle@localhost:1521/FREEPDB1`) → interactive prompt (tty only).

The CLI's `load` and `verify` commands auto-default to `--target local` when *none* of the first four are set, so the zero-config docker-compose flow works.

`DSN.parse` handles passwords containing `@` by scanning the `@host:port/service` suffix from the right.

### Verify (`src/onnx2oracle/verify.py`)

`smoke_test` returns a `VerifyResult` dataclass with: connected, registered, dims, L2 norm (expected ~1.0), and a cosine-sanity boolean (`cosine(king, queen) > cosine(king, banana)`). The CLI renders it with Rich checkmarks. Exits 1 if `error` is set.

### Docker compose

Two copies exist intentionally:
- `src/onnx2oracle/data/docker-compose.yml` — shipped inside the wheel; the CLI's `docker up/down/logs` subcommands point at this via `importlib.resources.files("onnx2oracle") / "data" / "docker-compose.yml"`.
- `docker/docker-compose.yml` — dev-only copy for git-clone workflow.

Both honor `ORACLE_PWD` (default `onnx2oracle`) and expose port 1521. Healthcheck retries for 10 minutes during the slow first PDB open.

## Testing notes

- `tests/conftest.py` adds the `--run-integration` flag. Integration tests are **skipped by default** even if you pass `-m integration` — you also need `--run-integration`.
- `test_pipeline.py::test_build_augmented_*` is marked `slow` because it actually downloads models from HuggingFace (~90 MB–540 MB depending on preset). CI skips these.
- `test_loader_integration.py` expects a live Oracle and reads `ORACLE_DSN`.
- Unit tests don't need network or DB; CI runs them on the 3.10 / 3.11 / 3.12 matrix.

## CI

`.github/workflows/ci.yml` — matrix install → `ruff check src/ tests/` → `pytest -m "not slow and not integration"`. No tolerance for ruff warnings; the existing config selects `E, F, W, I, UP, B, SIM` and ignores only `B008` (Typer uses function-call defaults by design). Line length is 120.

`.github/workflows/pages.yml` — deploys `docs/` to GitHub Pages (presentation, for-agents guide, reference).

## Sister projects

This is one of the "OraClaw" family. Stay consistent with the others (PicoOraClaw Go, IronOraClaw Rust, ZeroOraClaw Rust, OracLaw TS+Python, TinyOraClaw TS): Oracle AI Database is the headline feature, the README positions it as such, and the core value is "zero network round-trips — everything happens inside Oracle."
