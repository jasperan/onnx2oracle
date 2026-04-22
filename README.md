# onnx2oracle

**Load ONNX embedding models into Oracle AI Database with one command.**

[![CI](https://github.com/jasperan/onnx2oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/jasperan/onnx2oracle/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://jasperan.github.io/onnx2oracle/)
[![PyPI](https://img.shields.io/pypi/v/onnx2oracle.svg)](https://pypi.org/project/onnx2oracle/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bash
pip install onnx2oracle
onnx2oracle docker up
onnx2oracle load all-MiniLM-L6-v2 --target local
onnx2oracle verify --target local
```

Embeddings run entirely in-database via Oracle's `VECTOR_EMBEDDING` (no external API calls, no network round-trips, no serving layer).

## How it works

`onnx2oracle` downloads a sentence-transformer model from HuggingFace, wraps its tokenizer as ONNX ops, merges it with the transformer body, and appends pooling + L2 normalization. The resulting single-graph ONNX is uploaded to Oracle via `DBMS_VECTOR.LOAD_ONNX_MODEL`, after which you can query it with:

```sql
SELECT VECTOR_EMBEDDING(ALL_MINILM_L6_V2 USING 'hello world' AS DATA) FROM dual;
```

## Architecture

See [Architecture Diagram](docs/architecture.excalidraw) (Excalidraw, open at [excalidraw.com](https://excalidraw.com)) and [Architecture Overview](docs/architecture.html) for visual documentation of how the pieces fit together.

The augmented ONNX pipeline is the interesting bit. HuggingFace ships a Python tokenizer; Oracle needs an ONNX graph it can call directly. `pipeline.py` wraps the fast tokenizer as ONNX ops via `onnxruntime-extensions`, welds it onto the transformer body, bolts mean- or CLS-pooling on top, appends L2 normalization, and emits a single graph that goes string to float32 vector. That's what gets uploaded via `DBMS_VECTOR.LOAD_ONNX_MODEL`.

## Directory Structure

    src/onnx2oracle/
    ├── cli.py            # Typer commands: load, verify, presets, docker, config
    ├── presets.py        # 6 curated ModelSpecs
    ├── connection.py     # DSN resolution (CLI > env > toml > target > prompt)
    ├── pipeline.py       # HF model -> augmented ONNX bytes
    ├── loader.py         # DBMS_VECTOR.LOAD_ONNX_MODEL wrapper
    └── verify.py         # Smoke test via VECTOR_EMBEDDING
    docker/
    └── docker-compose.yml  # Oracle 26ai Free (gvenzl fallback documented)
    docs/                 # GitHub Pages site + architecture docs
    tests/                # Unit + integration (needs --run-integration)

## Presets

| Preset | Dims | Size | Pooling |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 90 MB | mean |
| `all-MiniLM-L12-v2` | 384 | 130 MB | mean |
| `all-mpnet-base-v2` | 768 | 420 MB | mean |
| `bge-small-en-v1.5` | 384 | 130 MB | cls |
| `nomic-embed-text-v1` | 768 | 540 MB | mean |

Any sentence-transformer-style HuggingFace model also works via `--from-huggingface`.

## Common tasks

```bash
# List all presets
onnx2oracle presets

# Load into a cloud ADB
onnx2oracle load all-mpnet-base-v2 --dsn 'app/pass@adb.region.oraclecloud.com:1522/xxx_high'

# Load a non-preset model
onnx2oracle load --from-huggingface BAAI/bge-base-en-v1.5 \
  --pooling cls --normalize --dims 768 --name BGE_BASE_EN_V1_5

# End-to-end verification
onnx2oracle verify --target local
```

## Requirements

- Python 3.10+
- Docker (for the local Oracle 26ai Free path) or any Oracle 23ai/26ai instance
- ~2 GB free RAM during model augmentation
- ~1 GB DB storage per preset

## Documentation

Full guide at **[jasperan.github.io/onnx2oracle](https://jasperan.github.io/onnx2oracle/)**.

## Development

```bash
git clone https://github.com/jasperan/onnx2oracle
cd onnx2oracle
conda create -n onnx2oracle python=3.12 -y
conda activate onnx2oracle
pip install -e ".[dev]"
pytest tests/ -v -m "not slow and not integration"
```

## License

MIT.
