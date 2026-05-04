#!/usr/bin/env python
"""Run real Oracle compatibility checks for one or more presets.

The output is JSON Lines so agents can append, diff, and summarize evidence
without scraping terminal text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import oracledb  # noqa: E402

from onnx2oracle.connection import resolve_dsn  # noqa: E402
from onnx2oracle.loader import drop_model, model_exists, upload_model  # noqa: E402
from onnx2oracle.pipeline import build_augmented  # noqa: E402
from onnx2oracle.presets import PRESETS, get_preset  # noqa: E402
from onnx2oracle.verify import smoke_test  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "integration-artifacts" / f"model-compat-{stamp}.jsonl"


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _cleanup_model(dsn, oracle_name: str) -> None:
    conn = oracledb.connect(user=dsn.user, password=dsn.password, dsn=dsn.to_oracle_dsn(), tcp_connect_timeout=30)
    try:
        if model_exists(conn, oracle_name):
            drop_model(conn, oracle_name)
    finally:
        conn.close()


def _check_preset(name: str, dsn, output: Path, keep_model: bool) -> bool:
    spec = get_preset(name)
    started = time.perf_counter()
    base: dict[str, Any] = {
        "event": "model_compatibility",
        "started_at": _utc_now(),
        "preset": name,
        "hf_repo": spec.hf_repo,
        "oracle_name": spec.oracle_name,
        "dims": spec.dims,
        "pooling": spec.pooling,
        "normalize": spec.normalize,
        "target": dsn.display(),
    }

    try:
        data = build_augmented(spec)
        upload_model(dsn, data, spec.oracle_name, force=True)
        verify = smoke_test(dsn, spec.oracle_name)
        ok = verify.connected and verify.model_registered and verify.similarity_sane
        ok = ok and verify.sample_embedding_dims == spec.dims
        record = {
            **base,
            "status": "passed" if ok else "failed",
            "onnx_bytes": len(data),
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "verify": {
                "connected": verify.connected,
                "model_registered": verify.model_registered,
                "sample_embedding_dims": verify.sample_embedding_dims,
                "sample_embedding_norm": verify.sample_embedding_norm,
                "similarity_sane": verify.similarity_sane,
                "elapsed_ms": verify.elapsed_ms,
                "error": verify.error,
            },
        }
        _write_record(output, record)
        return ok
    except Exception as exc:
        _write_record(
            output,
            {
                **base,
                "status": "error",
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            },
        )
        return False
    finally:
        if not keep_model:
            try:
                _cleanup_model(dsn, spec.oracle_name)
            except Exception as exc:
                _write_record(
                    output,
                    {
                        **base,
                        "event": "cleanup_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("presets", nargs="*", help="Preset names to test. Defaults to all-MiniLM-L6-v2.")
    parser.add_argument("--all-presets", action="store_true", help="Test every preset in onnx2oracle.presets.")
    parser.add_argument("--target", default="local", help="DSN target shortcut; default: local.")
    parser.add_argument("--dsn", help="Explicit DSN: user/pw@host:port/service.")
    parser.add_argument("--output", type=Path, default=_default_output(), help="JSONL evidence output path.")
    parser.add_argument("--keep-models", action="store_true", help="Leave loaded models in Oracle after the run.")
    args = parser.parse_args()

    names = list(PRESETS) if args.all_presets else args.presets or ["all-MiniLM-L6-v2"]
    unknown = [name for name in names if name not in PRESETS]
    if unknown:
        parser.error(f"unknown preset(s): {', '.join(unknown)}; available: {', '.join(PRESETS)}")

    os.environ.setdefault("ORACLE_PWD", "onnx2oracle")
    dsn = resolve_dsn(cli_dsn=args.dsn, target=args.target, interactive=False)

    _write_record(
        args.output,
        {
            "event": "run_start",
            "started_at": _utc_now(),
            "target": dsn.display(),
            "presets": names,
        },
    )
    results = [_check_preset(name, dsn, args.output, args.keep_models) for name in names]
    passed = sum(1 for result in results if result)
    _write_record(
        args.output,
        {
            "event": "run_finish",
            "finished_at": _utc_now(),
            "passed": passed,
            "failed": len(results) - passed,
            "output": str(args.output),
        },
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
