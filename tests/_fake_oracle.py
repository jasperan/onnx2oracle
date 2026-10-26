"""Test double for the slice of Oracle that onnx2oracle talks to.

``FakeOracle().install(monkeypatch)`` makes every ``DSN.connect()`` return a fake connection backed
by an in-memory model catalog. It answers the statements onnx2oracle issues (catalog lookups,
``DROP_ONNX_MODEL`` / ``LOAD_ONNX_MODEL``, ``VECTOR_EMBEDDING``, ``PREDICTION``) and fails loudly on
any other SQL, so a test cannot silently depend on a query the fake does not understand.
"""

from __future__ import annotations

import array
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import oracledb
import pytest
from rich.console import Console

import onnx2oracle.cli as cli
from onnx2oracle.connection import DSN

DSN_ARGS = ["--dsn", "system/pw@localhost:1521/FREEPDB1"]

_EMBEDDING_MODEL = re.compile(r"VECTOR_EMBEDDING\((\w+) USING")
_PREDICTION_MODEL = re.compile(r"PREDICTION\((\w+) USING")
_TEXT_BIND = re.compile(r"t(\d+)")

_KNOWN_WORDS = {
    "king": [1.0, 0.9, 0.0, 0.0],
    "queen": [0.9, 1.0, 0.0, 0.0],
    "banana": [0.0, 0.1, 1.0, 0.0],
}


def make_embedder(dims: int = 8, scale: float = 1.0) -> Callable[[str], list[float] | None]:
    """Deterministic embedder: king/queen are close, banana is far, every vector has L2 norm *scale*.

    Empty text yields ``None`` because Oracle treats '' as NULL.
    """
    assert dims >= 4

    def embed(text: str) -> list[float] | None:
        if not text:
            return None
        base = _KNOWN_WORDS.get(text, [0.3, 0.3, 0.3, 0.3])
        norm = math.sqrt(sum(x * x for x in base))
        return [scale * x / norm for x in base] + [0.0] * (dims - len(base))

    return embed


def word_overlap_scorer(query: str, doc: str) -> float:
    """Relevance = number of words *query* and *doc* share (the Berlin doc beats the banana doc)."""
    return float(len(set(re.findall(r"\w+", query.lower())) & set(re.findall(r"\w+", doc.lower()))))


class FakeCursor:
    def __init__(self, db: FakeOracle) -> None:
        self._db = db
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        db = self._db
        params = params or {}
        db.statements.append(" ".join(sql.split()))
        for fragment, error in db.errors.items():
            if fragment in sql:
                raise error
        self._rows = []

        if "COUNT(*) FROM user_mining_models" in sql:
            self._rows = [(1 if params["name"] in db.models else 0,)]
        elif "mining_function FROM user_mining_models" in sql:
            function = db.models.get(params["n"])
            self._rows = [] if function is None else [(function,)]
        elif "DROP_ONNX_MODEL" in sql:
            db.models.pop(params["n"], None)
            db.events.append(("drop", params["n"]))
        elif "LOAD_ONNX_MODEL" in sql:
            metadata = json.loads(params["metadata"])
            db.models[params["model_name"]] = "REGRESSION" if metadata["function"] == "regression" else "EMBEDDING"
            db.events.append(("load", params["model_name"], params["model_data"], metadata))
        elif "VECTOR_EMBEDDING" in sql:
            self._rows = self._embed(sql, params)
        elif "PREDICTION" in sql:
            self._rows = self._predict(sql, params)
        else:
            raise AssertionError(f"FakeOracle received SQL it does not understand: {sql!r}")

    def _require_model(self, name: str, function: str) -> None:
        if self._db.models.get(name) != function:
            raise oracledb.DatabaseError(f"ORA-40284: model {name} does not exist")

    def _embed(self, sql: str, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        db = self._db
        match = _EMBEDDING_MODEL.search(sql)
        assert match, sql
        self._require_model(match.group(1), "EMBEDDING")
        indexes = sorted(int(m.group(1)) for key in params if (m := _TEXT_BIND.fullmatch(key)))
        assert indexes == list(range(len(indexes))), f"binds must be t0..tN, got {sorted(params)}"
        db.events.append(("embed", len(indexes)))
        rows: list[tuple[Any, ...]] = []
        for i in indexes:
            vec = db.embedder(params[f"t{i}"])
            rows.append((i, None if vec is None else array.array("f", vec)))
        return rows[::-1] if db.shuffle_rows else rows

    def _predict(self, sql: str, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        db = self._db
        match = _PREDICTION_MODEL.search(sql)
        assert match, sql
        self._require_model(match.group(1), "REGRESSION")
        db.events.append(("predict", params["q"], params["d"]))
        return [(db.scorer(params["q"], params["d"]),)]

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows, self._rows = self._rows, []
        return rows


class FakeConnection:
    def __init__(self, db: FakeOracle) -> None:
        self._db = db

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._db)

    def commit(self) -> None:
        self._db.commits += 1

    def close(self) -> None:
        self._db.open_connections -= 1


class FakeOracle:
    """In-memory Oracle. ``models`` maps model name -> MINING_FUNCTION ("EMBEDDING" or "REGRESSION")."""

    def __init__(
        self,
        models: dict[str, str] | None = None,
        *,
        embedder: Callable[[str], list[float] | None] | None = None,
        scorer: Callable[[str, str], float | None] = word_overlap_scorer,
        shuffle_rows: bool = False,
    ) -> None:
        self.models: dict[str, str] = dict(models or {})
        self.embedder = embedder or make_embedder()
        self.scorer = scorer
        self.shuffle_rows = shuffle_rows
        self.connect_error: Exception | None = None
        # Raise the mapped exception for any statement containing the key, e.g. "LOAD_ONNX_MODEL".
        self.errors: dict[str, Exception] = {}
        # Ordered log of what reached the database: ("connect",), ("drop", name), ("load", name, bytes, metadata),
        # ("embed", n_texts), ("predict", query, doc). Tests append ("build", ...) to it to prove ordering.
        self.events: list[tuple[Any, ...]] = []
        self.statements: list[str] = []
        self.dsns: list[DSN] = []
        self.commits = 0
        self.open_connections = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeOracle:
        def connect(dsn: DSN, *_args: Any, **_kwargs: Any) -> FakeConnection:
            self.events.append(("connect",))
            self.dsns.append(dsn)
            if self.connect_error is not None:
                raise self.connect_error
            self.open_connections += 1
            return FakeConnection(self)

        monkeypatch.setattr(DSN, "connect", connect)
        return self

    def kinds(self, *wanted: str) -> list[str]:
        """The event kinds in order, restricted to *wanted*."""
        return [event[0] for event in self.events if event[0] in wanted]


def isolate_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Detach CLI tests from the developer's environment; returns the (non-existent) config file path."""
    for var in ("ORACLE_DSN", "ORACLE_PWD", "ORACLE_PORT"):
        monkeypatch.delenv(var, raising=False)
    config_path = tmp_path / ".onnx2oracle" / "config.toml"
    monkeypatch.setattr(cli, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(cli, "console", Console(width=200))
    return config_path
