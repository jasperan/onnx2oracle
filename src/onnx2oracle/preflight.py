"""Database preflight checks before loading ONNX models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import oracledb

from onnx2oracle.connection import DSN


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class PreflightResult:
    checks: list[PreflightCheck]
    elapsed_ms: int

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def _one(cur: oracledb.Cursor, sql: str, params: dict[str, object] | None = None) -> object | None:
    cur.execute(sql, params or {})
    row = cur.fetchone()
    if row is None:
        return None
    return row[0]


def run_preflight(dsn: DSN) -> PreflightResult:
    """Check whether *dsn* is ready for DBMS_VECTOR ONNX model loading."""
    t0 = time.perf_counter()
    checks: list[PreflightCheck] = []

    try:
        conn = oracledb.connect(
            user=dsn.user,
            password=dsn.password,
            dsn=dsn.to_oracle_dsn(),
            tcp_connect_timeout=30,
        )
    except Exception as exc:
        return PreflightResult(
            checks=[
                PreflightCheck(
                    name="connect",
                    ok=False,
                    detail=str(exc),
                )
            ],
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
        )

    try:
        cur = conn.cursor()
        checks.append(PreflightCheck("connect", True, dsn.display()))

        try:
            banner = str(_one(cur, "SELECT banner FROM v$version FETCH FIRST 1 ROW ONLY") or "")
            supported = "23ai" in banner or "26ai" in banner
            checks.append(
                PreflightCheck(
                    "database version",
                    supported,
                    banner or "no banner returned",
                )
            )
        except Exception as exc:
            checks.append(PreflightCheck("database version", False, str(exc)))

        try:
            dbms_vector_status = _one(
                cur,
                """
                SELECT status
                FROM all_objects
                WHERE owner = 'SYS'
                  AND object_name = 'DBMS_VECTOR'
                  AND object_type = 'PACKAGE'
                """,
            )
            checks.append(
                PreflightCheck(
                    "DBMS_VECTOR package",
                    dbms_vector_status == "VALID",
                    f"SYS.DBMS_VECTOR:{dbms_vector_status or 'missing'}",
                )
            )
        except Exception as exc:
            checks.append(PreflightCheck("DBMS_VECTOR package", False, str(exc)))

        try:
            vector_embedding_count = int(
                _one(
                    cur,
                    "SELECT COUNT(*) FROM v$sqlfn_metadata WHERE name = 'VECTOR_EMBEDDING'",
                )
                or 0
            )
            checks.append(
                PreflightCheck(
                    "VECTOR_EMBEDDING function",
                    vector_embedding_count > 0,
                    "available" if vector_embedding_count else "not found in v$sqlfn_metadata",
                )
            )
        except Exception as exc:
            checks.append(PreflightCheck("VECTOR_EMBEDDING function", False, str(exc)))

        try:
            mining_privs = int(
                _one(
                    cur,
                    """
                    SELECT COUNT(*)
                    FROM session_privs
                    WHERE privilege IN ('CREATE MINING MODEL', 'CREATE ANY MINING MODEL')
                    """,
                )
                or 0
            )
            checks.append(
                PreflightCheck(
                    "CREATE MINING MODEL privilege",
                    mining_privs > 0,
                    "available in session" if mining_privs else "missing from session_privs",
                )
            )
        except Exception as exc:
            checks.append(PreflightCheck("CREATE MINING MODEL privilege", False, str(exc)))

        try:
            execute_privs = int(
                _one(
                    cur,
                    """
                    SELECT COUNT(*)
                    FROM all_tab_privs
                    WHERE table_schema = 'SYS'
                      AND table_name = 'DBMS_VECTOR'
                      AND privilege = 'EXECUTE'
                    """,
                )
                or 0
            )
            execute_any = int(
                _one(
                    cur,
                    "SELECT COUNT(*) FROM session_privs WHERE privilege = 'EXECUTE ANY PROCEDURE'",
                )
                or 0
            )
            checks.append(
                PreflightCheck(
                    "EXECUTE on DBMS_VECTOR",
                    execute_privs > 0 or execute_any > 0,
                    "available" if execute_privs > 0 or execute_any > 0 else "missing",
                )
            )
        except Exception as exc:
            checks.append(PreflightCheck("EXECUTE on DBMS_VECTOR", False, str(exc)))

        try:
            model_count = int(_one(cur, "SELECT COUNT(*) FROM user_mining_models") or 0)
            checks.append(
                PreflightCheck(
                    "mining model catalog",
                    True,
                    f"user_mining_models visible ({model_count} existing)",
                )
            )
        except Exception as exc:
            checks.append(PreflightCheck("mining model catalog", False, str(exc)))

        return PreflightResult(checks=checks, elapsed_ms=int((time.perf_counter() - t0) * 1000))
    finally:
        conn.close()
