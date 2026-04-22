"""DSN resolution: CLI > env > config file > target shortcut > interactive prompt."""

from __future__ import annotations

import os
import re
import sys
import tomllib
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".onnx2oracle" / "config.toml"


@dataclass(frozen=True)
class DSN:
    user: str
    password: str
    host: str
    port: int
    service: str

    @classmethod
    def parse(cls, raw: str) -> "DSN":
        # user/password@host:port/service — password may contain '@'; split on the LAST
        # '@...<host>:<port>/<service>' suffix.
        m = re.match(r"^(?P<user>[^/]+)/(?P<rest>.+)$", raw)
        if not m:
            raise ValueError(f"Malformed DSN: {raw!r}")
        user = m.group("user")
        rest = m.group("rest")
        idx = rest.rfind("@")
        while idx >= 0:
            candidate = rest[idx + 1:]
            tail = re.match(r"^(?P<host>[^:/]+):(?P<port>\d+)/(?P<service>.+)$", candidate)
            if tail:
                return cls(
                    user=user,
                    password=rest[:idx],
                    host=tail.group("host"),
                    port=int(tail.group("port")),
                    service=tail.group("service"),
                )
            idx = rest.rfind("@", 0, idx)
        raise ValueError(f"Malformed DSN: {raw!r}")

    def to_oracle_dsn(self) -> str:
        """Return the host:port/service string for python-oracledb's connect()."""
        return f"{self.host}:{self.port}/{self.service}"

    def display(self) -> str:
        """Safe-for-logs representation (no password)."""
        return f"{self.user}@{self.host}:{self.port}/{self.service}"


def _local_dsn() -> DSN:
    return DSN(
        user="system",
        password="onnx2oracle",
        host="localhost",
        port=1521,
        service="FREEPDB1",
    )


def resolve_dsn(
    cli_dsn: str | None,
    target: str | None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    interactive: bool = True,
) -> DSN:
    """Resolve a DSN in precedence order: CLI > env > config file > target > prompt."""
    if cli_dsn:
        return DSN.parse(cli_dsn)
    env = os.environ.get("ORACLE_DSN")
    if env:
        return DSN.parse(env)
    if config_path.exists():
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if "default" in data and "dsn" in data["default"]:
            return DSN.parse(data["default"]["dsn"])
    if target == "local":
        return _local_dsn()
    if interactive and sys.stdin.isatty():
        print("No DSN resolved. Enter connection details:")
        user = input("  user: ").strip()
        password = getpass("  password: ")
        host = input("  host [localhost]: ").strip() or "localhost"
        port = int(input("  port [1521]: ").strip() or "1521")
        service = input("  service [FREEPDB1]: ").strip() or "FREEPDB1"
        return DSN(user=user, password=password, host=host, port=port, service=service)
    raise ValueError("No DSN: provide --dsn, set ORACLE_DSN, write a config, or pass --target local")
