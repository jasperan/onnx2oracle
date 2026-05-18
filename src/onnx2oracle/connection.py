"""DSN resolution: CLI > env > config file > target shortcut > interactive prompt."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

DEFAULT_CONFIG_PATH = Path.home() / ".onnx2oracle" / "config.toml"
DEFAULT_LOCAL_PASSWORD = "onnx2oracle"
DEFAULT_LOCAL_PORT = 1521


@dataclass(frozen=True)
class DSN:
    user: str
    password: str
    host: str
    port: int | None
    service: str = ""
    connect_string: str | None = None

    @classmethod
    def parse(cls, raw: str) -> DSN:
        # user/password@host:port/service — password may contain '@'; scan from
        # the right so Easy Connect strings still parse when passwords contain @.
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

        idx = rest.rfind("@")
        if 0 <= idx < len(rest) - 1:
            connect_string = rest[idx + 1:]
            return cls(
                user=user,
                password=rest[:idx],
                host=connect_string,
                port=None,
                service="",
                connect_string=connect_string,
            )
        raise ValueError(f"Malformed DSN: {raw!r}")

    def to_oracle_dsn(self) -> str:
        """Return the host:port/service string for python-oracledb's connect()."""
        if self.connect_string is not None:
            return self.connect_string
        if self.port is None:
            return self.host
        return f"{self.host}:{self.port}/{self.service}"

    def display(self) -> str:
        """Safe-for-logs representation (no password)."""
        if self.connect_string is not None:
            target = "<connect-descriptor>" if self.connect_string.lstrip().startswith("(") else self.connect_string
            return f"{self.user}@{target}"
        return f"{self.user}@{self.host}:{self.port}/{self.service}"


def _local_dsn() -> DSN:
    raw_port = os.environ.get("ORACLE_PORT", str(DEFAULT_LOCAL_PORT))
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"ORACLE_PORT must be an integer, got {raw_port!r}") from exc

    return DSN(
        user="system",
        password=os.environ.get("ORACLE_PWD", DEFAULT_LOCAL_PASSWORD),
        host="localhost",
        port=port,
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
