"""Guards for the local Oracle compose file.

`.audit/SECURITY.md` records a HIGH for the original file: it published the local Oracle
listener on every interface and set the SYSTEM password to a default, so the database was
reachable (with a public password) from the network. The fix binds 127.0.0.1 and lives in
two copies of the same file - `docker/docker-compose.yml` for humans and
`src/onnx2oracle/data/docker-compose.yml`, which the CLI reads at runtime (cli.py:303).

Nothing kept those two copies in sync, so the reviewed fix could survive in one and regress
in the other. These tests fail if that happens.
"""

from importlib.resources import files
from pathlib import Path

REPO_COPY = Path(__file__).resolve().parents[1] / "docker" / "docker-compose.yml"
PACKAGE_COPY = Path(str(files("onnx2oracle") / "data" / "docker-compose.yml"))


def test_both_compose_copies_are_byte_identical():
    """The documented file and the file the CLI launches must not drift apart."""
    assert REPO_COPY.read_bytes() == PACKAGE_COPY.read_bytes(), (
        "docker/docker-compose.yml and src/onnx2oracle/data/docker-compose.yml differ; "
        "the CLI would launch a different database than the one the docs describe"
    )


def test_published_ports_are_bound_to_loopback_in_both_copies():
    """A published port without a host IP is reachable from the whole network."""
    for path in (REPO_COPY, PACKAGE_COPY):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            value = stripped[2:].strip().strip('"').strip("'")
            # A port mapping looks like "1521:1521" or "127.0.0.1:1521:1521"; the first
            # form has no host address and therefore listens on every interface.
            parts = value.split(":")
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                raise AssertionError(
                    f"{path.name} publishes {value!r} without a host address; use "
                    f"'127.0.0.1:{value}'"
                )
