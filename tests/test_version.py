import re
import sys
from pathlib import Path

from typer.testing import CliRunner

import onnx2oracle
import onnx2oracle.cli as cli

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_pyproject_takes_version_from_package_attribute():
    """The release version lives only in ``onnx2oracle.__version__`` — pyproject must not carry a second copy."""
    data = tomllib.loads(PYPROJECT.read_text())
    assert "version" not in data["project"]
    assert "version" in data["project"]["dynamic"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "onnx2oracle.__version__"}


def test_version_is_pep440_release():
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", onnx2oracle.__version__)


def test_version_command_prints_package_version():
    result = CliRunner().invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert onnx2oracle.__version__ in result.stdout
