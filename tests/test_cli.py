from typer.testing import CliRunner

from onnx2oracle.cli import app

runner = CliRunner()


def test_cli_help_succeeds():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "onnx2oracle" in result.stdout.lower()


def test_presets_command_lists_all():
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0
    for name in [
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "all-mpnet-base-v2",
        "bge-small-en-v1.5",
        "nomic-embed-text-v1",
    ]:
        assert name in result.stdout


def test_load_requires_preset_or_from_hf():
    result = runner.invoke(app, ["load"])
    assert result.exit_code != 0
