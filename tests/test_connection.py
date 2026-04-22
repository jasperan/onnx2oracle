import os
from pathlib import Path

import pytest

from onnx2oracle.connection import DSN, resolve_dsn


def test_cli_arg_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("ORACLE_DSN", "env/env@env:1521/env")
    dsn = resolve_dsn(cli_dsn="user/pw@host:1521/FREEPDB1", target=None, config_path=tmp_path / "nope.toml")
    assert dsn.user == "user"
    assert dsn.password == "pw"
    assert dsn.host == "host"
    assert dsn.port == 1521
    assert dsn.service == "FREEPDB1"


def test_env_var_second(monkeypatch, tmp_path):
    monkeypatch.setenv("ORACLE_DSN", "envuser/envpw@envhost:1522/ENV")
    dsn = resolve_dsn(cli_dsn=None, target=None, config_path=tmp_path / "nope.toml")
    assert dsn.user == "envuser"
    assert dsn.host == "envhost"
    assert dsn.port == 1522


def test_config_third(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[default]\ndsn = "cfg/cfgpw@cfghost:1523/CFG"\n')
    dsn = resolve_dsn(cli_dsn=None, target=None, config_path=cfg)
    assert dsn.user == "cfg"
    assert dsn.host == "cfghost"
    assert dsn.port == 1523


def test_target_local_shortcut(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    dsn = resolve_dsn(cli_dsn=None, target="local", config_path=tmp_path / "nope.toml")
    assert dsn.user == "system"
    assert dsn.password == "onnx2oracle"
    assert dsn.host == "localhost"
    assert dsn.port == 1521
    assert dsn.service == "FREEPDB1"


def test_no_source_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    with pytest.raises(ValueError, match="No DSN"):
        resolve_dsn(cli_dsn=None, target=None, config_path=tmp_path / "nope.toml", interactive=False)


def test_parse_with_special_chars_in_password():
    dsn = DSN.parse("system/oracle@123!@localhost:1521/FREEPDB1")
    assert dsn.user == "system"
    assert dsn.password == "oracle@123!"
    assert dsn.host == "localhost"
