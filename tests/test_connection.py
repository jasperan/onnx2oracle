
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
    monkeypatch.delenv("ORACLE_PWD", raising=False)
    monkeypatch.delenv("ORACLE_PORT", raising=False)
    dsn = resolve_dsn(cli_dsn=None, target="local", config_path=tmp_path / "nope.toml")
    assert dsn.user == "system"
    assert dsn.password == "onnx2oracle"
    assert dsn.host == "localhost"
    assert dsn.port == 1521
    assert dsn.service == "FREEPDB1"


def test_target_local_shortcut_honors_oracle_pwd(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    monkeypatch.setenv("ORACLE_PWD", "custom-local-pw")

    dsn = resolve_dsn(cli_dsn=None, target="local", config_path=tmp_path / "nope.toml")

    assert dsn.user == "system"
    assert dsn.password == "custom-local-pw"
    assert dsn.host == "localhost"
    assert dsn.port == 1521
    assert dsn.service == "FREEPDB1"


def test_target_local_shortcut_honors_oracle_port(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    monkeypatch.setenv("ORACLE_PORT", "1524")

    dsn = resolve_dsn(cli_dsn=None, target="local", config_path=tmp_path / "nope.toml")

    assert dsn.host == "localhost"
    assert dsn.port == 1524
    assert dsn.service == "FREEPDB1"


def test_target_local_shortcut_rejects_invalid_oracle_port(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    monkeypatch.setenv("ORACLE_PORT", "not-a-port")

    with pytest.raises(ValueError, match="ORACLE_PORT must be an integer"):
        resolve_dsn(cli_dsn=None, target="local", config_path=tmp_path / "nope.toml")


def test_no_source_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    with pytest.raises(ValueError, match="No DSN"):
        resolve_dsn(cli_dsn=None, target=None, config_path=tmp_path / "nope.toml", interactive=False)


def test_parse_with_special_chars_in_password():
    dsn = DSN.parse("system/oracle@123!@localhost:1521/FREEPDB1")
    assert dsn.user == "system"
    assert dsn.password == "oracle@123!"
    assert dsn.host == "localhost"


def test_parse_tns_alias_passes_connect_string_through():
    dsn = DSN.parse("admin/oracle@123!@mydb_high")

    assert dsn.user == "admin"
    assert dsn.password == "oracle@123!"
    assert dsn.to_oracle_dsn() == "mydb_high"
    assert dsn.display() == "admin@mydb_high"


def test_parse_adb_descriptor_passes_connect_string_through():
    descriptor = (
        "(description=(address=(protocol=tcps)(port=1522)(host=adb.example.com))"
        "(connect_data=(service_name=abc_high.adb.oraclecloud.com)))"
    )
    dsn = DSN.parse(f"admin/oracle@123!@{descriptor}")

    assert dsn.user == "admin"
    assert dsn.password == "oracle@123!"
    assert dsn.to_oracle_dsn() == descriptor
    assert dsn.display() == "admin@<connect-descriptor>"


def test_explicit_target_local_overrides_env_dsn(monkeypatch, tmp_path):
    monkeypatch.setenv("ORACLE_DSN", "envuser/envpw@remote.example.com:1522/ENV")
    monkeypatch.setenv("ORACLE_PWD", "localpw")
    monkeypatch.setenv("ORACLE_PORT", "1555")
    dsn = resolve_dsn(cli_dsn=None, target="local", config_path=tmp_path / "nope.toml")
    assert (dsn.user, dsn.host, dsn.port, dsn.password) == ("system", "localhost", 1555, "localpw")


def test_explicit_target_local_overrides_config_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    monkeypatch.delenv("ORACLE_PORT", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[default]\ndsn = "u/p@remote.example.com:1523/SVC"\n')
    dsn = resolve_dsn(cli_dsn=None, target="local", config_path=cfg)
    assert dsn.host == "localhost"


def test_cli_dsn_still_beats_explicit_target(monkeypatch, tmp_path):
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    dsn = resolve_dsn(cli_dsn="u/p@dbhost:1521/SVC", target="local", config_path=tmp_path / "nope.toml")
    assert dsn.host == "dbhost"


def test_implicit_default_local_still_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ORACLE_DSN", "envuser/envpw@envhost:1522/ENV")
    dsn = resolve_dsn(cli_dsn=None, target=None, config_path=tmp_path / "nope.toml", default_local=True)
    assert dsn.host == "envhost"


def test_repr_and_str_hide_password():
    dsn = DSN.parse("system/S3cr3t-pw!@localhost:1521/FREEPDB1")
    for rendered in (repr(dsn), str(dsn), f"{dsn}", f"{dsn!r}", repr([dsn])):
        assert "S3cr3t-pw" not in rendered
        assert "system" in rendered
    assert dsn.password == "S3cr3t-pw!"


def test_malformed_dsn_error_hides_password():
    for raw in ("user/S3cr3t-pw-no-host", "S3cr3t-pw-no-slash"):
        with pytest.raises(ValueError) as exc:
            DSN.parse(raw)
        assert "S3cr3t" not in str(exc.value)
        assert "expected user/password@host:port/service" in str(exc.value)


def test_malformed_dsn_from_env_hides_password(monkeypatch, tmp_path):
    monkeypatch.setenv("ORACLE_DSN", "user/S3cr3t-pw-no-host")
    with pytest.raises(ValueError) as exc:
        resolve_dsn(cli_dsn=None, target=None, config_path=tmp_path / "nope.toml")
    assert "S3cr3t" not in str(exc.value)
