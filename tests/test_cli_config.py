"""`onnx2oracle config show|set` and the config file they manage."""

import stat
import sys

import pytest
from typer.testing import CliRunner

import onnx2oracle.cli as cli
from onnx2oracle.connection import resolve_dsn

from ._fake_oracle import isolate_cli

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

runner = CliRunner()

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


@pytest.fixture
def config_path(monkeypatch, tmp_path):
    return isolate_cli(monkeypatch, tmp_path)


def _set(arg: str):
    return runner.invoke(cli.app, ["config", "set", arg])


def _show():
    return runner.invoke(cli.app, ["config", "show"])


def _read(path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


# --- config set: values survive a round trip through TOML ---------------------------------------------------------

AWKWARD_DSNS = [
    "system/plain@localhost:1521/FREEPDB1",
    'system/pa"ss@localhost:1521/FREEPDB1',
    "system/pa\\ss\\\\@localhost:1521/FREEPDB1",
    "system/p#ss [x] {y}@localhost:1521/FREEPDB1",
    "system/p=ss=@host:1521/svc",
    "system/it's \"quoted\"@host:1521/svc",
    "system/tab\there@host:1521/svc",
    "system/new\nline@host:1521/svc",
    "system/cr\rlf\r\n@host:1521/svc",
    "system/del\x7fchar\x01@host:1521/svc",
    "system/ünïcödé-✓-日本@host:1521/svc",
    "system/trailing space @host:1521/svc",
    "[default]",
    "# not a comment",
    "",
]


@pytest.mark.parametrize("value", AWKWARD_DSNS, ids=[repr(v)[:40] for v in AWKWARD_DSNS])
def test_set_stores_any_value_exactly(config_path, value):
    result = _set(f"dsn={value}")

    assert result.exit_code == 0, result.output
    assert _read(config_path) == {"default": {"dsn": value}}


def test_set_value_is_read_back_by_the_dsn_resolver(config_path):
    assert _set('dsn=system/pa"ss\\w@db.example.com:1522/SVC').exit_code == 0

    dsn = resolve_dsn(cli_dsn=None, target=None, config_path=config_path)

    assert (dsn.user, dsn.password, dsn.host, dsn.port, dsn.service) == (
        "system", 'pa"ss\\w', "db.example.com", 1522, "SVC"
    )


# --- config set: where keys go ------------------------------------------------------------------------------------


def test_set_creates_the_default_table(config_path):
    assert _set("dsn=u/p@h:1/s").exit_code == 0

    assert config_path.read_text(encoding="utf-8").startswith("[default]\n")


@pytest.mark.parametrize(
    "existing_line",
    [
        'dsn = "old"',
        'dsn="old"',
        'dsn   =   "old"   # trailing comment',
        '\tdsn\t=\t"old"',
        '"dsn" = "old"',
        "dsn = 'old'",
    ],
)
def test_set_replaces_an_existing_key_whatever_its_spacing(config_path, existing_line):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(f'[default]\n{existing_line}\nother = 1\n', encoding="utf-8")

    assert _set("dsn=new/pw@h:1/s").exit_code == 0

    assert _read(config_path) == {"default": {"dsn": "new/pw@h:1/s", "other": 1}}


def test_set_keeps_the_rest_of_the_file_and_stays_inside_the_default_table(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '# my settings\n[default]\ndsn = "old/pw@h:1/s"\n\n[extra]\ndsn = "untouched"\n', encoding="utf-8"
    )

    assert _set("cache_dir=/tmp/cache").exit_code == 0
    assert _set("dsn=new/pw@h:2/s").exit_code == 0

    text = config_path.read_text(encoding="utf-8")
    assert text.startswith("# my settings\n")
    assert _read(config_path) == {
        "default": {"dsn": "new/pw@h:2/s", "cache_dir": "/tmp/cache"},
        "extra": {"dsn": "untouched"},
    }


def test_set_appends_a_default_table_when_only_other_tables_exist(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[extra]\ndsn = "untouched"\n', encoding="utf-8")

    assert _set("dsn=u/p@h:1/s").exit_code == 0

    assert _read(config_path) == {"extra": {"dsn": "untouched"}, "default": {"dsn": "u/p@h:1/s"}}


def test_set_keeps_top_level_keys_out_of_the_default_table(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text('top = "level"\n', encoding="utf-8")

    assert _set("dsn=u/p@h:1/s").exit_code == 0

    assert _read(config_path) == {"top": "level", "default": {"dsn": "u/p@h:1/s"}}


def test_set_repairs_a_file_broken_by_the_old_unescaped_writer(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[default]\ndsn = "system/pa"ss@h:1/s"\n', encoding="utf-8")

    assert _set("dsn=system/fixed@h:1/s").exit_code == 0

    assert _read(config_path) == {"default": {"dsn": "system/fixed@h:1/s"}}


def test_set_refuses_to_write_when_the_result_would_not_be_valid_toml(config_path):
    config_path.parent.mkdir(parents=True)
    broken = '[default]\ndsn = "unterminated\n'
    config_path.write_text(broken, encoding="utf-8")

    result = _set("other=1")

    assert result.exit_code == 1
    assert "Refusing to write" in result.stdout
    assert config_path.read_text(encoding="utf-8") == broken
    assert isinstance(result.exception, SystemExit)


@pytest.mark.parametrize("arg", ["no-equals-sign", "bad key=1", "=value", 'quo"te=1', "a.b=1"])
def test_set_rejects_malformed_arguments_without_touching_the_file(config_path, arg):
    result = _set(arg)

    assert result.exit_code == 2
    assert not config_path.exists()


def test_set_message_with_markup_like_key_does_not_crash(config_path):
    result = _set("[/x]=1")

    assert result.exit_code == 2
    assert "[/x]" in result.stdout


# --- config set: permissions --------------------------------------------------------------------------------------


@posix_only
def test_set_creates_the_file_private_and_its_directory_private(config_path):
    assert _set("dsn=u/secret@h:1/s").exit_code == 0

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(config_path.parent.stat().st_mode) == 0o700


@posix_only
def test_set_tightens_a_world_readable_config_left_by_an_older_version(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[default]\ndsn = "u/secret@h:1/s"\n', encoding="utf-8")
    config_path.chmod(0o664)

    assert _set("other=1").exit_code == 0

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


@posix_only
def test_set_writes_through_a_symlinked_config_without_replacing_the_link(config_path, tmp_path):
    real = tmp_path / "dotfiles" / "onnx2oracle.toml"
    real.parent.mkdir()
    real.write_text('[default]\ndsn = "old/pw@h:1/s"\n', encoding="utf-8")
    config_path.parent.mkdir(parents=True)
    config_path.symlink_to(real)

    assert _set("dsn=new/pw@h:1/s").exit_code == 0

    assert config_path.is_symlink()
    assert _read(real) == {"default": {"dsn": "new/pw@h:1/s"}}


def test_set_leaves_no_temporary_files_behind(config_path):
    assert _set("dsn=u/p@h:1/s").exit_code == 0
    assert _set("dsn=u/p@h:2/s").exit_code == 0

    assert [p.name for p in config_path.parent.iterdir()] == ["config.toml"]


# --- config show --------------------------------------------------------------------------------------------------


def test_show_prints_the_default_table_header(config_path):
    assert _set("dsn=system/pw@localhost:1521/FREEPDB1").exit_code == 0

    result = _show()

    assert result.exit_code == 0, result.output
    assert "[default]" in result.stdout
    assert "dsn = " in result.stdout


def test_show_masks_the_dsn_password_but_not_the_rest(config_path):
    assert _set("dsn=scott/S3cr3t@db.example.com:1522/SVC").exit_code == 0

    out = _show().stdout

    assert "S3cr3t" not in out
    assert 'dsn = "scott/***@db.example.com:1522/SVC"' in out


@pytest.mark.parametrize("password", ['pa"ss@w', "p\\a@ss", "x]y[z", "a#b", "ünï"])
def test_show_masks_passwords_that_needed_escaping(config_path, password):
    assert _set(f"dsn=scott/{password}@db.example.com:1522/SVC").exit_code == 0

    out = _show().stdout

    assert password not in out
    assert "scott/***@db.example.com:1522/SVC" in out


def test_show_masks_connect_descriptors_and_aliases(config_path):
    assert _set("dsn=admin/S3cr3t@mydb_high").exit_code == 0

    out = _show().stdout

    assert "S3cr3t" not in out
    assert "admin/***@mydb_high" in out


def test_show_masks_token_and_password_like_keys(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '[default]\nhf_token = "hf_abcdef"\nwallet_password="w4ll3t"\nApiKey = \'k3y\'\ncache_dir = "/tmp/cache"\n',
        encoding="utf-8",
    )

    out = _show().stdout

    for secret in ("hf_abcdef", "w4ll3t", "k3y"):
        assert secret not in out
    assert 'cache_dir = "/tmp/cache"' in out
    assert out.count('"***"') == 3


def test_show_masks_a_multi_line_secret_completely(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '[default]\nhf_token = """\nline-one-xyz\nline-two-xyz\n"""\ncache_dir = "/tmp/cache"\n', encoding="utf-8"
    )

    out = _show().stdout

    assert "xyz" not in out
    assert 'cache_dir = "/tmp/cache"' in out


@pytest.mark.parametrize(
    "text",
    [
        'default.dsn = "scott/S3cr3t@h:1/s"\n',
        'default = { dsn = "scott/S3cr3t@h:1/s" }\n',
        '[default]\n"dsn" = "scott/S3cr3t@h:1/s"\n',
        "[default]\ndsn = '''scott/S3cr3t@h:1/s'''\n",
        '[profiles.prod]\ndsn = "scott/S3cr3t@h:1/s"\n',
        '[[profiles]]\nname = "a"\ndsn = "scott/S3cr3t@h:1/s"\n',
    ],
)
def test_show_masks_the_dsn_password_however_the_key_is_spelled(config_path, text):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(text, encoding="utf-8")

    result = _show()

    assert result.exit_code == 0, result.output
    assert "S3cr3t" not in result.stdout
    assert "scott/***@h:1/s" in result.stdout


def test_show_masks_everything_inside_a_secret_looking_table_or_list(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '[credentials]\nuser = "u-zzz1"\nlogin = "l-zzz2"\n\n[default]\nhf_tokens = ["t-zzz3", "t-zzz4"]\n',
        encoding="utf-8",
    )

    out = _show().stdout

    assert "zzz" not in out
    assert "[credentials]" in out
    assert '"***"' in out


def test_show_masks_a_dsn_that_does_not_parse_entirely(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[default]\ndsn = "S3cr3t-with-no-structure"\n', encoding="utf-8")

    out = _show().stdout

    assert "S3cr3t" not in out
    assert 'dsn = "***"' in out


def test_show_output_is_the_same_configuration_minus_secrets(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "# a comment\n"
        'title = "top level"\n'
        "[default]\n"
        'cache_dir = "/tmp/a \\"quoted\\" path\\twith\\\\backslash"\n'
        "retries = 3\n"
        "ratio = 0.25\n"
        "verbose = true\n"
        "hosts = ['a', 'b']\n"
        "since = 2024-01-02\n"
        "[default.sub]\n"
        "deep = { x = 1, y = [1, 2] }\n"
        "[[servers]]\n"
        'name = "one"\n'
        "[[servers]]\n"
        'name = "two"\n',
        encoding="utf-8",
    )
    original = _read(config_path)

    out = _show().stdout

    assert "# a comment" not in out
    assert tomllib.loads(out) == original


def test_show_names_the_file_it_printed(config_path):
    assert _set("dsn=u/p@h:1/s").exit_code == 0

    assert str(config_path) in _show().stdout


def test_show_does_not_interpret_markup_in_values(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[default]\ncache_dir = "[red]x [/oops] [bold]y :smile:"\n', encoding="utf-8")

    result = _show()

    assert result.exit_code == 0, result.output
    assert 'cache_dir = "[red]x [/oops] [bold]y :smile:"' in result.stdout


def test_show_refuses_to_print_a_file_that_is_not_valid_toml(config_path):
    config_path.parent.mkdir(parents=True)
    config_path.write_text('[default]\ndsn = "system/S3cr3t@h:1/s" and junk\n', encoding="utf-8")

    result = _show()

    assert result.exit_code == 1
    assert "not valid TOML" in result.stdout
    assert "line 2" in result.stdout
    assert "S3cr3t" not in result.stdout
    assert isinstance(result.exception, SystemExit)


def test_show_without_a_config_file(config_path):
    result = _show()

    assert result.exit_code == 0
    assert "No config at" in result.stdout
