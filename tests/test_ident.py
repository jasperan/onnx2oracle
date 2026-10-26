import pytest

from onnx2oracle._ident import validate_oracle_name


def test_valid_names_pass():
    for n in ["ALL_MINILM_L6_V2", "BGE_SMALL_EN_V1_5", "A", "_X"]:
        assert validate_oracle_name(n) == n


def test_lowercase_rejected():
    with pytest.raises(ValueError):
        validate_oracle_name("all_minilm")


def test_sql_injection_rejected():
    for bad in ["FOO) FROM DUAL; DROP TABLE X--", "FOO'; SELECT 1--", "FOO BAR"]:
        with pytest.raises(ValueError):
            validate_oracle_name(bad)


def test_trailing_newline_rejected():
    with pytest.raises(ValueError):
        validate_oracle_name("FOO\n")


def test_empty_and_overlong_rejected():
    for bad in ["", "A" * 129]:
        with pytest.raises(ValueError):
            validate_oracle_name(bad)
    assert validate_oracle_name("A" * 128) == "A" * 128
