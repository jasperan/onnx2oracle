import pytest

from onnx2oracle.presets import PRESETS, ModelSpec, get_preset


def test_all_presets_registered():
    expected = {
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "all-mpnet-base-v2",
        "bge-small-en-v1.5",
        "nomic-embed-text-v1",
    }
    assert set(PRESETS.keys()) == expected


def test_get_preset_returns_modelspec():
    spec = get_preset("all-MiniLM-L6-v2")
    assert isinstance(spec, ModelSpec)
    assert spec.hf_repo == "sentence-transformers/all-MiniLM-L6-v2"
    assert spec.dims == 384
    assert spec.pooling == "mean"
    assert spec.normalize is True
    assert spec.oracle_name == "ALL_MINILM_L6_V2"


def test_get_preset_unknown_raises():
    with pytest.raises(KeyError):
        get_preset("no-such-model")


def test_bge_uses_cls_pooling():
    spec = get_preset("bge-small-en-v1.5")
    assert spec.pooling == "cls"
    assert spec.oracle_name == "BGE_SMALL_EN_V1_5"


def test_oracle_names_are_uppercase_identifiers():
    for spec in PRESETS.values():
        assert spec.oracle_name.isupper()
        assert all(c.isalnum() or c == "_" for c in spec.oracle_name)
