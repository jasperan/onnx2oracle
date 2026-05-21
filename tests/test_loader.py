"""Unit tests for loader.py (non-DB coverage).

Full DB coverage is in tests/test_loader_integration.py.
"""

import json

import pytest

from onnx2oracle.loader import build_metadata_json


def test_metadata_json_embedding_shape():
    js = build_metadata_json()
    parsed = json.loads(js)
    assert parsed["function"] == "embedding"
    assert parsed["embeddingOutput"] == "embedding"
    assert parsed["input"] == {"pre_text": ["DATA"]}


def test_metadata_json_embedding_explicit_task():
    parsed = json.loads(build_metadata_json("embedding"))
    assert parsed["function"] == "embedding"


def test_metadata_json_reranker_shape():
    parsed = json.loads(build_metadata_json("reranker"))
    assert parsed["function"] == "regression"
    assert parsed["regressionOutput"] == "logits"
    assert parsed["input"] == {
        "pre_text_1": ["DATA1"],
        "pre_text_2": ["DATA2"],
    }


def test_metadata_json_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        build_metadata_json("classification")


def test_metadata_is_valid_json_string():
    json.loads(build_metadata_json())
    json.loads(build_metadata_json("reranker"))
