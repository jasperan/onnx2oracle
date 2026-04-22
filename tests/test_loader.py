"""Unit tests for loader.py (non-DB coverage).

Full DB coverage is in tests/test_loader_integration.py.
"""

import json

from onnx2oracle.loader import build_metadata_json


def test_metadata_json_embedding_shape():
    js = build_metadata_json()
    parsed = json.loads(js)
    assert parsed["function"] == "embedding"
    assert parsed["embeddingOutput"] == "embedding"
    assert parsed["input"] == {"pre_text": ["DATA"]}


def test_metadata_is_valid_json_string():
    js = build_metadata_json()
    json.loads(js)  # must not raise
