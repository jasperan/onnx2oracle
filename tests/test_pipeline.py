"""Unit tests for the augmented ONNX pipeline.

These tests do real HF downloads on first run — they are slow (~30-60s first time,
cached thereafter under ~/.cache/huggingface/). They do NOT require Oracle.

Run with: pytest tests/test_pipeline.py -v -m slow
"""

import onnx
import pytest

from onnx2oracle.pipeline import build_augmented
from onnx2oracle.presets import get_preset


@pytest.mark.slow
def test_build_augmented_miniLM_L6_shape_and_valid_onnx():
    spec = get_preset("all-MiniLM-L6-v2")
    data = build_augmented(spec)  # Uses default HF cache
    assert isinstance(data, bytes)
    assert len(data) > 1_000_000  # augmented ONNX is >1 MB
    model = onnx.load_from_string(data)
    onnx.checker.check_model(model)
    outs = list(model.graph.output)
    assert len(outs) == 1
    assert outs[0].name == "embedding"
    shape = outs[0].type.tensor_type.shape.dim
    assert len(shape) == 1
    assert shape[0].dim_value == 384


@pytest.mark.slow
def test_build_augmented_runs_end_to_end_on_cpu():
    """Load the augmented model in onnxruntime, feed a string, get a 384-d L2-normalized vec."""
    import numpy as np
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path

    spec = get_preset("all-MiniLM-L6-v2")
    data = build_augmented(spec)

    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    sess = ort.InferenceSession(data, sess_options=so, providers=["CPUExecutionProvider"])

    # Input name is "pre_text" because pipeline.py uses prefix1="pre_" when merging
    input_names = [i.name for i in sess.get_inputs()]
    assert "pre_text" in input_names, f"expected pre_text in inputs, got {input_names}"
    input_name = "pre_text"

    out = sess.run(None, {input_name: np.array(["hello world"])})[0]
    assert out.shape == (384,)
    norm = float(np.linalg.norm(out))
    assert 0.99 < norm < 1.01, f"expected L2 norm ~1.0, got {norm}"
