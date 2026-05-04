"""Integration test: full pipeline against a live Oracle 26ai Free container.

Run with:
    pytest tests/test_loader_integration.py --run-integration -v -s

Defaults to the credentials that `docker/docker-compose.yml` provisions
(system/${ORACLE_PWD:-onnx2oracle}@localhost:${ORACLE_PORT:-1521}/FREEPDB1).
To point at a different container set ORACLE_DSN in the environment:

    ORACLE_DSN='user/password@host:port/service' pytest ... --run-integration
"""

import os

import pytest

from onnx2oracle.connection import DSN
from onnx2oracle.loader import drop_model, model_exists, upload_model
from onnx2oracle.pipeline import build_augmented
from onnx2oracle.presets import get_preset
from onnx2oracle.verify import smoke_test

pytestmark = pytest.mark.integration


def _get_dsn() -> DSN:
    env = os.environ.get("ORACLE_DSN")
    if env:
        return DSN.parse(env)
    # Default matches docker/docker-compose.yml (local Oracle 26ai Free)
    return DSN(
        user="system",
        password=os.environ.get("ORACLE_PWD", "onnx2oracle"),
        host="localhost",
        port=int(os.environ.get("ORACLE_PORT", "1521")),
        service="FREEPDB1",
    )


@pytest.mark.slow
def test_end_to_end_miniLM_L6():
    """Build augmented ONNX, load into Oracle, run VECTOR_EMBEDDING, verify shape+norm+sanity."""
    import oracledb

    spec = get_preset("all-MiniLM-L6-v2")
    dsn = _get_dsn()

    data = build_augmented(spec)
    # force=True so a prior run's artifact doesn't block us
    upload_model(dsn, data, spec.oracle_name, force=True)

    try:
        result = smoke_test(dsn, spec.oracle_name)
        print(f"\n\n=== VerifyResult ===\n{result}\n===\n")
        assert result.connected, result.error
        assert result.model_registered, result.error
        assert result.sample_embedding_dims == 384, result
        assert result.sample_embedding_norm is not None
        assert 0.99 < result.sample_embedding_norm < 1.01, (
            f"expected L2 norm ~1.0, got {result.sample_embedding_norm}"
        )
        assert result.similarity_sane, result
    finally:
        # Clean up — don't pollute the user's DB with a test model
        conn = oracledb.connect(user=dsn.user, password=dsn.password, dsn=dsn.to_oracle_dsn())
        try:
            if model_exists(conn, spec.oracle_name):
                drop_model(conn, spec.oracle_name)
                print(f"Cleaned up {spec.oracle_name}")
        finally:
            conn.close()
