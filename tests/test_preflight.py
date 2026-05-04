from onnx2oracle.preflight import PreflightCheck, PreflightResult


def test_preflight_result_ok_requires_all_checks_ok():
    result = PreflightResult(
        checks=[
            PreflightCheck("connect", True, "ok"),
            PreflightCheck("DBMS_VECTOR package", True, "ok"),
        ],
        elapsed_ms=1,
    )

    assert result.ok


def test_preflight_result_not_ok_when_any_check_fails():
    result = PreflightResult(
        checks=[
            PreflightCheck("connect", True, "ok"),
            PreflightCheck("VECTOR_EMBEDDING function", False, "missing"),
        ],
        elapsed_ms=1,
    )

    assert not result.ok
