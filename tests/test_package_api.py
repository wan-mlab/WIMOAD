import wimoad


def test_package_exports_core_result_utilities():
    assert callable(wimoad.fuse_scores)
    assert callable(wimoad.exact_mcnemar_from_correctness)
    assert callable(wimoad.threshold_correct)
