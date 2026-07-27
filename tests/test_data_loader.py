import pandas as pd

from data_loader import _apply_label_map


def test_legacy_integer_label_map_uses_deterministic_label_order():
    labels = pd.Series(["MCI", "CN", "AD", "CN", "AD"])

    mapped = _apply_label_map(labels, [{0: 1, 1: 0}])

    assert mapped.isna().tolist() == [True, False, False, False, False]
    assert mapped.dropna().tolist() == [0.0, 1.0, 0.0, 1.0]


def test_raw_label_map_is_applied_directly():
    labels = pd.Series(["MCI", "CN", "AD", "CN", "AD"])

    mapped = _apply_label_map(labels, [{"CN": 0, "AD": 1}])

    assert mapped.isna().tolist() == [True, False, False, False, False]
    assert mapped.dropna().tolist() == [0.0, 1.0, 0.0, 1.0]
