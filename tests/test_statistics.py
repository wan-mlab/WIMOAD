import math

import pandas as pd

from wimoad.statistics import exact_mcnemar_from_correctness


def test_exact_mcnemar_uses_discordant_pairs_only():
    # a=178, b=3, c=2, d=92 from the R3 Supplemental Table S2 example.
    m1 = pd.Series([True] * 178 + [True] * 3 + [False] * 2 + [False] * 92)
    m2 = pd.Series([True] * 178 + [False] * 3 + [True] * 2 + [False] * 92)
    result = exact_mcnemar_from_correctness(m1, m2)

    assert result.a == 178
    assert result.b == 3
    assert result.c == 2
    assert result.d == 92
    assert math.isclose(result.p_value, 1.0)


def test_exact_mcnemar_detects_imbalanced_discordant_pairs():
    m1 = pd.Series([True] * 94 + [True] * 26 + [False] * 47 + [False] * 41)
    m2 = pd.Series([True] * 94 + [False] * 26 + [True] * 47 + [False] * 41)
    result = exact_mcnemar_from_correctness(m1, m2)

    assert result.b == 26
    assert result.c == 47
    assert result.p_value < 0.05

