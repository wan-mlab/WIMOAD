"""Statistical utilities for WIMOAD result comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import binomtest


@dataclass(frozen=True)
class McNemarResult:
    """Paired correctness counts for McNemar's test."""

    a: int
    b: int
    c: int
    d: int
    p_value: float


def exact_mcnemar_from_correctness(
    model1_correct: pd.Series, model2_correct: pd.Series
) -> McNemarResult:
    """Compute exact McNemar's test from paired correctness indicators.

    Cells follow the manuscript notation:
    a: both models correct
    b: model 1 correct, model 2 incorrect
    c: model 1 incorrect, model 2 correct
    d: both models incorrect
    """

    m1 = model1_correct.astype(bool)
    m2 = model2_correct.astype(bool)
    if len(m1) != len(m2):
        raise ValueError("model1_correct and model2_correct must have the same length")

    a = int((m1 & m2).sum())
    b = int((m1 & ~m2).sum())
    c = int((~m1 & m2).sum())
    d = int((~m1 & ~m2).sum())
    p_value = 1.0
    if b + c > 0:
        p_value = float(
            binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue
        )
    return McNemarResult(a=a, b=b, c=c, d=d, p_value=p_value)


def threshold_correct(
    labels: pd.Series, scores: pd.Series, threshold: float = 0.5
) -> pd.Series:
    """Convert scores to hard predictions and return correctness."""

    predictions = (scores.astype(float) >= threshold).astype(int)
    return predictions == labels.astype(int)

