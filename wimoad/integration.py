"""Score-level integration utilities for WIMOAD."""

from __future__ import annotations

import pandas as pd


def fuse_scores(
    expression_scores: pd.Series,
    methylation_scores: pd.Series,
    expression_weight: float,
) -> pd.Series:
    """Fuse expression and methylation scores using an expression weight."""

    if not 0 <= expression_weight <= 1:
        raise ValueError("expression_weight must be between 0 and 1")
    methylation_weight = 1.0 - expression_weight
    return (
        expression_weight * expression_scores.astype(float)
        + methylation_weight * methylation_scores.astype(float)
    )


def prediction_column(metric_type: str) -> str:
    """Map the selected AUC metric variant to the sample prediction column."""

    if metric_type == "AUC":
        return "Prediction"
    if metric_type == "AUC_f":
        return "Prediction_f"
    raise ValueError(f"Unsupported metric type: {metric_type!r}")


def variant_name(metric_type: str) -> str:
    """Return a short variant name for tables."""

    return "pred" if metric_type == "AUC" else "predf"


def load_model_scores(
    sample_csv: str,
    combination: str,
    metamodel: str,
    prediction_col: str,
    score_col: str,
) -> pd.DataFrame:
    """Load one model's sample-level scores from a WIMOAD single-sample CSV."""

    samples = pd.read_csv(sample_csv)
    subset = samples[
        (samples["Combination"] == combination) & (samples["MetaModel"] == metamodel)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No rows found for combination={combination!r}, metamodel={metamodel!r}"
        )
    if subset["RID"].duplicated().any():
        raise ValueError("Expected one prediction row per RID for the selected model")
    return subset[["RID", "DX_bl", prediction_col]].rename(
        columns={"DX_bl": "label", prediction_col: score_col}
    )

