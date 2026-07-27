"""Weighted multi-omics integration framework for WIMOAD.

Given each omics branch's mean stacking metrics (as produced by ``main.py``)
and its sample-level predictions, selects the best stacking model per branch
and searches the expression weight that maximizes the fused AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def classifier_count(combination):
    """Count base classifiers in a comma-joined combination string."""

    return len(str(combination).split(","))


def select_best_branch_model(mean_metrics, min_combination_size=3,
                              auc_col="avg_AUC", auc_f_col="avg_AUC_f"):
    """Pick the best stacking combination for one omics branch.

    Restricts candidates to combinations with at least `min_combination_size`
    base classifiers, then returns whichever AUC variant (plain or extended)
    scores higher.
    """

    counts = mean_metrics["Combination"].apply(classifier_count)
    candidates = mean_metrics[counts >= min_combination_size]
    if candidates.empty:
        raise ValueError(
            f"No combinations with at least {min_combination_size} classifiers"
        )

    best_plain = candidates.loc[candidates[auc_col].idxmax()]
    best_extended = candidates.loc[candidates[auc_f_col].idxmax()]

    if best_plain[auc_col] >= best_extended[auc_f_col]:
        return {
            "Combination": best_plain["Combination"],
            "MetaModel": best_plain["MetaModel"],
            "Metric_Type": "AUC",
            "Metric_Value": float(best_plain[auc_col]),
        }
    return {
        "Combination": best_extended["Combination"],
        "MetaModel": best_extended["MetaModel"],
        "Metric_Type": "AUC_f",
        "Metric_Value": float(best_extended[auc_f_col]),
    }


def extract_branch_predictions(single_samples, combination, metamodel, metric_type,
                                label_col="DX_bl", score_name="score"):
    """Pull one model's sample-level predictions from a WIMOAD single-sample table."""

    prediction_col = "Prediction" if metric_type == "AUC" else "Prediction_f"
    subset = single_samples[
        (single_samples["Combination"] == combination)
        & (single_samples["MetaModel"] == metamodel)
    ]
    if subset.empty:
        raise ValueError(
            f"No rows found for combination={combination!r}, metamodel={metamodel!r}"
        )
    if subset["SampleRID"].duplicated().any():
        raise ValueError("Expected one prediction row per SampleRID for the selected model")
    return subset[["SampleRID", label_col, prediction_col]].rename(
        columns={label_col: "label", prediction_col: score_name}
    )


def search_best_alpha(expression_scores, methylation_scores, labels, alpha_step=0.01):
    """Grid search the expression weight that maximizes fused AUC.

    Returns the (alpha, AUC) curve as a DataFrame plus the best alpha and AUC.
    """

    alphas = np.round(np.arange(0, 1 + alpha_step, alpha_step), 2)
    rows = []
    for alpha in alphas:
        fused = (
            alpha * expression_scores.astype(float)
            + (1 - alpha) * methylation_scores.astype(float)
        )
        rows.append((alpha, roc_auc_score(labels, fused)))
    fusion_curve = pd.DataFrame(rows, columns=["Alpha", "AUC"])
    best_row = fusion_curve.loc[fusion_curve["AUC"].idxmax()]
    return fusion_curve, float(best_row["Alpha"]), float(best_row["AUC"])


def run_group_integration(expression_mean_metrics, expression_samples,
                           methylation_mean_metrics, methylation_samples,
                           min_combination_size=3, alpha_step=0.01):
    """Select each branch's best stacking model and search the fusion weight."""

    best_expression = select_best_branch_model(expression_mean_metrics, min_combination_size)
    best_methylation = select_best_branch_model(methylation_mean_metrics, min_combination_size)

    expression_scores = extract_branch_predictions(
        expression_samples, best_expression["Combination"], best_expression["MetaModel"],
        best_expression["Metric_Type"], score_name="expression_score",
    )
    methylation_scores = extract_branch_predictions(
        methylation_samples, best_methylation["Combination"], best_methylation["MetaModel"],
        best_methylation["Metric_Type"], score_name="methylation_score",
    )

    paired = expression_scores.merge(
        methylation_scores, on="SampleRID", suffixes=("_expr", "_met")
    )
    if not (paired["label_expr"] == paired["label_met"]).all():
        raise ValueError("Expression and methylation labels do not match for paired samples")

    fusion_curve, best_alpha, best_auc = search_best_alpha(
        paired["expression_score"], paired["methylation_score"], paired["label_expr"], alpha_step,
    )

    return {
        "best_expression": best_expression,
        "best_methylation": best_methylation,
        "expression_weight": best_alpha,
        "methylation_weight": round(1 - best_alpha, 2),
        "fusion_auc": best_auc,
        "fusion_curve": fusion_curve,
    }
