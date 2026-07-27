"""Build supplemental McNemar tables from WIMOAD result CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .integration import fuse_scores, load_model_scores, prediction_column, variant_name
from .statistics import exact_mcnemar_from_correctness, threshold_correct


TASK_ORDER = ["ae", "al", "am", "ca", "ce", "cl", "cm", "cp", "el"]


def single_sample_path(results_dir: Path, task_code: str, modality: str) -> Path:
    if modality == "Expression":
        return results_dir / f"results_{task_code}_single_samples.csv"
    if modality == "Methylation":
        return results_dir / f"results_m_{task_code}_single_samples.csv"
    raise ValueError(f"Unsupported modality: {modality!r}")


def metric_path(results_dir: Path, task_code: str, modality: str) -> Path:
    if modality == "Expression":
        return results_dir / f"results_{task_code}_evaluation_metrics.csv"
    if modality == "Methylation":
        return results_dir / f"results_m_{task_code}_evaluation_metrics.csv"
    raise ValueError(f"Unsupported modality: {modality!r}")


def prepare_integration_scores(
    results_dir: Path,
    best_models: pd.DataFrame,
    alpha_table: pd.DataFrame,
    task_code: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, float, float]:
    """Load expression/met model scores and return matched fused predictions."""

    task_best = best_models[best_models["Task_Code"] == task_code]
    expr_best = task_best[task_best["Modality"] == "Expression"].iloc[0]
    met_best = task_best[task_best["Modality"] == "Methylation"].iloc[0]
    expression_weight = float(
        alpha_table.loc[alpha_table["Task_Code"] == task_code, "Expression_Weight"].iloc[0]
    )
    methylation_weight = 1.0 - expression_weight

    expr = load_model_scores(
        str(single_sample_path(results_dir, task_code, "Expression")),
        expr_best["Combination"],
        expr_best["MetaModel"],
        prediction_column(expr_best["Metric_Type"]),
        "expression_score",
    )
    met = load_model_scores(
        str(single_sample_path(results_dir, task_code, "Methylation")),
        met_best["Combination"],
        met_best["MetaModel"],
        prediction_column(met_best["Metric_Type"]),
        "methylation_score",
    )
    paired = expr.merge(met, on="RID", how="inner", suffixes=("_expr", "_met"))
    if not (paired["label_expr"] == paired["label_met"]).all():
        raise ValueError(f"Expression/Methylation labels do not match for {task_code}")
    paired["label"] = paired["label_expr"].astype(int)
    paired["integration_score"] = fuse_scores(
        paired["expression_score"], paired["methylation_score"], expression_weight
    )
    paired["integration_correct"] = threshold_correct(
        paired["label"], paired["integration_score"]
    )
    return paired, expr_best, met_best, expression_weight, methylation_weight


def best_base_within_modality(
    results_dir: Path, task_code: str, modality: str
) -> dict[str, object]:
    """Select the best base model within one modality by available AUC variant."""

    metrics = pd.read_csv(metric_path(results_dir, task_code, modality))
    base = metrics[metrics["MetaModel"].astype(str).str.startswith("BaseModel_")]
    candidates: list[dict[str, object]] = []
    for _, row in base.iterrows():
        for metric_type in ["AUC", "AUC_f"]:
            if metric_type not in base.columns:
                continue
            value = row.get(metric_type)
            if pd.isna(value):
                continue
            candidates.append(
                {
                    "Combination": row["Combination"],
                    "MetaModel": row["MetaModel"],
                    "Metric_Type": metric_type,
                    "Metric_Value": float(value),
                }
            )
    if not candidates:
        raise ValueError(f"No base-model candidates for {task_code} {modality}")
    candidates.sort(key=lambda item: (item["Metric_Value"], item["MetaModel"]), reverse=True)
    return candidates[0]


def build_base_vs_integration_table(
    results_dir: Path, best_models_path: Path, alpha_path: Path
) -> pd.DataFrame:
    """Build S2-style table: best base model per modality vs fused integration."""

    best_models = pd.read_csv(best_models_path)
    alpha_table = pd.read_csv(alpha_path)[["group", "alpha"]].rename(
        columns={"group": "Task_Code", "alpha": "Expression_Weight"}
    )
    rows = []
    for task_code in TASK_ORDER:
        paired, expr_best, met_best, expression_weight, methylation_weight = (
            prepare_integration_scores(results_dir, best_models, alpha_table, task_code)
        )
        row = {
            "Groups": expr_best["Task"],
            "Task_Code": task_code,
            "Expression_Weight": expression_weight,
            "Methylation_Weight": methylation_weight,
        }
        for label, modality, score_col in [
            ("Exp", "Expression", "expression_score"),
            ("Met", "Methylation", "methylation_score"),
        ]:
            base = best_base_within_modality(results_dir, task_code, modality)
            base_scores = load_model_scores(
                str(single_sample_path(results_dir, task_code, modality)),
                str(base["Combination"]),
                str(base["MetaModel"]),
                prediction_column(str(base["Metric_Type"])),
                "base_score",
            )
            comparison = paired[["RID", "label", "integration_correct"]].merge(
                base_scores, on="RID", how="inner"
            )
            if not (comparison["label_x"] == comparison["label_y"]).all():
                raise ValueError(f"Label mismatch for {task_code} {modality}")
            base_correct = threshold_correct(comparison["label_x"], comparison["base_score"])
            result = exact_mcnemar_from_correctness(
                base_correct, comparison["integration_correct"]
            )
            row.update(
                {
                    f"{label}_Model": base["MetaModel"],
                    f"{label}_Variant": variant_name(str(base["Metric_Type"])),
                    f"{label}_N": int(len(comparison)),
                    f"{label}_a": result.a,
                    f"{label}_b": result.b,
                    f"{label}_c": result.c,
                    f"{label}_d": result.d,
                    f"{label}_p_value": result.p_value,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_integration_input_vs_fusion_table(
    results_dir: Path, best_models_path: Path, alpha_path: Path
) -> pd.DataFrame:
    """Build S2-style table: each integration input model vs fused integration."""

    best_models = pd.read_csv(best_models_path)
    alpha_table = pd.read_csv(alpha_path)[["group", "alpha"]].rename(
        columns={"group": "Task_Code", "alpha": "Expression_Weight"}
    )
    rows = []
    for task_code in TASK_ORDER:
        paired, expr_best, met_best, expression_weight, methylation_weight = (
            prepare_integration_scores(results_dir, best_models, alpha_table, task_code)
        )
        row = {
            "Groups": expr_best["Task"],
            "Task_Code": task_code,
            "N": int(len(paired)),
            "Expression_Weight": expression_weight,
            "Methylation_Weight": methylation_weight,
        }
        for label, score_col, best_row in [
            ("Exp", "expression_score", expr_best),
            ("Met", "methylation_score", met_best),
        ]:
            model_correct = threshold_correct(paired["label"], paired[score_col])
            result = exact_mcnemar_from_correctness(
                model_correct, paired["integration_correct"]
            )
            row.update(
                {
                    f"{label}_Model": best_row["MetaModel"],
                    f"{label}_Combination": best_row["Combination"],
                    f"{label}_Variant": variant_name(best_row["Metric_Type"]),
                    f"{label}_a": result.a,
                    f"{label}_b": result.b,
                    f"{label}_c": result.c,
                    f"{label}_d": result.d,
                    f"{label}_p_value": result.p_value,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)

