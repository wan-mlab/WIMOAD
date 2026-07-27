import pandas as pd
import pytest

from wimoad.integration import fuse_scores, prediction_column


def test_fuse_scores_uses_expression_weight():
    expression = pd.Series([1.0, 0.0])
    methylation = pd.Series([0.0, 1.0])

    fused = fuse_scores(expression, methylation, expression_weight=0.25)

    assert fused.tolist() == [0.25, 0.75]


def test_prediction_column_maps_auc_variants():
    assert prediction_column("AUC") == "Prediction"
    assert prediction_column("AUC_f") == "Prediction_f"


def test_prediction_column_rejects_unknown_metric():
    with pytest.raises(ValueError):
        prediction_column("Accuracy")

