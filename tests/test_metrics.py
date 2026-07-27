import pandas as pd

from metrics import calculate_metrics


def test_prediction_f_zero_is_valid_score():
    group = pd.DataFrame(
        {
            "DX_bl": [0, 0, 1, 1],
            "Prediction": [0.1, 0.2, 0.8, 0.9],
            "Prediction_f": [0.0, 0.4, 0.6, 1.0],
        }
    )

    metrics = calculate_metrics(group)

    assert metrics["Accuracy_f"] == 1.0
    assert metrics["AUC_f"] == 1.0
