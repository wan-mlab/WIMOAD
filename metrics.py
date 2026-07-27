# metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix
)
import pandas as pd


F_METRIC_NAMES = ['Accuracy_f', 'AUC_f', 'F1_f', 'Recall_f', 'Precision_f', 'Specificity_f', 'MCC_f']


def _empty_f_metrics():
    return {name: None for name in F_METRIC_NAMES}


def calculate_metrics(group):
    y_true = group['DX_bl']
    metrics = {}

    # For 'Pred'
    y_pred = (group['Prediction'] >= 0.5).astype(int)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['AUC'] = roc_auc_score(y_true, group['Prediction']) if len(y_true.unique()) > 1 else None
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else None
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    # For 'Predf'
    if 'Prediction_f' not in group or group['Prediction_f'].isna().any():
        metrics.update(_empty_f_metrics())
        return pd.Series(metrics)

    y_pred_f = (group['Prediction_f'] >= 0.5).astype(int)
    metrics['Accuracy_f'] = accuracy_score(y_true, y_pred_f)
    metrics['AUC_f'] = roc_auc_score(y_true, group['Prediction_f']) if len(y_true.unique()) > 1 else None
    metrics['F1_f'] = f1_score(y_true, y_pred_f, zero_division=0)
    metrics['Recall_f'] = recall_score(y_true, y_pred_f, zero_division=0)
    metrics['Precision_f'] = precision_score(y_true, y_pred_f, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_f, labels=[0, 1]).ravel()
    metrics['Specificity_f'] = tn / (tn + fp) if (tn + fp) > 0 else None
    metrics['MCC_f'] = matthews_corrcoef(y_true, y_pred_f)

    return pd.Series(metrics)
