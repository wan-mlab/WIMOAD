# metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix
)
import pandas as pd


def calculate_metrics(group):
    y_true = group['DX_bl']
    metrics = {}

    # For 'Pred'
    y_pred = (group['Prediction'] >= 0.5).astype(int)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['AUC'] = roc_auc_score(y_true, group['Prediction']) if len(y_true.unique()) > 1 else None
    metrics['F1'] = f1_score(y_true, y_pred)
    metrics['Recall'] = recall_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else None
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    # For 'Predf'
    if 'Prediction_f' in group and group['Prediction_f'].notnull().all():
        if (group['Prediction_f'] != 0).all():
            y_pred_f = (group['Prediction_f'] >= 0.5).astype(int)
            metrics['Accuracy_f'] = accuracy_score(y_true, y_pred_f)
            metrics['AUC_f'] = roc_auc_score(y_true, group['Prediction_f']) if len(y_true.unique()) > 1 else None
            metrics['F1_f'] = f1_score(y_true, y_pred_f)
            metrics['Recall_f'] = recall_score(y_true, y_pred_f)
            metrics['Precision_f'] = precision_score(y_true, y_pred_f)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_f).ravel()
            metrics['Specificity_f'] = tn / (tn + fp) if (tn + fp) > 0 else None
            metrics['MCC_f'] = matthews_corrcoef(y_true, y_pred_f)
        else:
            metrics['Accuracy_f'] = metrics['AUC_f'] = metrics['F1_f'] = metrics['Recall_f'] = metrics['Precision_f'] = metrics['Specificity_f'] = metrics['MCC_f'] = None
    else:
        metrics['Accuracy_f'] = metrics['AUC_f'] = metrics['F1_f'] = metrics['Recall_f'] = metrics['Precision_f'] = metrics['Specificity_f'] = metrics['MCC_f'] = None

    return pd.Series(metrics)