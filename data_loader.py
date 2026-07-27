# data loader
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif


def _flatten_label_map(label_map):
    merged = {}
    for mapping in label_map:
        merged.update(mapping)
    return merged


def _apply_label_map(labels, label_map):
    mapping = _flatten_label_map(label_map)
    if set(mapping).issubset(set(labels.dropna().unique())):
        return labels.map(mapping)

    ordered_labels = sorted(labels.dropna().unique())
    encoded_labels = labels.map({label: idx for idx, label in enumerate(ordered_labels)})
    return encoded_labels.map(mapping)


def _load_dataset(filename, label_map, drop_columns=None):
    df = pd.read_csv(filename)
    if drop_columns:
        df.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    df.set_index('RID', inplace=True)
    df['DX_bl'] = _apply_label_map(df['DX_bl'], label_map)
    df = df[(df['DX_bl'] == 0) | (df['DX_bl'] == 1)]
    X = df.drop(['DX_bl'], axis=1)
    y = df['DX_bl'].astype(int)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}, Positive labels: {y.sum()}")

    return X, y


def load_data(filename, label_map, drop_columns=None):
    """Load and preprocess an expression dataset."""
    return _load_dataset(filename, label_map, drop_columns=drop_columns)


def load_data_m(filename, label_map, drop_columns=None):
    """Load and preprocess a methylation dataset."""
    return _load_dataset(filename, label_map, drop_columns=drop_columns)

def feature_selection(X_train, y_train, X_val, k=1000):
    """Perform variance thresholding and ANOVA feature selection."""
    # Variance thresholding
    variance_selector = VarianceThreshold(threshold=1e-5)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)
    if X_train_var.shape[1] == 0:
        raise ValueError("No features remain after variance thresholding")

    # ANOVA feature selection
    selected_k = min(k, X_train_var.shape[1])
    anova_selector = SelectKBest(score_func=f_classif, k=selected_k)
    X_train_selected = anova_selector.fit_transform(X_train_var, y_train)
    X_val_selected = anova_selector.transform(X_val_var)

    return pd.DataFrame(X_train_selected, index=X_train.index), pd.DataFrame(X_val_selected, index=X_val.index)
