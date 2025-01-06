# data loader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

def load_data(filename, label_map):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filename)
    df.drop(['DX_bl_nodia', 'DX_bl_dia', 'RID_nodia'], axis=1, inplace=True)
    df.set_index('RID', inplace=True)

    le = LabelEncoder()
    df['DX_bl'] = le.fit_transform(df['DX_bl'])

    for mapping in label_map:
        for orig, new in mapping.items():
            df.loc[df['DX_bl'] == orig, 'DX_bl'] = new

    df = df[(df['DX_bl'] == 0) | (df['DX_bl'] == 1)]
    X = df.drop(['DX_bl'], axis=1)
    y = df['DX_bl']
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}, Positive labels: {y.sum()}")

    return X, y

def load_data_m(filename, label_map):
    df = pd.read_csv(filename)
    df.set_index('RID', inplace=True)

    le = LabelEncoder()
    df['DX_bl'] = le.fit_transform(df['DX_bl'])

    for mapping in label_map:
        for orig, new in mapping.items():
            df.loc[df['DX_bl'] == orig, 'DX_bl'] = new
    
    df = df[(df['DX_bl'] == 0) | (df['DX_bl'] == 1)]
    X = df.drop(['DX_bl'], axis=1)
    y = df['DX_bl']
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}, Positive labels: {y.sum()}")

    return X, y

def feature_selection(X_train, y_train, X_val, k=1000):
    """Perform variance thresholding and ANOVA feature selection."""
    # Variance thresholding
    variance_selector = VarianceThreshold(threshold=1e-5)
    X_train_var = variance_selector.fit_transform(X_train)
    X_val_var = variance_selector.transform(X_val)

    # ANOVA feature selection
    anova_selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = anova_selector.fit_transform(X_train_var, y_train)
    X_val_selected = anova_selector.transform(X_val_var)

    return pd.DataFrame(X_train_selected, index=X_train.index), pd.DataFrame(X_val_selected, index=X_val.index)
