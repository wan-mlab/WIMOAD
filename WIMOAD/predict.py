# Integration
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

def process_data_e(filename, label_mappings):
    tpm = pd.read_csv(filename)
    #'DX_bl_nodia', 'RID_nodia', 'DX_bl_dia', 'DX_bl'
    tpm = tpm.drop(['DX_bl_nodia', 'DX_bl_dia', 'RID_nodia'], axis=1)
    tpm.set_index('RID', inplace=True)  

    enc = LabelEncoder()
    enc.fit(tpm['DX_bl'])
    tpm['DX_bl'] = enc.transform(tpm['DX_bl'])

    for mapping in label_mappings:
        for original_label, new_label in mapping.items():
            tpm.loc[tpm['DX_bl'] == original_label, 'DX_bl'] = new_label

    tpm = tpm[(tpm['DX_bl'] == 0) | (tpm['DX_bl'] == 1)]

    X_traintpm = tpm.drop(['DX_bl'], axis=1)
    ytpm = tpm['DX_bl']

    X_traintpm = np.log(X_traintpm)
    X1 = X_traintpm
    y1 = ytpm

    k_best = SelectKBest(score_func=f_classif, k=1000)
    X_new = k_best.fit_transform(X1, y1)

    selected_features = k_best.get_support(indices=True)
    X_selected = X1.iloc[:, selected_features]
    X1 = X_selected

    return X1, y1

def process_data_m(filename, label_mappings):
    tpm = pd.read_csv(filename)
    tpm.set_index('RID', inplace=True)  

    enc = LabelEncoder()
    enc.fit(tpm['DX_bl'])
    tpm['DX_bl'] = enc.transform(tpm['DX_bl'])

    for mapping in label_mappings:
        for original_label, new_label in mapping.items():
            tpm.loc[tpm['DX_bl'] == original_label, 'DX_bl'] = new_label

    tpm = tpm[(tpm['DX_bl'] == 0) | (tpm['DX_bl'] == 1)]

    X_traintpm = tpm.drop(['DX_bl'], axis=1)
    ytpm = tpm['DX_bl']

    X1 = X_traintpm
    y1 = ytpm

    k_best = SelectKBest(score_func=f_classif, k=1000)
    X_new = k_best.fit_transform(X1, y1)

    selected_features = k_best.get_support(indices=True)
    X_selected = X1.iloc[:, selected_features]
    X1 = X_selected

    return X1, y1
def get_grid(X_train, y_train, X_test, y_test, param_grid):
    svc = SVC(random_state=0, probability=False)
    grid_model = GridSearchCV(svc, param_grid, n_jobs=-1, verbose=1, cv=10)
    grid_model.fit(X_train, y_train)

    pred_label = grid_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred_label)

    return grid_model, accuracy

def get_method_grid(X, y, methods):
    method_grid_models = {method['name']: {} for method in methods}
    # Split the data into train and test sets
    new_X_train, new_X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    param_grids = [
    {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
    'kernel': ['linear']},
    {'gamma': [2**(-10), 2**(-9), 2**(-8), 2**(-7), 2**(-6), 2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**(0), 2**(10), 2**(9), 2**(8), 2**(7), 2**(6), 2**(5), 2**(4), 2**(3), 2**(2), 2**(1)],
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
    'kernel': ['rbf']},
    {'gamma': [2**(-10), 2**(-9), 2**(-8), 2**(-7), 2**(-6), 2**(-5), 2**(-4), 2**(-3), 2**(-2), 2**(-1), 2**(0), 2**(10), 2**(9), 2**(8), 2**(7), 2**(6), 2**(5), 2**(4), 2**(3), 2**(2), 2**(1)],
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
    'kernel': ['poly']}
    ]

    for method_dict in methods:
        best_acc = 0.0
        best_model = None
        method_name = method_dict['name']
        print(method_name)
        for param_grid in param_grids:
            Xt_s, yt_s = method_train_process(new_X_train, y_train, method_name)
            grid_model, accuracy = get_grid(Xt_s, yt_s, new_X_test, y_test, param_grid=param_grid)
            method_grid_models[method_name] = grid_model
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = grid_model
        print(method_name)
        method_grid_models[method_name] = best_model
        print('Best_param:', method_grid_models[method_name].best_params_)
    return method_grid_models

def run_simple(X_train, y_train):
    Xt_s = X_train
    yt_s = y_train
    return Xt_s, yt_s

def run_smote(X_train, y_train):
    Xt_s, yt_s = SMOTE(random_state=2, k_neighbors=3).fit_resample(X_train, y_train)
    return Xt_s, yt_s

def run_ransele(X_train, y_train):
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]
    num_samples = min(len(class_0_indices), len(class_1_indices))
    selected_indices = np.concatenate((np.random.choice(class_0_indices, num_samples, replace=False),
                                        np.random.choice(class_1_indices, num_samples, replace=False)))
    Xt_s = X_train.iloc[selected_indices]
    yt_s = y_train.iloc[selected_indices]

    return Xt_s, yt_s

def run_ransmote(X_train, y_train):
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]

    if len(class_0_indices) > len(class_1_indices):
        majority_class_indices = class_0_indices
        minority_class_indices = class_1_indices
    else:
        majority_class_indices = class_1_indices
        minority_class_indices = class_0_indices

    half_majority_samples = math.ceil(len(majority_class_indices) / 2)
    selected_majority_indices = np.random.choice(majority_class_indices, half_majority_samples, replace=False)
    selected_minority_indices = minority_class_indices
    
    selected_indices = np.concatenate((selected_majority_indices, selected_minority_indices))
    X_train_selected = X_train.iloc[selected_indices]
    y_train_selected = y_train.iloc[selected_indices]
    Xt_s,yt_s = SMOTE(random_state=2,k_neighbors = 3).fit_resample(X_train_selected, y_train_selected)
    return Xt_s, yt_s


def method_train_process(X_train, y_train, name):
    if name == 'Origin':
        Xt_s, yt_s = run_simple(X_train, y_train)
    elif name == 'Oversampling_SMOTE':
        Xt_s, yt_s = run_smote(X_train, y_train)
    elif name == 'Undersampling':
        Xt_s, yt_s = run_ransele(X_train, y_train)
    elif name == 'Ransmote':
        Xt_s, yt_s = run_ransmote(X_train, y_train)
    else:
        raise ValueError("Invalid method selected")

    return Xt_s, yt_s

def k_folds(Xe, Xm, y, k, t, gride, gridm, methods):  
    results_per_cv = {method['name']: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'MCC': [],
                                'Sensitivity': [], 'Specificity': [], 'G_measure': [], 'Jaccard': [],
                                'AUC': [],'Accuracy_e': [], 'Precision_e': [], 'Recall_e': [], 'F1_e': [], 'MCC_e': [],
                                'Sensitivity_e': [], 'Specificity_e': [], 'G_measure_e': [], 'Jaccard_e': [],
                                'AUC_e': [], 'Accuracy_m': [], 'Precision_m': [], 'Recall_m': [], 'F1_m': [], 'MCC_m': [],
                                'Sensitivity_m': [], 'Specificity_m': [], 'G_measure_m': [], 'Jaccard_m': [],
                                'AUC_m': []} for method in methods}
    for j in range(t):
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=j)
        method_pred = {method['name']: {'y_oof': np.zeros(Xe.shape[0]), 'score': np.zeros(Xe.shape[0]), 'pro': np.zeros(shape = (Xe.shape[0],2)),
                                       'y_oof_e': np.zeros(Xe.shape[0]), 'score_e': np.zeros(Xe.shape[0]), 'pro_e': np.zeros(shape = (Xe.shape[0],2)),
                                       'y_oof_m': np.zeros(Xm.shape[0]), 'score_m': np.zeros(Xm.shape[0]), 'pro_m': np.zeros(shape = (Xm.shape[0],2))} for method in methods}
        for i, (train_idx, val_idx) in enumerate(folds.split(Xe, y)):
            for method_dict in methods:
                method_name = method_dict['name']
                Xet_s, yt_s = method_train_process(Xe.iloc[train_idx], y.iloc[train_idx], method_name)
                Xmt_s, yt_s = method_train_process(Xm.iloc[train_idx], y.iloc[train_idx], method_name)
                
                modele = SVC(random_state=0, probability=True, **gride[method_name].best_params_)
                modele.fit(Xet_s, yt_s)
                modelm = SVC(random_state=0, probability=True, **gridm[method_name].best_params_)
                modelm.fit(Xmt_s, yt_s)
                # Predictions
                method_pred[method_name]['pro'][val_idx] = 0.8*(modele.predict_proba(Xe.iloc[val_idx])+0.2*modelm.predict_proba(Xm.iloc[val_idx]))
                method_pred[method_name]['y_oof'][val_idx] = np.argmax(method_pred[method_name]['pro'][val_idx], axis = 1)
                method_pred[method_name]['score'][val_idx] = 0.8*(modele.decision_function(Xe.iloc[val_idx,])+0.2*modele.decision_function(Xe.iloc[val_idx,]))

                method_pred[method_name]['y_oof_e'][val_idx] = modele.predict(Xe.iloc[val_idx,])
                method_pred[method_name]['pro_e'][val_idx] = modele.predict_proba(Xe.iloc[val_idx,])
                method_pred[method_name]['score_e'][val_idx] = modele.decision_function(Xe.iloc[val_idx,])
                
                method_pred[method_name]['y_oof_m'][val_idx] = modelm.predict(Xm.iloc[val_idx,])
                method_pred[method_name]['pro_m'][val_idx] = modelm.predict_proba(Xm.iloc[val_idx,])
                method_pred[method_name]['score_m'][val_idx] = modelm.decision_function(Xm.iloc[val_idx,])


        for method_name, method_pred in method_pred.items():
            y_oof = method_pred['y_oof']
            y_prob = method_pred['pro']
            y_oof_e = method_pred['y_oof_e']
            y_prob_e = method_pred['pro_e']
            y_oof_m = method_pred['y_oof_m']
            y_prob_m = method_pred['pro_m']

            # Calculate performance metrics for the base model
            Acc = accuracy_score(y, y_oof)
            Auc = roc_auc_score(y, y_prob[:,1])
            precision = precision_score(y, y_oof)
            recall = recall_score(y, y_oof)
            f1 = f1_score(y, y_oof)
            MCC = matthews_corrcoef(y, y_oof)
            tn, fp, fn, tp = confusion_matrix(y, y_oof).ravel()
            Sensitivity = tp / (tp + fn)
            Specificity = tn / (tn + fp)
            G_measure = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
            Jaccard = tp / (tp + fp + fn)

            # Calculate performance metrics for the exp model
            Acc_e = accuracy_score(y, y_oof_e)
            Auc_e = roc_auc_score(y, y_prob_e[:,1])
            precision_e = precision_score(y, y_oof_e)
            recall_e = recall_score(y, y_oof_e)
            f1_e = f1_score(y, y_oof_e)
            MCC_e = matthews_corrcoef(y, y_oof_e)
            tn_e, fp_e, fn_e, tp_e = confusion_matrix(y, y_oof_e).ravel()
            Sensitivity_e = tp_e / (tp_e + fn_e)
            Specificity_e = tn_e / (tn_e + fp_e)
            G_measure_e = np.sqrt((tp_e / (tp_e + fn_e)) * (tn_e / (tn_e + fp_e)))
            Jaccard_e = tp_e / (tp_e + fp_e + fn_e)

            # Calculate performance metrics for the methylation model
            Acc_m = accuracy_score(y, y_oof_m)
            Auc_m = roc_auc_score(y, y_prob_m[:,1])
            precision_m = precision_score(y, y_oof_m)
            recall_m = recall_score(y, y_oof_m)
            f1_m = f1_score(y, y_oof_m)
            MCC_m = matthews_corrcoef(y, y_oof_m)
            tn_m, fp_m, fn_m, tp_m = confusion_matrix(y, y_oof_m).ravel()
            Sensitivity_m = tp_m / (tp_m + fn_m)
            Specificity_m = tn_m / (tn_m + fp_m)
            G_measure_m = np.sqrt((tp_m / (tp_m + fn_m)) * (tn_m / (tn_m + fp_m)))
            Jaccard_m = tp_m / (tp_m + fp_m + fn_m)

            # Append results to lists
            results_per_cv[method_name]['Accuracy'].append(Acc)
            results_per_cv[method_name]['Precision'].append(precision)
            results_per_cv[method_name]['Recall'].append(recall)
            results_per_cv[method_name]['F1'].append(f1)
            results_per_cv[method_name]['MCC'].append(MCC)
            results_per_cv[method_name]['Sensitivity'].append(Sensitivity)
            results_per_cv[method_name]['Specificity'].append(Specificity)
            results_per_cv[method_name]['G_measure'].append(G_measure)
            results_per_cv[method_name]['Jaccard'].append(Jaccard)
            results_per_cv[method_name]['AUC'].append(Auc)

            # Append results for ensemble model to lists
            results_per_cv[method_name]['Accuracy_e'].append(Acc_e)
            results_per_cv[method_name]['Precision_e'].append(precision_e)
            results_per_cv[method_name]['Recall_e'].append(recall_e)
            results_per_cv[method_name]['F1_e'].append(f1_e)
            results_per_cv[method_name]['MCC_e'].append(MCC_e)
            results_per_cv[method_name]['Sensitivity_e'].append(Sensitivity_e)
            results_per_cv[method_name]['Specificity_e'].append(Specificity_e)
            results_per_cv[method_name]['G_measure_e'].append(G_measure_e)
            results_per_cv[method_name]['Jaccard_e'].append(Jaccard_e)
            results_per_cv[method_name]['AUC_e'].append(Auc_e)

            # Append results for meta model to lists
            results_per_cv[method_name]['Accuracy_m'].append(Acc_m)
            results_per_cv[method_name]['Precision_m'].append(precision_m)
            results_per_cv[method_name]['Recall_m'].append(recall_m)
            results_per_cv[method_name]['F1_m'].append(f1_m)
            results_per_cv[method_name]['MCC_m'].append(MCC_m)
            results_per_cv[method_name]['Sensitivity_m'].append(Sensitivity_m)
            results_per_cv[method_name]['Specificity_m'].append(Specificity_m)
            results_per_cv[method_name]['G_measure_m'].append(G_measure_m)
            results_per_cv[method_name]['Jaccard_m'].append(Jaccard_m)
            results_per_cv[method_name]['AUC_m'].append(Auc_m)

    return results_per_cv

def results_to_dataframe(results):
    rows = []
    for method_name, metrics in results.items():
        for metric_name, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            rows.append({
                'Method': method_name,
                'Metric': metric_name,
                'Mean': mean_value,
                'Std': std_value,
                'Values': values
            })
    df = pd.DataFrame(rows)

    return df


def predict():
    # Select the sampling method
    methods = [{'name':'Origin'}, {'name':'Oversampling_SMOTE'}, {'name':'Undersampling'}, {'name':'Ransmote'}]
    datasets = [
        {'group': 'cp', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv', 'label_mappings': [{2: 0, 3: 0}]},
        {'group': 'am', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv', 'label_mappings': [{1: 5, 2: 1, 3: 1}]},
        {'group': 'el', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv', 'label_mappings': [{1: 5, 0: 6, 2: 0, 3: 1}]},
        {'group': 'cm', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv', 'label_mappings': [{0: 6, 2: 0, 3: 0}]},
        {'group': 'ae', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv', 'label_mappings': [{1: 5, 2: 1}]},
        {'group': 'al', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv', 'label_mappings': [{1: 5, 3: 1}]},
        {'group': 'ce', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv', 'label_mappings': [{0: 5, 2: 0}]},
        {'group': 'ca', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv', 'label_mappings': [{0: 0, 1: 1}]},
        {'group': 'cl', 'filename_e': './data/exp_exmethy_map_matrix.csv', 'filename_m': './data/methl_exmethy_map_matrix.csv','label_mappings': [{0:5, 3: 0, 1: 1}]}
    ]
    for dataset in datasets:
        print(dataset['group'])
        Xe, y1 = process_data_e(dataset['filename_e'], dataset['label_mappings'])
        Xm, y1 = process_data_m(dataset['filename_m'], dataset['label_mappings'])
        grid_models_e = get_method_grid(Xe, y1, methods)
        grid_models_m = get_method_grid(Xm, y1, methods)
        k_folds_pred = k_folds(Xe, Xm, y1, k=10, t=10, gride=grid_models_e, gridm = grid_models_m, methods=methods)
        method_results = results_to_dataframe(k_folds_pred)
        method_results.to_csv(f"{dataset['group']}_inte.csv", index=False)
        return "Prediction results have been saved to CSV files."
        print(method_results)


