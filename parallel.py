# parallel training
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.base import clone
from data_loader import feature_selection

FEATURE_K = 1000


def generate_combinations(classifiers):
    classifier_names = list(classifiers.keys())
    all_combinations = []
    for r in range(1, len(classifier_names) + 1):
        all_combinations.extend(itertools.combinations(classifier_names, r))
    return all_combinations

def train_classifier(clf_name, clf, X_train, y_train, X_test):
    model = clone(clf)
    model.fit(X_train, y_train)
    train_predictions = model.predict_proba(X_train)[:, 1]
    test_predictions = model.predict_proba(X_test)[:, 1]
    return clf_name, train_predictions, test_predictions

def parallel_train_inner_classifiers(classifiers, X_inner_train, y_inner_train, X_inner_val, oof_predictions, inner_val_idx):
    futures = []
    
    with ThreadPoolExecutor() as executor:
        for idx, (clf_name, clf) in enumerate(classifiers.items()):
            futures.append(
                executor.submit(train_classifier, clf_name, clf, X_inner_train, y_inner_train, X_inner_val)
            )

        for future in as_completed(futures):
            clf_name, train_predictions, test_predictions = future.result()
            idx = list(classifiers.keys()).index(clf_name)
            oof_predictions[inner_val_idx, idx] = test_predictions

def parallel_train_outer_classifiers(classifiers, X_train, y_train, X_test, base_predictions_out, test_idx):
    futures = []
    with ThreadPoolExecutor() as executor:
        for idx, (clf_name, clf) in enumerate(classifiers.items()):
            futures.append(
                executor.submit(train_classifier, clf_name, clf, X_train, y_train, X_test)
            )
        for future in as_completed(futures):
            clf_name, train_predictions, test_predictions = future.result()
            idx = list(classifiers.keys()).index(clf_name)
            base_predictions_out[test_idx, idx] = test_predictions

def parallel_train_meta_models(meta_models, meta_train_X, y_train, meta_test_X, X_train, X_test, test_idx, comb, meta_predictions, meta_predictions2, results, data, extended_meta_train_X, extended_meta_test_X, repeat_idx):
    """Parallel training for meta-models using ThreadPoolExecutor."""
    futures = []

    with ThreadPoolExecutor() as executor:
        # Submit tasks for training and predicting with each meta-model
        for idx, (meta_name, meta_model) in enumerate(meta_models.items()):
            futures.append(
                executor.submit(
                    train_meta_model_and_predict, idx, meta_name, meta_model, 
                    meta_train_X, y_train, meta_test_X, 
                    X_train, X_test, test_idx, 
                    meta_predictions, meta_predictions2, results, comb, data, extended_meta_train_X, extended_meta_test_X, repeat_idx
                )
            )
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()

def train_meta_model_and_predict(idx, meta_name, meta_model, 
                                  meta_train_X, y_train, meta_test_X, 
                                  X_train, X_test, test_idx, 
                                  meta_predictions, meta_predictions2, 
                                  results, comb, data, extended_meta_train_X, extended_meta_test_X, repeat_idx):
    """Train the meta-model and predict."""
    stacked_model = clone(meta_model)
    stacked_model.fit(meta_train_X, y_train)

    # Get predictions from meta-model
    if hasattr(stacked_model, 'predict_proba'):
        meta_predictions[test_idx, idx] = stacked_model.predict_proba(meta_test_X)[:, 1]
    else:
        meta_predictions[test_idx, idx] = stacked_model.predict(meta_test_X)

    extended_model = clone(meta_model)
    extended_model.fit(extended_meta_train_X, y_train)
    if hasattr(extended_model, 'predict_proba'):
        meta_predictions2[test_idx, idx] = extended_model.predict_proba(extended_meta_test_X)[:, 1]
    else:
        meta_predictions2[test_idx, idx] = extended_model.predict(extended_meta_test_X)

    # Store the results for each test sample
    for i in test_idx:
        results.append({
            "RunID": repeat_idx + 1,
            "Combination": comb,
            "MetaModel": meta_name,
            "SampleRID": data.index[i],
            "Prediction": meta_predictions[i, idx],
            "Prediction_f": meta_predictions2[i, idx]
        })

def run_cv_repeat(repeat_idx, data, labels, cv_method, n_splits, classifiers, meta_models):
    print(f"Starting CV Repeat {repeat_idx + 1}")
    outer_cv = LeaveOneOut() if cv_method == "LOO" else KFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat_idx)
    
    combinations = generate_combinations(classifiers)
    results = []
    base_predictions_out = np.zeros((len(labels), len(classifiers)))
    meta_predictions = np.zeros((len(labels), len(meta_models)))
    meta_predictions2 = np.zeros((len(labels), len(meta_models)))

    for train_idx, test_idx in outer_cv.split(data, labels):
        print(f"Repeat {repeat_idx + 1}, Test: {test_idx}")
        X_train_o, X_test_o = data.iloc[train_idx], data.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        X_train, X_test = feature_selection(
            X_train_o, y_train, X_test_o, k=FEATURE_K
        )

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42 + repeat_idx)
        oof_predictions = np.zeros((len(y_train), len(classifiers)))

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_o):
            X_inner_train_o = X_train_o.iloc[inner_train_idx]
            X_inner_val_o = X_train_o.iloc[inner_val_idx]
            y_inner_train, y_inner_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
            X_inner_train, X_inner_val = feature_selection(
                X_inner_train_o, y_inner_train, X_inner_val_o, k=FEATURE_K
            )
            parallel_train_inner_classifiers(classifiers, X_inner_train, y_inner_train, X_inner_val, oof_predictions, inner_val_idx)

        parallel_train_outer_classifiers(classifiers, X_train, y_train, X_test, base_predictions_out, test_idx)

        for comb in combinations:
            comb_indices = [list(classifiers.keys()).index(name) for name in comb]

            if len(comb) == 1:
                base_model_name = f"BaseModel_{comb[0]}"
                for i in test_idx:
                    results.append({
                        "RunID": repeat_idx + 1,
                        "Combination": comb,
                        "MetaModel": base_model_name,
                        "SampleRID": data.index[i],
                        "Prediction": base_predictions_out[i, comb_indices[0]]
                    })
            else:
                meta_train_X = oof_predictions[:, comb_indices]
                meta_test_X = base_predictions_out[test_idx][:, comb_indices]
                extended_meta_train_X = np.hstack([X_train.values, meta_train_X])
                extended_meta_test_X = np.hstack([X_test.values, meta_test_X])

                parallel_train_meta_models(
                    meta_models, meta_train_X, y_train, meta_test_X, 
                    X_train, X_test, test_idx, comb, 
                    meta_predictions, meta_predictions2, results, data, extended_meta_train_X, extended_meta_test_X, repeat_idx
                )
    return results
