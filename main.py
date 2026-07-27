# main functions
from metrics import calculate_metrics
from joblib import Parallel, delayed
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from parallel import run_cv_repeat
from parallel_loo import run_single_loo_parallel
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def main(
    data,
    labels,
    output_filename,
    cv_method,
    n_splits,
    n_repeats,
    classifiers,
    meta_models_config=None,
):
    """Main workflow for nested cross-validation with parallel repetitions."""
    if meta_models_config is None:
        from model_config import meta_models as meta_models_config
    
    combined_results = []

    if cv_method == "LOO":
        all_results = run_single_loo_parallel(
            repeat_idx=0,
            data=data,
            labels=labels,
            cv_method=cv_method,
            n_splits=n_splits,
            classifiers=classifiers,
            meta_models=meta_models_config
    )
        combined_results = all_results

    else:
        all_results = Parallel(n_jobs=-1)(
            delayed(run_cv_repeat)(
                repeat_idx,
                data,
                labels,
                cv_method,
                n_splits,
                classifiers,
                meta_models_config,
            )
            for repeat_idx in range(n_repeats)
        )
        combined_results = [item for sublist in all_results for item in sublist]

    # combined_results = [item for sublist in all_results for item in sublist]
    output = [
        {
            "RunID": res["RunID"],
            "Combination": ",".join(res["Combination"]),
            "MetaModel": res["MetaModel"],
            "SampleRID": res["SampleRID"],
            "Prediction": res["Prediction"],
            "Prediction_f": res.get("Prediction_f", pd.NA)
        }
        for res in combined_results
    ]
    
    output_df = pd.DataFrame(output)
    labels_df = pd.DataFrame(labels).reset_index()
    merged_df = pd.merge(output_df, labels_df, left_on="SampleRID", right_on="RID", how="left")

    # Save single sample results to a CSV
    single_sample_filename = output_filename.replace(".csv", "_single_samples.csv")
    merged_df.to_csv(single_sample_filename, index=False)
    print(f"Single sample pred results saved to {single_sample_filename}")

    # Calculate evaluation metrics
    result = (
        merged_df.groupby(['RunID', 'Combination', 'MetaModel'], group_keys=False)
        .apply(calculate_metrics)
        .reset_index()
    )

    # Calculate mean and standard deviation evaluation metrics across all RunID
    mean_results = (
        result.groupby(['Combination', 'MetaModel'])[
            ['Accuracy', 'AUC', 'F1', 'Recall', 'Precision', 'Specificity', 
            'MCC', 'Accuracy_f', 'AUC_f', 'F1_f', 'Recall_f', 'Precision_f',
            'Specificity_f', 'MCC_f']
        ]
        .agg(['mean', 'std'])
    )

    # Flatten multi-level column names and add "avg_" and "std_" prefixes
    mean_results.columns = [f"{stat}_{metric}" for metric, stat in mean_results.columns]
    mean_results = mean_results.reset_index()

    # Rename columns for clarity
    mean_results = mean_results.rename(columns={
        'mean_Accuracy': 'avg_Accuracy',
        'std_Accuracy': 'std_Accuracy',
        'mean_AUC': 'avg_AUC',
        'std_AUC': 'std_AUC',
        'mean_F1': 'avg_F1',
        'std_F1': 'std_F1',
        'mean_Recall': 'avg_Recall',
        'std_Recall': 'std_Recall',
        'mean_Precision': 'avg_Precision',
        'std_Precision': 'std_Precision',
        'mean_Specificity': 'avg_Specificity',
        'std_Specificity': 'std_Specificity',
        'mean_MCC': 'avg_MCC',
        'std_MCC': 'std_MCC',
        'mean_Accuracy_f': 'avg_Accuracy_f',
        'std_Accuracy_f': 'std_Accuracy_f',
        'mean_AUC_f': 'avg_AUC_f',
        'std_AUC_f': 'std_AUC_f',
        'mean_F1_f': 'avg_F1_f',
        'std_F1_f': 'std_F1_f',
        'mean_Recall_f': 'avg_Recall_f',
        'std_Recall_f': 'std_Recall_f',
        'mean_Precision_f': 'avg_Precision_f',
        'std_Precision_f': 'std_Precision_f',
        'mean_Specificity_f': 'avg_Specificity_f',
        'std_Specificity_f': 'std_Specificity_f',
        'mean_MCC_f': 'avg_MCC_f',
        'std_MCC_f': 'std_MCC_f'
    })

    # Save the detailed evaluation metrics (per RunID) to a CSV
    evaluation_filename = output_filename.replace(".csv", "_evaluation_metrics.csv")
    result.to_csv(evaluation_filename, index=False)
    print(f"Evaluation metrics saved to {evaluation_filename}")

    # Save the mean evaluation metrics to a separate CSV
    mean_evaluation_filename = output_filename.replace(".csv", "_mean_evaluation_metrics.csv")
    mean_results.to_csv(mean_evaluation_filename, index=False)
    print(f"Mean evaluation metrics saved to {mean_evaluation_filename}")

    # Return both results for further use
    return merged_df, result, mean_results
