"""Command-line runner for WIMOAD stacking experiments."""

import argparse
from pathlib import Path
import time


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def parse_args(argv=None):
    from model_config import TASK_ORDER

    parser = argparse.ArgumentParser(
        description="Run WIMOAD stacking for one or more configured task groups."
    )
    parser.add_argument(
        "--group",
        default="ca",
        choices=["all", *TASK_ORDER],
        help="Task group to run. Use 'all' for every configured group.",
    )
    parser.add_argument(
        "--omics",
        default="both",
        choices=["expression", "methylation", "both"],
        help="Omics branch to run.",
    )
    parser.add_argument(
        "--cv",
        default="LOO",
        choices=["LOO", "KFold"],
        help="Cross-validation strategy.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where result CSV files will be written.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of splits for KFold runs.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeated KFold runs. Defaults to 1 for LOO and 10 for KFold.",
    )
    return parser.parse_args(argv)


def run_expression(dataset, meta_models, cv_method, n_splits, n_repeats, output_dir):
    from data_loader import load_data
    from main import main

    print(f"Processing expression group: {dataset['group']}")
    data, labels = load_data(dataset['file_e'], dataset['label_map'])
    output_filename = output_dir / f"results_{dataset['group']}.csv"
    return main(
        data,
        labels,
        str(output_filename),
        cv_method=cv_method,
        n_splits=n_splits,
        n_repeats=n_repeats,
        classifiers=dataset['classifiers_e'],
        meta_models_config=meta_models,
    )


def run_methylation(dataset, meta_models, cv_method, n_splits, n_repeats, output_dir):
    from data_loader import load_data_m
    from main import main

    print(f"Processing methylation group: {dataset['group']}")
    data_m, labels_m = load_data_m(dataset['file_m'], dataset['label_map'])
    output_filename_m = output_dir / f"results_m_{dataset['group']}.csv"
    return main(
        data_m,
        labels_m,
        str(output_filename_m),
        cv_method=cv_method,
        n_splits=n_splits,
        n_repeats=n_repeats,
        classifiers=dataset['classifiers_m'],
        meta_models_config=meta_models,
    )


def run(args):
    from model_config import get_datasets, meta_models

    start_time = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_repeats = args.n_repeats if args.n_repeats is not None else (1 if args.cv == "LOO" else 10)

    for dataset in get_datasets(args.group):
        if args.omics in {"expression", "both"}:
            run_expression(
                dataset, meta_models, args.cv, args.n_splits, n_repeats, output_dir
            )

        if args.omics in {"methylation", "both"}:
            run_methylation(
                dataset, meta_models, args.cv, args.n_splits, n_repeats, output_dir
            )

    elapsed_time = time.time() - start_time
    print(f"Running time: {format_time(elapsed_time)}")


if __name__ == "__main__":
    run(parse_args())
