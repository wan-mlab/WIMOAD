# runner

import time
from data_loader import load_data, load_data_m
from model_config import datasets
from main import main

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"

if __name__ == "__main__":
    start_time = time.time()

    for dataset in datasets:
        # for selected dataset
        print(f"Processing group: {dataset['group']}")
        data, labels = load_data(dataset['file_e'], dataset['label_map'])
        classifiers = dataset['classifiers_e']
        output_filename = f"results_{dataset['group']}.csv"

        # 设置cv_method和n_repeats的逻辑
        # cv_method = "KFold"
        cv_method = "LOO"
        n_repeats = 1 if cv_method == "LOO" else 10

        single_pred, evaluation, mean_evaluation = main(
            data, labels, output_filename,
            cv_method=cv_method, n_splits=5, n_repeats=n_repeats, classifiers=classifiers
        )
        # Uncomment the following for methylation group processing if needed
        # print(f"Processing methylation group: {dataset['group']}")
        # data_m, labels_m = load_data_m(dataset['file_m'], dataset['label_map'])
        # classifiers = dataset['classifiers_m']
        # output_filename_m = f"results_m_{dataset['group']}.csv"
        # single_pred, evaluation, mean_evaluation = main(
        #     data_m, labels_m, output_filename_m,
        #     cv_method=cv_method, n_splits=5, n_repeats=n_repeats, classifiers=classifiers
        # )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Running time: {format_time(elapsed_time)}")


# if __name__ == "__main__":
#     n_repeats = 1
#     start_time = time.time()

#     for dataset in datasets:
#         # for selected dataset
#         print(f"Processing group: {dataset['group']}")
#         data, labels = load_data(dataset['file_e'], dataset['label_map'])
#         classifiers = dataset['classifiers_e']
#         output_filename = f"results_{dataset['group']}.csv"
#         single_pred, evaluation, mean_evaluation = main(data, labels, output_filename, cv_method="KFold", n_splits=5, n_repeats=n_repeats, classifiers=classifiers)
#         # single_pred, evaluation, mean_evaluation = main(data, labels, output_filename, cv_method="LOO", n_splits=10, n_repeats=n_repeats, classifiers=classifiers)
#         # print(f"Processing methylation group: {dataset['group']}")
#         # data_m, labels_m = load_data_m(dataset['file_m'], dataset['label_map'])
#         # classifiers = dataset['classifiers_m']
#         # output_filename_m = f"results_m_{dataset['group']}.csv"
#         # single_pred, evaluation, mean_evaluation = main(data_m, labels_m, output_filename_m, cv_method="KFold", n_splits=5, n_repeats=n_repeats, classifiers=classifiers)

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Running time: {format_time(elapsed_time)}")
