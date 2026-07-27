from model_config import TASK_ORDER, TASKS, datasets, get_datasets


def test_all_task_groups_are_registered():
    assert TASK_ORDER == ["ae", "al", "am", "ca", "ce", "cl", "cm", "cp", "el"]
    assert list(TASKS) == TASK_ORDER
    assert [dataset["group"] for dataset in datasets] == TASK_ORDER


def test_get_datasets_selects_single_group_or_all():
    assert [dataset["group"] for dataset in get_datasets("ca")] == ["ca"]
    assert [dataset["group"] for dataset in get_datasets("all")] == TASK_ORDER


def test_yaml_task_config_builds_expected_estimators():
    ca = TASKS["ca"]

    assert ca["file_e"] == "exp_exmethy_map_matrix.csv"
    assert ca["file_m"] == "methl_exmethy_map_matrix.csv"
    assert ca["label_map"] == [{0: 0, 1: 1}]
    assert ca["classifiers_e"]["RandomForest"].n_estimators == 50
    assert ca["classifiers_e"]["RandomForest"].random_state == 0
    assert ca["classifiers_e"]["SVM"].probability is True
    assert ca["classifiers_e"]["MLP"].hidden_layer_sizes == (50, 50)
