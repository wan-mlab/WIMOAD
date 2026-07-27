"""Model and task configuration for WIMOAD experiments."""

from pathlib import Path

import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
TASK_CONFIG_PATH = CONFIG_DIR / "tasks.yaml"
RANDOM_STATE = 0


meta_models = {
    "Meta_RF": RandomForestClassifier(random_state=RANDOM_STATE),
    "Meta_Logistic": LogisticRegression(max_iter=10000, random_state=RANDOM_STATE),
    "Meta_LinearRegression": LinearRegression(),
    "Meta_MLP1": MLPClassifier(
        max_iter=10000, hidden_layer_sizes=(50,), random_state=RANDOM_STATE
    ),
    "Meta_MLP2": MLPClassifier(
        max_iter=10000, hidden_layer_sizes=(50, 50), random_state=RANDOM_STATE
    ),
    "Meta_MLP3": MLPClassifier(
        max_iter=10000, hidden_layer_sizes=(50, 50, 50), random_state=RANDOM_STATE
    ),
    "Meta_MLP4": MLPClassifier(
        max_iter=10000,
        hidden_layer_sizes=(50, 50, 50, 50),
        random_state=RANDOM_STATE,
    ),
}


def _with_random_state(params):
    configured = dict(params)
    configured.setdefault("random_state", RANDOM_STATE)
    return configured


def _with_max_iter(params):
    configured = dict(params)
    configured.setdefault("max_iter", 10000)
    return configured


def _classifier(model_name, params):
    params = dict(params or {})
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

    if model_name == "RandomForest":
        return RandomForestClassifier(**_with_random_state(params))
    if model_name == "SVM":
        params.setdefault("probability", True)
        return SVC(**_with_random_state(params))
    if model_name == "LogisticRegression":
        return LogisticRegression(**_with_random_state(_with_max_iter(params)))
    if model_name == "MLP":
        return MLPClassifier(**_with_random_state(_with_max_iter(params)))
    if model_name == "KNN":
        return KNeighborsClassifier(**params)
    if model_name == "NaiveBayes":
        return GaussianNB(**params)
    raise ValueError(f"Unsupported classifier: {model_name}")


def _build_classifiers(model_configs):
    return {
        model_name: _classifier(model_name, params)
        for model_name, params in model_configs.items()
    }


def load_task_config(config_path=TASK_CONFIG_PATH):
    """Load raw task configuration from YAML."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_tasks(config_path=TASK_CONFIG_PATH):
    """Build sklearn-ready task dictionaries from YAML configuration."""

    config = load_task_config(config_path)
    defaults = config["defaults"]
    tasks = {}
    for group in config["task_order"]:
        task_config = config["tasks"][group]
        tasks[group] = {
            "group": group,
            "file_e": task_config.get("expression_file", defaults["expression_file"]),
            "file_m": task_config.get("methylation_file", defaults["methylation_file"]),
            "label_map": task_config["label_map"],
            "classifiers_e": _build_classifiers(task_config["expression"]),
            "classifiers_m": _build_classifiers(task_config["methylation"]),
        }
    return config["task_order"], tasks


TASK_ORDER, TASKS = build_tasks()
datasets = [TASKS[group] for group in TASK_ORDER]


def get_datasets(group="ca"):
    """Return configured task dictionaries for one group or all groups."""

    if group == "all":
        return list(datasets)
    if group not in TASKS:
        available = ", ".join(TASK_ORDER)
        raise ValueError(f"Unknown group '{group}'. Available groups: {available}")
    return [TASKS[group]]
