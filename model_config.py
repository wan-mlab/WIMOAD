# model config - datasets

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

meta_models = {
    "Meta_RF": RandomForestClassifier(),
    "Meta_Logistic": LogisticRegression(max_iter=10000),
    "Meta_LinearRegression": LinearRegression(),
    "Meta_MLP1": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50)),
    "Meta_MLP2": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50)),
    "Meta_MLP3": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50, 50)),
    "Meta_MLP4": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50, 50, 50)),
}

datasets = [
    # ca
    {'group': 'ca', 'file_e': 'exp_exmethy_map_matrix.csv','file_m': 'methl_exmethy_map_matrix.csv',
     'label_map': [{0: 0, 1: 1}],
     'classifiers_e': {
        "RandomForest": RandomForestClassifier(max_depth=10, n_estimators=50),
        "SVM": SVC(random_state=0, probability=True, C=1000, kernel='rbf'),
        "LogisticRegression": LogisticRegression(random_state=0, max_iter=1000, C=1),
        "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes=(50, 50), alpha=0.0001),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "NaiveBayes": GaussianNB()
     },
     'classifiers_m': {
        "RandomForest": RandomForestClassifier(max_depth = 10, n_estimators = 10),
        "SVM": SVC(random_state=0, probability=True, C=10, kernel='poly'),
        "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=1000),
        "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50, 50), alpha = 0.0001),
        "KNN": KNeighborsClassifier(n_neighbors = 5),
        "NaiveBayes": GaussianNB(),
    }},
   #  #al
   #  {'group': 'al', 'file_e': 'exp_exmethy_map_matrix.csv', 'file_m': 'methl_exmethy_map_matrix.csv',
   #   'label_map': [{1: 5, 3: 1}],
   #   'classifiers_e': {
   #      "RandomForest": RandomForestClassifier(max_depth = None, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=10, kernel='poly'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=1000),
   #      "MLP": MLPClassifier(max_iter=1000, hidden_layer_sizes = (50,), alpha = 0.0001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 3),
   #      "NaiveBayes": GaussianNB(),
   #   },
   #   'classifiers_m': {
   #      "RandomForest": RandomForestClassifier(max_depth = 20, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=0.1, kernel='linear'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=0.01),
   #      "MLP": MLPClassifier(max_iter=1000, hidden_layer_sizes = (50,50,50), alpha = 0.0001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 7),
   #      "NaiveBayes": GaussianNB(),
   #  }},
   # #  #ce
   #  {'group': 'ce', 'file_e': 'exp_exmethy_map_matrix.csv', 'file_m': 'methl_exmethy_map_matrix.csv',
   #   'label_map': [{0: 5, 2: 0}],
   #   'classifiers_e': {
   #      "RandomForest": RandomForestClassifier(max_depth = 20, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=0.01, kernel='linear'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=1000),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50,), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 3),
   #      "NaiveBayes": GaussianNB(),
   #   },
   #   'classifiers_m': {
   #      "RandomForest": RandomForestClassifier(max_depth = 10, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=1, kernel='linear'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=1000),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50,50,50), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 3),
   #      "NaiveBayes": GaussianNB(),
   #  }},
   #  #cl
   #  {'group': 'cl', 'file_e': 'exp_exmethy_map_matrix.csv', 'file_m': 'methl_exmethy_map_matrix.csv',
   #   'label_map': [{0:5, 3: 0, 1: 1}],
   #   'classifiers_e': {
   #     "RandomForest": RandomForestClassifier(max_depth = 10, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=100, kernel='rbf'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=100),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50), alpha = 0.0001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 7),
   #      "NaiveBayes": GaussianNB(),
   #   },
   #   'classifiers_m': {
   #      "RandomForest": RandomForestClassifier(max_depth = 20, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=0.01, kernel='rbf'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=0.01),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (100,), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 7),
   #      "NaiveBayes": GaussianNB(),
   #  }},
   #  #cm
   #  {'group': 'cm', 'file_e': 'exp_exmethy_map_matrix.csv', 'file_m': 'methl_exmethy_map_matrix.csv', 
   #   'label_map': [{0: 6, 2: 0, 3: 0}],
   #   'classifiers_e': {
   #      "RandomForest": RandomForestClassifier(max_depth = 10, n_estimators = 50),
   #      "SVM": SVC(random_state=0, probability=True, C=10, kernel='poly'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=1000),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (100,), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 5),
   #      "NaiveBayes": GaussianNB(),
   #   },
   #   'classifiers_m': {
   #      "RandomForest": RandomForestClassifier(max_depth = 20, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=1, kernel='linear'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=1000),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (100,), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 5),
   #      "NaiveBayes": GaussianNB(),
   #  }},
   #  #el
   #  {'group': 'el', 'file_e': 'exp_exmethy_map_matrix.csv', 'file_m': 'methl_exmethy_map_matrix.csv', 
   #   'label_map': [{1: 5, 0: 6, 2: 0, 3: 1}],
   #   'classifiers_e': {
   #      "RandomForest": RandomForestClassifier(max_depth = 10, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=10, kernel='poly'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=1000),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50,), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 3),
   #      "NaiveBayes": GaussianNB(),
   #   },
   #   'classifiers_m': {
   #      "RandomForest": RandomForestClassifier(max_depth = 20, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=0.1, kernel='rbf'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=0.01),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50,50,50), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 5),
   #      "NaiveBayes": GaussianNB(),
   #  }},
   #  #am
   #  {'group': 'am', 'file_e': 'exp_exmethy_map_matrix.csv', 'file_m': 'methl_exmethy_map_matrix.csv', 
   #   'label_map': [{1: 5, 2: 1, 3: 1}],
   #   'classifiers_e': {
   #      "RandomForest": RandomForestClassifier(max_depth = None, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=0.01, kernel='poly'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=0.01),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50, 50), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 5),
   #      "NaiveBayes": GaussianNB(),
   #   },
   #   'classifiers_m': {
   #      "RandomForest": RandomForestClassifier(max_depth = None, n_estimators = 10),
   #      "SVM": SVC(random_state=0, probability=True, C=100, kernel='rbf'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=1000),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50,), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 5),
   #      "NaiveBayes": GaussianNB(),
   #  }},
   #  #cp
   #  {'group': 'cp', 'file_e': 'exp_exmethy_map_matrix.csv', 'file_m': 'methl_exmethy_map_matrix.csv', 
   #   'label_map': [{2: 0, 3: 0}],
   #   'classifiers_e': {
   #      "RandomForest": RandomForestClassifier(max_depth = 20, n_estimators = 100),
   #      "SVM": SVC(random_state=0, probability=True, C=0.01, kernel='linear'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=100),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 3),
   #      "NaiveBayes": GaussianNB(),
   #   },
   #   'classifiers_m': {
   #      "RandomForest": RandomForestClassifier(max_depth = 10, n_estimators = 50),
   #      "SVM": SVC(random_state=0, probability=True, C=0.01, kernel='poly'),
   #      "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=0.01),
   #      "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50, 50), alpha = 0.001),
   #      "KNN": KNeighborsClassifier(n_neighbors = 3),
   #      "NaiveBayes": GaussianNB(),
   #  }},
    # #ae
    # {'group': 'ae', 'file_e': 'exp_exmethy_map_matrix.csv', 'file_m': 'methl_exmethy_map_matrix.csv', 
    #  'label_map': [{1: 5, 2: 1}],
    #  'classifiers_e': {
    #     "RandomForest": RandomForestClassifier(max_depth = 10, n_estimators = 10),
    #     "SVM": SVC(random_state=0, probability=True, C=0.01, kernel='linear'),
    #     "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=0.1),
    #     "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50, 50), alpha = 0.001),
    #     "KNN": KNeighborsClassifier(n_neighbors = 5),
    #     "NaiveBayes": GaussianNB(),
    #  },
    #  'classifiers_m': {
    #     "RandomForest": RandomForestClassifier(max_depth = 10, n_estimators = 100),
    #     "SVM": SVC(random_state=0, probability=True, C=1, kernel='rbf'),
    #     "LogisticRegression": LogisticRegression(random_state=0, max_iter=10000, C=0.01),
    #     "MLP": MLPClassifier(max_iter=10000, hidden_layer_sizes = (50, 50, 50), alpha = 0.001),
    #     "KNN": KNeighborsClassifier(n_neighbors = 5),
    #     "NaiveBayes": GaussianNB(),
    # }}
]