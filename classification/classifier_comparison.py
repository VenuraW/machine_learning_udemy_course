# Importing the libraries
from preprocessing import x_train_scaled, x_test_scaled, y_train, y_test
from classifier import Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = []

# Adding Logistic Regression Model
models.append({"name": "Logistic Regression",
               "classifier": LogisticRegression(),
               "params": [{'penalty': ['l2', 'none'], 'solver':['newton-cg', 'lbfgs']},
                          {'penalty': ['l1', 'l2'], 'solver': ['liblinear']}]})

# Adding K-Nearest Neighbours Model
models.append({"name": "K-Nearest Neighbours",
               "classifier": KNeighborsClassifier(),
               "params": [{'weights': ['uniform', 'distance'],
                           'algorithm': ['ball_tree', 'kd_tree'],
                           'n_neighbors': [x for x in range(1, 10)]}]})

# Adding SVM model
models.append({"name": "Support Vector Machine",
               "classifier": SVC(),
               "params": [{'kernel': ['linear', 'sigmoid']},
                          {'kernel': ['poly'],
                           'degree': [x for x in range(1, 5)]}]})

# Adding Naive Bayes
models.append({"name": "Naive Bayes",
               "classifier": GaussianNB(),
               "params": [{'var_smoothing': [x/1e-9 for x in range(1, 10)]}]})

# Adding Decision Tree Classifier
models.append({"name": "Decision Tree Classifier",
               "classifier": DecisionTreeClassifier(),
               "params" : [{'criterion': ['gini', 'entropy'],
                            'min_samples_split': [x/10 for x in range(1, 5)],
                            'min_samples_leaf': [x for x in range(1, 5)],
                            'max_features': ['auto', 'sqrt', 'log2']}]})

# Adding Random Forest Classifier
models.append({"name": "Random Forest Classifier",
               "classifier": RandomForestClassifier(),
               "params": [{'criterion': ['gini', 'entropy'],
                           'min_samples_split': [x/10 for x in range(1, 5)],
                           'min_samples_leaf': [x for x in range(1, 5)],
                           'max_features': ['auto', 'sqrt', 'log2']}]})

sorted_classifiers = [0] * len(models)

# Training models
for model in models:
    classifier = Classifier(model["name"], model["classifier"], x_train_scaled, x_test_scaled, y_train, y_test)

    # Training default model and using grid search to find best classifier
    classifier.train_classifier()
    classifier.train_grid_search(model["params"])
    classifier.set_best_classifer()

    # Find position in array to add
    insert_index = 0
    while sorted_classifiers[insert_index] != 0:
        if sorted_classifiers[insert_index].get_best_score() < classifier.get_best_score():
            break
        insert_index += 1

    sorted_classifiers[insert_index:insert_index] = [classifier]

# Printing models by name and score
for i in range(len(models)):
    print("{}. {}".format(i + 1, sorted_classifiers[i]))