# Importing the libraries
from sklearn.linear_model import LogisticRegression
import util
from preprocessing import split_set

# Training the model
log_reg_classifier = util.get_trained_classifier("Logistic Regression", LogisticRegression(), split_set)

# Hyperparamter Tuning
parameters = [{'penalty': ['l2', 'none'], 'solver':['newton-cg', 'lbfgs']},
              {'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
              {'penalty': ['l2'], 'solver': ['lbfgs']}]
grid_search = util.get_grid_search(parameters, LogisticRegression(), split_set)
