# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Import dataset
dataset = pd.read_csv("breast-cancer-wisconsin.data", header=None, na_values='?')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Fill missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x = imputer.fit_transform(x)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Training the model
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(x_train, y_train)
log_reg_score = log_reg_classifier.score(x_test, y_test)
print("Score: {:.2f}".format(log_reg_score*100))


