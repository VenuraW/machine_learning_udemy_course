# Import the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding dummy variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = ct.fit_transform(x)

# Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
score = regressor.score(x_train, y_train)

# Predicting a new value
y_pred = regressor.predict([[0, 1, 0, 130000, 140000, 300000]])

# Backward Elimination
# Remove a dummy variable column and adding the constant column
x = x[:, 1:]
x = np.append(np.ones((len(x), 1), dtype=int), x, axis=1)

print(x)

import statsmodels.api as sm
x_regressor = np.array(x, dtype=float)
ordinary_least_squares_regressor = sm.OLS(endog= y, exog= x_regressor).fit()

# Removing New York
x_regressor = np.array(x[:, [0, 1, 3, 4, 5]], dtype=float)
ordinary_least_squares_regressor = sm.OLS(endog= y, exog= x_regressor).fit()

# Removing Florida
x_regressor = np.array(x[:, [0, 3, 4, 5]], dtype=float)
ordinary_least_squares_regressor = sm.OLS(endog= y, exog= x_regressor).fit()

# Removing Administration
x_regressor = np.array(x[:, [0, 3, 5]], dtype=float)
ordinary_least_squares_regressor = sm.OLS(endog= y, exog= x_regressor).fit()

# Removing Marketing Spend
x_regressor = np.array(x[:, [0, 3]], dtype=float)
ordinary_least_squares_regressor = sm.OLS(endog= y, exog= x_regressor).fit()
print(ordinary_least_squares_regressor.summary())


