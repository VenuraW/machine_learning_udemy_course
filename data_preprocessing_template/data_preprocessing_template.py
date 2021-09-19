# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset from csv
dataset = pd.read_csv("test_file.csv")
x = dataset.iloc[:, :-1].values # Select ALL rows and columns up to the last
y = dataset.iloc[:, -1].values # Select ALL rows and LAST column

# Filling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:, 1:3] = imputer.fit_transform(x[:, 1:3]) # fit obtains the mean, transform applies the method

# Transform independent variables into dummy variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # Transform column 0 with the encoder and ignore the rest
x = ct.fit_transform(x)

# Transform dependent variables into binary variables (0 to n - 1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler();
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])


