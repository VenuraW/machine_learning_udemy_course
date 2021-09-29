import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Import dataset
dataset = pd.read_csv("breast-cancer-wisconsin.data", header=None, na_values='?')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Fill missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x = imputer.fit_transform(x)

# Splitting the data
split_set = train_test_split(x, y, test_size=0.25)