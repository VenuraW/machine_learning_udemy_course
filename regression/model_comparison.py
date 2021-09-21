# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Getting the data
file = open('king_penguin_population.txt')
x = []
y = []
for line in file:
    x.append(int(line.split()[0]))
    y.append(int(line.split()[1]))

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Training the Linear Model
simple_linear_regressor = LinearRegression()
simple_linear_regressor.fit(x_train, y_train)
linear_score = simple_linear_regressor.score(x_test, y_test)
print("Linear Score: {:.2f}".format(linear_score*100))

# Training the Polynomial Model
from sklearn.preprocessing import PolynomialFeatures
polynominal_features = PolynomialFeatures(degree=2)
x_poly = polynominal_features.fit_transform(x_train)
x_poly_test = polynominal_features.transform(x_test)

polynomial_regressor = LinearRegression()
polynomial_regressor.fit(x_poly, y_train)
poly_score = polynomial_regressor.score(x_poly_test, y_test)
print("Polynomial Score: {:.2f}".format(poly_score*100))

# Training the Support Vector Regression Model
# Feature Scaling
sc_x = StandardScaler()
x_train_scaled = sc_x.fit_transform(x_train)
x_test_scaled = sc_x.transform(x_test)

sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train)
y_test_scaled = sc_y.transform(y_test)

# Training the Model
svr_regressor = SVR()
svr_regressor.fit(x_train_scaled, y_train_scaled)
svr_score = svr_regressor.score(x_test_scaled, y_test_scaled)
print("SVR Score: {:.2f}".format(svr_score*100))

# Plotting the data
plt.scatter(x, y, color='red')

# Plotting the Linear Model
plt.plot(x, simple_linear_regressor.predict(x), color='blue', label='Linear')

# Plotting the Polynomial Model
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(-1, 1)
x_grid_poly = polynominal_features.fit_transform(x_grid)
plt.plot(x_grid, polynomial_regressor.predict(x_grid_poly), color='green', label='Polynomial')

# Plotting the SVR Model
plt.plot(x_grid, sc_y.inverse_transform(svr_regressor.predict(sc_x.transform(x_grid))), color='m', label='SVR')

# Plot Features
plt.xlabel("Year (1967-2003)")
plt.ylabel("Breeding Pairs")
plt.title("King Penguin Breeding Pairs")
plt.legend()
plt.show()

