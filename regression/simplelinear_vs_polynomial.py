# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

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
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Training the Linear Model
from sklearn.linear_model import LinearRegression
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

# Plotting the data
plt.scatter(x, y, color='red')

# Plotting the Linear Model
plt.plot(x, simple_linear_regressor.predict(x), color='blue')

# Plotting the Polynomial Model
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
x_grid_poly = polynominal_features.fit_transform(x_grid)
plt.plot(x_grid, polynomial_regressor.predict(x_grid_poly), color='green')
plt.xlabel("Year (1967-2003)")
plt.ylabel("Breeding Pairs")
plt.title("King Penguin Breeding Pairs")
plt.legend()
plt.show()

