import numpy as np
import pandas as pd

time_per_iter = np.array(pd.read_csv('a.txt', header = None)).T[0]
covariance_size = covariance_size = np.arange(100, 1050, 50)


from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
def func(x, a, b, Offset): # Sigmoid A With Offset from zunzun.com
    return  a*x**2 + b*x + Offset

# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    val = func(covariance_size**2, *parameterTuple)
    return np.sum((time_per_iter - val) ** 2.0)


def generate_Initial_Parameters():
    # min and max used for bounds
    maxX = max(covariance_size**2)
    minX = min(covariance_size**2)
    maxY = max(time_per_iter)
    minY = min(time_per_iter)

    parameterBounds = []
    parameterBounds.append([minX, maxX]) # search bounds for a
    parameterBounds.append([minX, maxX]) # search bounds for b
    parameterBounds.append([0.0, maxY]) # search bounds for Offset

    # "seed" the np.random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

# generate initial parameter values
geneticParameters = generate_Initial_Parameters()

# curve fit the test data
fittedParameters, pcov = curve_fit(func, covariance_size**2, time_per_iter, geneticParameters)

print('Parameters', fittedParameters)

modelPredictions = func(covariance_size**2, *fittedParameters)

absError = modelPredictions - time_per_iter

SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(time_per_iter))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)

import matplotlib.pyplot as plt
plt.figure(figsize = (8, 6))
plt.scatter(covariance_size**2, time_per_iter)
plt.yscale('log')
plt.xscale('log')
plt.xticks(
    [1e4, 1e5, 2e5, 5e5, 1e6],
    ["1e4", "1e5", "2e5", "5e5", "1e6"],
    fontsize = 16)

plt.yticks(
    [1e-3, 1e-2, 1e-1, 1e-0, 1e1],
    ["1e-3", "1e-2", "1e-1", "1", "10"],
    fontsize = 16)
plt.ylim(1e-3, 100)

x_model = np.linspace(10000, 10110000, 100000)
y_model = func(x_model, *fittedParameters)
plt.plot(x_model, y_model)


