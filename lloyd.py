import pandas as pd
import numpy as np
import typing

# Euclidean distance between x and y is sum of squared differences.
def distance(x, y):
    return np.sqrt(sum((x-y)**2))


# to initialise the centroids we need a minimum and maximum for the data range: be careful of this affecting privacy
# k is the number of centers, X is the data
def initialCentroids(k: int, X: pd.DataFrame):
    minRange=X.min()
    maxRange=X.max()

    C = np.linspace(minRange, maxRange, num=k)

    return C

def getClosestCenter(x, C):
    # x is the data point, C is all the centers. Return index of the closest center. 
    distances = np.apply_along_axis(lambda c: distance(x,c), axis=1, arr=C)
    return np.argmin(distances)

def noise(scale, d):
    rng = np.random.default_rng()
    return rng.laplace(0, scale, size=d)


def lloyd(k: int, X: pd.DataFrame, n_iter: int):
    # initalise centers
    C = initialCentroids(k, X)
    # repeat for n_iter for each cluster:
    for _ in range(0, n_iter):
        # assign each point to its closest center
        assignments = X.apply(lambda row: getClosestCenter(row, C), axis=1)
        # update center to be the average of all points assigned
        for i in range(0, len(C)):
            C[i] = X[assignments == i].mean()
    return C

def normalise(col):
    return (col - col.mean()) / col.std()

def dplloyd(k: int, X: pd.DataFrame, n_iter: int, e: float):
    # initalise centers
    C = initialCentroids(k, X)
    d = X.shape[1] # number of dimensions
    # repeat for n_iter for each cluster:
    for _ in range(0, n_iter):
        # assign each point to its closest center
        assignments = X.apply(lambda row: getClosestCenter(row, C), axis=1)
        # update center to be the average of all points assigned
        for i in range(0, len(C)):
            C[i] = X[assignments == i].mean() + noise(((d+1)*n_iter) / e, d) # add noise here 
    return C

X = pd.DataFrame(np.random.multivariate_normal(mean=(5,10), cov=[[5,0],[0,5]], size=50))
normalised = (X - X.min())/(X.max() - X.mean())
print(normalised)

dplloyd(3, normalised, 5, 0.1)