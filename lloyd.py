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

