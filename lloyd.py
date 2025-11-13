import pandas as pd
import numpy as np
from util import distance, noise, normalise

# k is the number of centroids, d is the number of dimensions. This uses no access to the data so is dp by default
def initialCentroids(k:int, d:int, minimum:int = -1, maximum:int= 1):
    return np.linspace([minimum]*d, [maximum]*d, num=k)

# to initialise the centroids we need a minimum and maximum for the data range: be careful of this affecting privacy
# k is the number of centers, X is the data
def initialCentroidsWithData(k: int, X: pd.DataFrame):
    minRange=X.min()
    maxRange=X.max()

    C = np.linspace(minRange, maxRange, num=k)

    return C

def initialize_spherical_clusters(k, d, radius=None, random_state=42):
    if radius is None:
        radius = d
    
    rng = np.random.default_rng(random_state)
    vecs = rng.normal(size=(k, d))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # normalize to unit sphere
    centers = vecs * radius

    return pd.DataFrame(centers)


def getClosestCenter(x, C):
    # x is the data point, C is all the centers. Return index of the closest center. 
    distances = np.apply_along_axis(lambda c: distance(x,c), axis=1, arr=C)
    return np.argmin(distances)


def lloyd(k: int, X: pd.DataFrame, n_iter: int):
    # initalise centers
    C = initialCentroids(k, X.shape[1])
    # repeat for n_iter for each cluster:
    for _ in range(0, n_iter):
        # assign each point to its closest center
        assignments = X.apply(lambda row: getClosestCenter(row, C), axis=1)
        # update center to be the average of all points assigned
        for i in range(0, len(C)):
            if (assignments == i).any():
                C.iloc[i, :] = X[assignments == i].mean()
    return C


def dplloyd(k: int, X: pd.DataFrame, n_iter: int, e: float, return_steps: bool = False):
    # initalise centers
    C = initialCentroids(k, X.shape[1])
    all_centers = []
    all_centers.append(C.copy())
    d = X.shape[1] # number of dimensions
    # repeat for n_iter for each cluster:
    for _ in range(0, n_iter):
        # assign each point to its closest center
        assignments = X.apply(lambda row: getClosestCenter(row, C), axis=1)
        # update center to be the average of all points assigned
        for i in range(0, len(C)):
            if (assignments == i).any():
                # noisily calculate the number of points in the cluster
                n = X[assignments == i].count() + noise(n_iter / e, 1) # DO I NEED TO SPLIT EPSILON HERE OVER THE TWO NOISY UPDATES?    
                # noisily calculate the sum of points in the cluster
                s = X[assignments == i].sum() + noise((d*n_iter) / e, d)
                # update centroid
                C.iloc[i, :] = s / n
        all_centers.append(C.copy())
    if return_steps:
        return all_centers
    return C

def lloyd_with_weights(k: int, X: pd.DataFrame, weights: pd.DataFrame, n_iter: int):
    # initalise centers
    C = initialize_spherical_clusters(k, X.shape[1], radius=1)
    # repeat for n_iter for each cluster:
    for _ in range(0, n_iter):
        # assign each point to its closest center
        assignments = X.apply(lambda row: getClosestCenter(row, C), axis=1)
        # update center to be the average of all points assigned
        for i in range(0, len(C)):
            if (assignments == i).any():
                C.iloc[i, :] = (X[assignments == i].mul(weights[assignments == i], axis=0).sum()) / (weights[assignments==i].sum())
    return C


X1 = pd.DataFrame(np.random.multivariate_normal(mean=(5,10), cov=[[5,0],[0,5]], size=200))
X2 = pd.DataFrame(np.random.multivariate_normal(mean=(2,3), cov=[[5,0],[0,5]], size=150))
X = pd.concat([X1, X2])

# print(dplloyd(k=2, X=normalise(X), n_iter=5, e =1, return_steps=True))

