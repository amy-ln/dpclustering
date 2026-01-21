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
    C = pd.DataFrame(initialCentroids(k, X.shape[1]))
    # repeat for n_iter for each cluster:
    for _ in range(0, n_iter):
        # assign each point to its closest center
        assignments = X.apply(lambda row: getClosestCenter(row, C), axis=1)
        # update center to be the average of all points assigned
        for i in range(0, len(C)):
            if (assignments == i).any():
                C.iloc[i, :] = X[assignments == i].mean()
    return C

class PrivacyBudget:

    def __init__(self, epsilon: float, delta: float = None, method: str = "uniform", total_iter: int = None):
        self.epsilon = epsilon
        self.delta = delta
        self.total_iter = total_iter
        self.method = method

    def uniform_epsilon(self, t: int) -> float:
        # get epsilon on iteration t when following uniform approach
        if self.total_iter:
            return self.epsilon/self.total_iter
        else:
            raise Exception("Haven't set the total number of iterations")
    
    def dichotomy_epsilon(self, t:int) -> float:
        return self.epsilon / 2**t 
    
    def series_sum_epsilon(self, t:int) -> float:
        return self.epsilon / ((t+2)*(t+1))
    
    def getEpsilon(self, t:int) -> float:
        match self.method:
            case "uniform":
                return self.uniform_epsilon(t)
            case "dichotomy":
                return self.dichotomy_epsilon(t)
            case "series sum":
                return self.series_sum_epsilon(t)
            case _:
                raise Exception(f"{self.method} is not a valid privacy budget allocation method")


# assume data is normalised to [-1,1]
def dplloyd(k: int, X: np.ndarray, n_iter: int, priv: PrivacyBudget, seed=42, return_steps: bool = False) -> np.ndarray:
    """

    Args:
        k (int): The number of clusters 
        X (np.ndarray): The dataset containing the points
        n_iter (int): Number of iterations to do
        priv (PrivacyBudget): An object describing the privacy budget
        return_steps (bool, optional): Option to return the centroids at each iteration for debugging. Defaults to False.

    Returns:
        np.ndarray: the cluster centers 
    """    
    # initalise centers
    C = initialize_spherical_clusters(k, X.shape[1], radius=1).to_numpy()
    all_centers = [C.copy()]
    d = X.shape[1] # number of dimensions
    # repeat for n_iter for each cluster:
    for j in range(0, n_iter):
        # assign each point to its closest center
        assignments = np.array([
            getClosestCenter(x, C) for x in X
        ])
        # update center to be the average of all points assigned
        for i in range(k):
            # get epsilon for this iteration
            e = priv.getEpsilon(i)
            mask = (assignments==i)
            if (assignments == i).any():
                X_i = X[mask]
                # noisily calculate number of points in cluster
                n = max((X_i.shape[0] + noise((2 * n_iter) / e, 1, seed)), 1e-6) # don't allow negative counts 
                # noisily calculate sum of points in cluster
                s = X_i.sum(axis=0) + noise((2 * d * n_iter) / e, d, seed)
                print(f"Center {i}, Iteration {j}, points assigned {X_i.shape[0]}, n {n}, s{s}")
                # update centroid
                C[i, :] = s / n
            else:
                rng = np.random.default_rng(seed)
                C[i] = rng.integers(low=0, high=1, size=(1,d))
        all_centers.append(C.copy())
    if return_steps:
        return all_centers
    return C

"""
def lloyd_with_weights(k: int, X: pd.DataFrame, weights: pd.DataFrame, n_iter: int, rs=42):
    # initalise centers
    C = initialize_spherical_clusters(k, X.shape[1], radius=1, random_state=rs)
    # repeat for n_iter for each cluster:
    for _ in range(0, n_iter):
        # assign each point to its closest center
        assignments = X.apply(lambda row: getClosestCenter(row, C), axis=1)
        # update center to be the average of all points assigned
        for i in range(0, len(C)):
            if (assignments == i).any():
                C.iloc[i, :] = (X[assignments == i].mul(weights[assignments == i], axis=0).sum()) / (weights[assignments==i].sum())
    return C
"""

def lloyd_with_weights(
    k: int,
    X: pd.DataFrame,
    weights: pd.Series,
    n_iter: int,
    rs: int = 42
):
    rng = np.random.default_rng(rs)

    # Ensure alignment
    weights = weights.loc[X.index]

    # Initialize centers
    C = initialize_spherical_clusters(
        k, X.shape[1], radius=1, random_state=rs
    )

    for _ in range(n_iter):

        # Assign each point to closest center
        assignments = X.apply(
            lambda row: getClosestCenter(row, C),
            axis=1
        )

        for i in range(k):
            mask = assignments == i

            # Case 1: empty cluster → reinitialize
            if not mask.any():
                idx = rng.choice(X.index)
                C.iloc[i, :] = X.loc[idx].values
                continue

            X_i = X.loc[mask]
            w_i = weights.loc[mask]
            w_sum = w_i.sum()

            # Case 2: zero total weight → reinitialize
            if w_sum == 0:
                idx = rng.choice(X.index)
                C.iloc[i, :] = X.loc[idx].values
                continue

            # Weighted mean update
            C.iloc[i, :] = X_i.mul(w_i, axis=0).sum() / w_sum

    return C

