import pandas as pd
import numpy as np
import itertools
from typing import Optional
from util import distance,  noise

def getSquare(x: np.array, grid: pd.DataFrame):
    distances = grid.apply(lambda row: distance(x, row), axis=1)
    return grid.iloc[distances.idxmin(),:]


# M is the number of squares to split grid into, d is the dimension, e is epsilon
# assume the data is normalised so each dimension is in [-1,1]
def create_grid_synopsis(X: np.ndarray, e: float, d: int, M: Optional[float] = None, seed=42) -> np.ndarray:
    """ Create the private synopsis using the Grid algorithm. 

    Args:
        X (np.ndarray): The non private data which should be centred and each column should take values in [-1,1]
        e (float): The privacy budget epsilon. 
        d (int): The dimension of the data X
        M (Optional[float], optional): The resolution of the grid. Uses a non-private heuristic if not given. 
        seed (int, optional): Randomness seed for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: _description_
    """    

    # a non-private approximation for M using heuristic described in final report. Ideally M would be given. 
    if not M:
        M = round((X.shape[0]*e) / 10)
        print(M)

    # create the grid 
    edges = np.linspace(-1, 1, M + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    grid = np.array(list(itertools.product(centers, repeat=d)))

    # assign each point to a square in the grid 
    diffs = X[:, None, :] - grid[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    assigned = np.argmin(dists, axis=1)

    # get counts of points in the grid, used for the weights 
    counts = np.bincount(assigned, minlength=grid.shape[0]).astype(float)

    # here we would add laplace noise to the counts, since the l1 norm of the counts vector is 1 we want to add a vector sampled from lap(1/e)
    counts += noise(1 / e, grid.shape[0], seed)

    # combine the grid coordinate with the counts 
    synopsis = np.hstack([grid, counts[:, None]])

    return synopsis

# exactly the same as the method above but the calculations are done in chunks due to reduce memory usage
def create_grid_synopsis_large(X: np.ndarray, e: float, d: int, M: int, seed=42) -> np.ndarray:
    # create the grid 
    edges = np.linspace(-1, 1, M + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    grid = np.array(list(itertools.product(centers, repeat=d)))

    # assign each point to a square in the grid 
    chunk_size = 1000
    assigned = np.empty(len(X), dtype=int)
    for i in range(0, len(X), chunk_size):
        diffs = X[i:i+chunk_size, None, :] - grid[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        assigned[i:i+chunk_size] = np.argmin(dists, axis=1)

    # get counts of points in the grid, used for the weights 
    counts = np.bincount(assigned, minlength=grid.shape[0]).astype(float)

    # here we would add laplace noise to the counts, since the l1 norm of the counts vector is 1 we want to add a vector sampled from lap(1/e)
    counts += noise(1 / e, grid.shape[0], seed)

    # combine the grid coordinate with the counts 
    synopsis = np.hstack([grid, counts[:, None]])

    return synopsis