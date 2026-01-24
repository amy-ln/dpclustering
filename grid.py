import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from typing import Optional
from util import distance, normalise, noise
from lloyd import lloyd_with_weights

def getSquare(x: np.array, grid: pd.DataFrame):
    distances = grid.apply(lambda row: distance(x, row), axis=1)
    return grid.iloc[distances.idxmin(),:]


# M is the number of squares to split grid into, d is the dimension, e is epsilon
# assume the data is normalised so each dimension is in [-1,1]
def create_grid_synopsis(X: np.ndarray, e: float, d: int, M: Optional[float] = None) -> np.ndarray:

    # a non-private approximation for M. Ideally M would be given. 
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
    counts += noise(1 / e, grid.shape[0])

    # combine the grid coordinate with the counts 
    synopsis = np.hstack([grid, counts[:, None]])

    return synopsis


rng = np.random.default_rng(42)
X1 = rng.multivariate_normal(mean=(0.5,0.5), cov=[[1,0],[0,1]], size=200)
X2 = rng.multivariate_normal(mean=(-0.5,-0.5), cov=[[1,0],[0,1]], size=150)
X = np.concat([X1, X2])
p = pd.DataFrame(create_grid_synopsis(X, 1, 2, M=3))
plt.scatter(x=p.iloc[:, 0], y=p.iloc[:, 1], s=p.iloc[:, 2])
plt.show()
print(p)
print(lloyd_with_weights(2, p.iloc[:, :-1], p.iloc[:, -1], n_iter=5))