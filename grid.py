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
def create_grid_synopsis(X: pd.DataFrame, e: float, d: int, M: Optional[float] = None) -> pd.DataFrame:

    # a non-private approximation for M. Ideally M would be given. 
    if not M:
        M = round((X.shape[0]*e) / 10)
        print(M)

    # create the grid 
    edges = np.linspace(-1, 1, M + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    grid = pd.DataFrame(np.array(list(itertools.product(centers, repeat=d))))

    # assign each point to a square in the grid 
    squares = X.apply(lambda row: getSquare(row, grid), axis=1)

    # get counts of points in the grid, used for the weights 
    points = squares.value_counts().reset_index(name="count")
    all_points = pd.merge(points, grid, on = list(grid.columns),  how="outer").fillna(0)
    # here we would add laplace noise to the counts, since the l1 norm of the counts vector is 1 we want to add a vector sampled from lap(1/e)
    all_points["count"] += noise(1/e, grid.shape[0])
    return all_points



X1 = pd.DataFrame(np.random.multivariate_normal(mean=(5,10), cov=[[5,0],[0,5]], size=200))
X2 = pd.DataFrame(np.random.multivariate_normal(mean=(2,3), cov=[[5,0],[0,5]], size=150))
X = normalise(pd.concat([X1, X2]))
p = create_grid_synopsis(X, 1, 2)
plt.scatter(x=p.iloc[:, 0], y=p.iloc[:, 1], s=p.iloc[:, 2])
plt.show()
print(p)
print(lloyd_with_weights(2, p.iloc[:, :-1], p.iloc[:, -1], n_iter=5))
