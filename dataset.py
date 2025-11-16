import numpy as np
import pandas as pd 
import itertools
from typing import Optional, List 
from util import getClosestCenter, initialCentroids, getSquare
from lloyd import lloyd_with_weights

""" 
Want to make a class which stores datasets and parameters required for all algorithms. 
Can then add methods to test each algorithm using the pre-defined parameters. 
"""

def scale(X: pd.Series, minimum: float, maximum: float):
    return 2 * ((X - minimum) / maximum - minimum) - 1

def non_private_scale(X: pd.Series):
    return 2 * ((X - X.min()) / X.max() - X.min()) - 1

class Dataset:
    def __init__(self, data: pd.DataFrame, epsilon: float = 1, dataset_bounds_min: Optional[List[float]] = None, dataset_bounds_max: Optional[List[float]] = None, dimension: Optional[int] = None, random_seed: int = 42):

        # ensure dimension given matches dataset
        if dimension is None:
            self.dimension = data.shape[1]
        elif dimension != data.shape[1]:
            raise Exception("Dimension given does not match the shape of the data")
        else:
            self.dimension = dimension
        
        self.data = data
        self.epsilon = epsilon
        self.random_seed = random_seed
        self.random_generator = np.random.default_rng(random_seed)

        # normalise data to be in [-1,1]
        if (dataset_bounds_min is None) or (dataset_bounds_max is None):
            print("NON PRIVATELY normalising data..")
            self.data = self.data.apply(non_private_scale, axis="columns")
        elif (len(dataset_bounds_min) != self.dimension) or (len(dataset_bounds_max) != self.dimension):
            raise Exception("Bounds given do not match the shape of the data")
        else:
            for col_index in range (0, self.dimension):
                self.data.iloc[:, col_index] = scale(self.data.iloc[:, col_index], dataset_bounds_min[col_index], dataset_bounds_max[col_index])

    def laplace_mechanism(self, scale:float, d: Optional[int] = None):
        if d:
            return self.random_generator.laplace(0, scale, size=d)
        else:
            return self.random_generator.laplace(0, scale, self.dimension)

    def non_private_lloyd(self, k: int):
        # initalise centers
        n_iter=10
        C = initialCentroids(k, self.dimension)
        # repeat for n_iter for each cluster:
        for _ in range(0, n_iter):
            # assign each point to its closest center
            assignments = self.data.apply(lambda row: getClosestCenter(row, C), axis=1)
            # update center to be the average of all points assigned
            for i in range(0, len(C)):
                if (assignments == i).any():
                    C.iloc[i, :] = self.data[assignments == i].mean()
        return C
    
    def dplloyd(self, k: int, n_iter: int):
        # initalise centers
        C = initialCentroids(k, self.dimension)
        e_per_iter = self.epsilon / n_iter
        # repeat for n_iter for each cluster:
        for _ in range(0, n_iter):
            # assign each point to its closest center
            assignments = self.data.apply(lambda row: getClosestCenter(row, C), axis=1)
            # update center to be the average of all points assigned
            for i in range(0, len(C)):
                if (assignments == i).any():
                    # noisily calculate the number of points in the cluster
                    n = self.data[assignments == i].count() + self.laplace_mechanism(2 * n_iter / e_per_iter, 1) # DO I NEED TO SPLIT EPSILON HERE OVER THE TWO NOISY UPDATES?    
                    # noisily calculate the sum of points in the cluster
                    s = self.data[assignments == i].sum() + self.laplace_mechanism((2*self.dimension*n_iter) / e_per_iter)
                    # update centroid
                    C.iloc[i, :] = s / n
        return C
    
    def private_grid(self, k: int, M:int):
        if not M:
            M = round((self.dimension*self.epsilon) / 10)

        edges = np.linspace(-1, 1, M + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        grid = pd.DataFrame(np.array(list(itertools.product(centers, repeat=self.dimension))))

        squares = self.data.apply(lambda row: getSquare(row, grid), axis=1)

        points = squares.value_counts().reset_index(name="count")
        w_points = pd.merge(points, grid, on = list(grid.columns),  how="outer").fillna(0)
        # here we would add laplace noise to the counts, since the l1 norm of the counts vector is 1 we want to add a vector sampled from lap(1/e)
        w_points["count"] += self.laplace_mechanism(1/self.epsilon, grid.shape[0])
        
        # now we have the weighted points we can apply weighted lloyd with any number of iterations on this private synopsis

        return lloyd_with_weights(k, w_points.iloc[:, :self.dimension], w_points.iloc[:, :-1], n_iter=10, rs=self.random_seed)
    
    # move private bucket method here 
    def private_bucket():
        pass
    


        
        
