import numpy as np
import pandas as pd 
from typing import Optional, List 

""" 
Want to make a class which stores datasets and parameters required for all algorithms. 
Can then add methods to test each algorithm using the pre-defined parameters. 
"""

def scale(X: pd.Series, minimum: float, maximum: float):
    return 2 * ((X - minimum) / maximum - minimum) - 1

def non_private_scale(X: pd.Series):
    return 2 * ((X - X.min()) / X.max() - X.min()) - 1

class Dataset:
    def __init__(self, data: pd.DataFrame, epsilon: float = 1, dataset_bounds_min: Optional[List[float]] = None, dataset_bounds_max: Optional[List[float]] = None, dimension: Optional[int] = None):

        # ensure dimension given matches dataset
        if dimension is None:
            self.dimension = data.shape[1]
        elif dimension != data.shape[1]:
            raise Exception("Dimension given does not match the shape of the data")
        else:
            self.dimension = dimension
        
        self.data = data
        self.epsilon = epsilon

        # normalise data to be in [-1,1]
        if (dataset_bounds_min is None) or (dataset_bounds_max is None):
            print("NON PRIVATELY normalising data..")
            self.data = self.data.apply(non_private_scale, axis="columns")
        elif (len(dataset_bounds_min) != self.dimension) or (len(dataset_bounds_max) != self.dimension):
            raise Exception("Bounds given do not match the shape of the data")
        else:
            for col_index in range (0, self.dimension):
                self.data.iloc[:, col_index] = scale(self.data.iloc[:, col_index], dataset_bounds_min[col_index], dataset_bounds_max[col_index])
        


        
        
