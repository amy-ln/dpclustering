import pandas as pd
import numpy as np

# Euclidean distance between x and y is sum of squared differences.
def distance(x, y):
    return np.sqrt(sum((x-y)**2))

def noise(scale, d, seed=42):
    rng = np.random.default_rng(seed) # add random state for reproducibility
    return rng.laplace(0, scale, size=d)

def normalise(df: pd.DataFrame):
    return df.apply(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)

def getClosestCenter(x, C):
    # x is the data point, C is all the centers. Return index of the closest center. 
    distances = np.apply_along_axis(lambda c: distance(x,c), axis=1, arr=C)
    return np.argmin(distances)

def initialCentroids(k:int, d:int, minimum:int = -1, maximum:int= 1):
    return np.linspace([minimum]*d, [maximum]*d, num=k)

def getSquare(x: np.array, grid: pd.DataFrame):
    distances = grid.apply(lambda row: distance(x, row), axis=1)
    return grid.iloc[distances.idxmin(),:]