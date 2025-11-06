import pandas as pd
import numpy as np

# Euclidean distance between x and y is sum of squared differences.
def distance(x, y):
    return np.sqrt(sum((x-y)**2))

def noise(scale, d):
    rng = np.random.default_rng()
    return rng.laplace(0, scale, size=d)

def normalise(df: pd.DataFrame):
    return df.apply(lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1)