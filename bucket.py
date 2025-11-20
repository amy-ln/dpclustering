import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from typing import Optional

# use this from the diffprivlibrary for hashing 
from lsh import SimHash

# import my own functions from util
from util import noise, normalise

# TO IMPROVE
# not sure adding privacy correctly - how to split up privacy budget? how much noise should we add to the noisy counts?
# is the tree splitting correctly? observed many leaf nodes with large numbers of points. does normalisation affect this? - I think splitting is weird because of large amounts of noise 
# perhaps number of clusters will affect the max_depth and branching_thresholds somehow?

class Params:

    def __init__(self, epsilon: float, delta: float, radius:float, dimension: int, branching_threshold: int,  max_depth: int = 20):
        self.epsilon = epsilon
        self.delta = delta
        self.radius = radius
        self.dimension = dimension
        self.branching_threshold = branching_threshold
        self.max_depth = max_depth

def create_bucket_synopsis(X: pd.DataFrame, p: Params):

    # give half the privacy budget to computing the tree and half to computing weighted averages of points?
    e1, e2 = p.epsilon/2, p.epsilon/2 

    # make sure data is centered and that all points fall within provided radius
    
 
    # create tree : return leaf nodes pointing to all points "in" that node
    tree = LshTree(e1, p.branching_threshold, p.max_depth, X, X.shape[1])
    leaves = tree.get_leaves()

    # use leaf nodes to create the weighted points 
    rows = []
    weights = []
    for leaf in leaves:
        # a sum query has sensitivity d * radius
        coords, weight = leaf
        average = ((coords).sum() + noise((p.dimension * p.radius)/e2, p.dimension )) / weight
        row = list(average)
        weights.append(weight)
        rows.append(row)

    coreset_points = pd.DataFrame(rows)
    coreset_weights = pd.DataFrame(weights)

    # scale coreset points to defined radius - improves accuracy
    # does this violate privacy? 
    scale = p.radius / np.maximum(
        np.linalg.norm(coreset_points, axis=-1), p.radius
    ).reshape(-1, 1)
    coreset_points = coreset_points * scale

    return coreset_points, coreset_weights

class LshTree:

    def __init__(self, e:float, branching_threshold:int, max_depth:int, X:np.ndarray, dimension:int):
        self.e_per_layer = e / max_depth
        self.branching_threshold = branching_threshold
        self.max_depth = max_depth
        self.hash = SimHash(dimension, max_depth)
        self.tree = self.create_lsh_tree(X)

    def branch(self, points, depth: int, hash_prefix: str):

        ## need to add noise to the counts of points here to satisfy privacy
        noisy_count = len(points) + noise(1 / self.e_per_layer, 1)[0]
        if (noisy_count <= self.branching_threshold) or (depth >= self.max_depth):
            return (points, noisy_count)
        
        tree = self.hash.group_by_next_hash(points, hash_prefix)

        tree["0"] = self.branch(tree["0"], depth + 1, hash_prefix + "0")
        tree["1"] = self.branch(tree["1"], depth + 1, hash_prefix + "1")

        return tree 
    
    def get_leaves_of_tree(self, tree):
        leaves = []
        for key, value in tree.items():
            if isinstance(value, dict):
                leaves.extend(self.get_leaves_of_tree(value))
            else:
                leaves.append(value)
        return leaves
    
    def get_leaves(self):
        all_leaves = self.get_leaves_of_tree(self.tree)
        # remove empty ones
        return list(filter(lambda l : l[0].size > 0, all_leaves))
        
    def create_lsh_tree(self, X: pd.DataFrame):

        print("Creating tree...")
        tree = self.branch(X, 0, "")

        return tree


rng = np.random.default_rng(42)
data = np.concat(
    [rng.multivariate_normal(mean=[1,1], cov=[[1,0],[0,1]], size=100),
    rng.multivariate_normal(mean=[5,5], cov=[[1,0],[0,1]], size=100),
    rng.multivariate_normal(mean=[5,0], cov=[[1,0],[0,1]], size=100)]
)
# center
data = data - data.mean()

# define parameters
p = Params(epsilon=1, delta=0.0001, radius=5, dimension=2, branching_threshold=3, max_depth=10)

# set the radius to be 5 and scale so everything is inside
scale = p.radius / np.maximum(
        np.linalg.norm(data, axis=1), p.radius
    ).reshape(-1, 1)
print(data.shape)
data = data * scale

points, weights = create_bucket_synopsis(data, p)
print(type(points))
plt.scatter(data[:,0], data[:,1], 1)
plt.scatter(points[0], points[1], weights, color="red")
plt.show()