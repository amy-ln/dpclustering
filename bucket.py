import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from typing import Optional

# use this from the diffprivlibrary for hashing 
from lsh import SimHash

# import my own functions from util
from util import noise

# TO IMPROVE
# not sure adding privacy correctly - how to split up privacy budget? how much noise should we add to the noisy counts?
# is the tree splitting correctly? observed many leaf nodes with large numbers of points. does normalisation affect this? - I think splitting is weird because of large amounts of noise 
# perhaps number of clusters will affect the max_depth and branching_thresholds somehow?

class Params:

    def __init__(self, epsilon: float, delta: float, radius:float, dimension: int, k:int):
        self.epsilon = epsilon
        self.delta = delta
        self.radius = radius
        self.dimension = dimension
        self.k=k

    def calculate_thresholds(self, n : int):
        # heuristics chosen from details of implementation in appendix to paper 
        self.max_depth = int(np.ceil(np.log2(self.k)) + 3)
        self.branching_threshold = 0.1 * np.floor(n/self.k)
        self.include_threshold = self.branching_threshold / 3
        print(f"Parameters used \n max depth: {self.max_depth} \n branching threshold: {self.branching_threshold} \n include_threshold: {self.include_threshold}")


def create_bucket_synopsis(X: pd.DataFrame, p: Params):

    # give half the privacy budget to computing the and half to computing weighted averages of points?
    e1, e2 = p.epsilon*0.9, p.epsilon*0.1 

    # make sure data is centered and that all points fall within provided radius

    # compute a noisy count of number of rows in entire dataset
    noisy_n = X.shape[0] + noise(1 / p.epsilon, 1)[0]
    p.calculate_thresholds(noisy_n)
 
    # create tree : return leaf nodes pointing to all points "in" that node
    tree = LshTree(e1, p.branching_threshold, p.include_threshold, p.max_depth, X, p.dimension)
    leaves, leaves_noisy_counts = tree.get_leaves()

    # a sum query has sensitivity d * radius
    averages = []
    for i in range(len(leaves)):
        a = np.sum(leaves[i], axis=0) + noise((p.dimension * p.radius)/e2, p.dimension )
        print(f"Actual average: {np.sum(leaves[i], axis=0)}")
        print(f"Noisy average: {a}")
        print(f"Divided by noisy count: {a / leaves_noisy_counts[i]}")
        averages.append(a)

    coreset_points = pd.DataFrame(averages)
    coreset_weights = pd.DataFrame(leaves_noisy_counts)

    # scale coreset points to defined radius - improves accuracy. does not violate privacy as coreset points are private
    scale = p.radius / np.maximum(
        np.linalg.norm(coreset_points, axis=-1), p.radius
    ).reshape(-1, 1)
    coreset_points = coreset_points * scale

    return coreset_points, coreset_weights

class LshTree:

    def __init__(self, e:float, branching_threshold:int, include_node_threshold: int, max_depth:int, X:np.ndarray, dimension:int):

        self.e_per_layer = e / max_depth
        self.branching_threshold = branching_threshold
        self.include_node_threshold = include_node_threshold
        self.max_depth = max_depth
        self.hash = SimHash(dimension, max_depth)
        self.tree = self.create_lsh_tree(X)

    def branch(self, points, depth: int, hash_prefix: str):

        ## need to add noise to the counts of points here to satisfy privacy
        noisy_count = len(points) + noise(1 / self.e_per_layer, 1)[0]
        if (noisy_count <= self.branching_threshold) or (depth >= self.max_depth):
            if noisy_count >= self.include_node_threshold:
                self.leaves.append(points)
                self.leaves_noisy_counts.append(noisy_count)
            return (points, noisy_count)
        
        tree = self.hash.group_by_next_hash(points, hash_prefix)
        
        # print(type(tree["0"]), tree["0"].shape)
        # print(f"Split {len(tree["0"])} / {len(tree["1"])} with averages {np.mean(tree["0"], axis=0)} / {np.mean(tree["1"], axis=0)}")

        tree["0"] = self.branch(tree["0"], depth + 1, hash_prefix + "0")
        tree["1"] = self.branch(tree["1"], depth + 1, hash_prefix + "1")

        return tree 
    
    def get_leaves(self):
        return self.leaves, self.leaves_noisy_counts
        
    def create_lsh_tree(self, X: pd.DataFrame):
        self.leaves = []
        self.leaves_noisy_counts = []

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
p = Params(epsilon=1, delta=0.0001, radius=5, dimension=2, k=3)

# set the radius to be 5 and scale so everything is inside
scale = p.radius / np.maximum(
        np.linalg.norm(data, axis=1), p.radius
    ).reshape(-1, 1)
print(data.shape)
data = data * scale

points, weights = create_bucket_synopsis(data, p)
print(points)
plt.scatter(data[:,0], data[:,1], 1)
plt.scatter(points[0], points[1], np.abs(weights), color="red")
plt.show()