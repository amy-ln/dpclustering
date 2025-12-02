import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from typing import Optional
from dataclasses import dataclass

# use this from the diffprivlibrary for hashing 
from lsh import SimHash

# import my own functions from util
from util import noise

# TO IMPROVE
# not sure adding privacy correctly - how to split up privacy budget? how much noise should we add to the noisy counts?
# is the tree splitting correctly? observed many leaf nodes with large numbers of points. does normalisation affect this? - I think splitting is weird because of large amounts of noise 
# perhaps number of clusters will affect the max_depth and branching_thresholds somehow?

class Params:

    def __init__(self, epsilon: float, delta: float, radius:float, dimension: int, k:int, max_depth: int = 20):
        self.epsilon = epsilon
        self.delta = delta
        self.radius = radius
        self.dimension = dimension
        self.k=k
        self.max_depth = max_depth

    def calculate_thresholds(self, n : int):
        # heuristics chosen from details of implementation in appendix to paper 
        self.branching_threshold = 0.5 * np.floor(n/self.k)
        self.include_threshold = self.branching_threshold / 3
        print(f"Parameters used \n max depth: {self.max_depth}\n branching threshold: {self.branching_threshold} \n include_threshold: {self.include_threshold}")

def count_with_noise(points, e):
    return len(points) + noise(1 / e, 1)[0]

def create_bucket_synopsis(X: pd.DataFrame, p: Params):

    # give half the privacy budget to computing the and half to computing weighted averages of points?
    e1, e2 = p.epsilon*0.8, p.epsilon*0.2 

    # make sure data is centered and that all points fall within provided radius

    # we compute max_depth + 1 private counts so epsilon we can use here is e1/(max_depth + 1)
    # compute a noisy count of number of rows in entire dataset
    noisy_n = count_with_noise(X, (e1 / (p.max_depth) + 1))
    print("Noisy total count", noisy_n)
    p.calculate_thresholds(noisy_n)
 
    # create tree : return leaf nodes pointing to all points "in" that node
    tree = LshTree(e1, p.branching_threshold, p.include_threshold, p.max_depth, X, p.dimension, noisy_n)
    leaves = tree.get_leaves()

    # a sum query has sensitivity d * radius
    averages = []
    for (points, noisy_count) in leaves:
        a = np.sum(points, axis=0) + noise((p.dimension * p.radius)/e2, p.dimension)
        averages.append(a / noisy_count)

    coreset_points = pd.DataFrame(averages)
    coreset_weights = pd.DataFrame([l[1] for l in leaves])

    # scale coreset points to defined radius - improves accuracy. does not violate privacy as coreset points are private
    scale = p.radius / np.maximum(
        np.linalg.norm(coreset_points, axis=-1), p.radius
    ).reshape(-1, 1)
    coreset_points = coreset_points * scale

    return coreset_points, coreset_weights


@dataclass
class TreeNode:

    noisy_count: float 
    hash_prefix: str
    points: np.ndarray
    child0: Optional[TreeNode] = None
    child1: Optional[TreeNode] = None

class LshTree:

    def __init__(self, e:float, branching_threshold:int, include_node_threshold: int, max_depth:int, X:np.ndarray, dimension:int, noisy_total_count: float):

        self.e_per_layer = e / (max_depth + 1)
        self.branching_threshold = branching_threshold
        self.include_node_threshold = include_node_threshold
        self.max_depth = max_depth
        self.hash = SimHash(dimension, max_depth)
        self.tree = self.create_lsh_tree(X, noisy_total_count)
        

    def branch(self, t : TreeNode, depth: int) -> None:

        if (t.noisy_count <= self.branching_threshold) or (depth >= self.max_depth):
            if t.noisy_count >= self.include_node_threshold:
                self.leaves.append(t)
        
        else:
            points_dict = self.hash.group_by_next_hash(t.points, t.hash_prefix)

            t.child0 = TreeNode(count_with_noise(points_dict["0"], self.e_per_layer), t.hash_prefix + "0", points_dict["0"])
            t.child1 = TreeNode(count_with_noise(points_dict["1"], self.e_per_layer), t.hash_prefix + "1", points_dict["1"])

            self.branch(t.child0, depth + 1)
            self.branch(t.child1, depth + 1)
    
    def get_leaves(self):
        l = [(t.points, t.noisy_count) for t in self.leaves]
        print("leaves", [np.sum(t.points, axis=0) for t in self.leaves])
        return l
        
    def create_lsh_tree(self, X: pd.DataFrame, noisy_total_count:float ):
        self.leaves: list[TreeNode] = []

        print("Creating tree...")
        tree = TreeNode(noisy_total_count, "", X)
        self.branch(tree, 1)

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
p = Params(epsilon=1, delta=0.0001, radius=5, dimension=2, k=3, max_depth = 5)

# set the radius to be 5 and scale so everything is inside
scale = p.radius / np.maximum(
        np.linalg.norm(data, axis=1), p.radius
    ).reshape(-1, 1)
print(data.shape)
data = data * scale

points, weights = create_bucket_synopsis(data, p)
print(points)
print(weights)
plt.scatter(data[:,0], data[:,1], 1)
plt.scatter(points[0], points[1], np.abs(weights), color="red")
plt.show()