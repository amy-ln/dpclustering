import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from typing import Optional, override
from dataclasses import dataclass


# use this from the diffprivlibrary for hashing 
from lsh import SimHash

# google privacy accountant
from privacy_accountant import privacy_calculator
from privacy_accountant import clustering_params
from privacy_accountant import central_privacy_utils

# import my own functions from util
from util import noise

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

def bucket_using_privacy_accountant(X: pd.DataFrame, p: Params):

    multipliers = clustering_params.PrivacyCalculatorMultiplier()
    privacy_param = clustering_params.DifferentialPrivacyParam(p.epsilon, p.delta)
    pcalc = privacy_calculator.PrivacyCalculator(
      privacy_param, p.radius, p.max_depth, multipliers)
    
    noisy_n = central_privacy_utils.get_private_count(X.shape[0], pcalc.count_privacy_param)

    # copy heuristic thresholds from google code 
    num_points_in_node_for_low_noise = int(
      10 * np.sqrt(X.shape[1]) *
      pcalc.average_privacy_param.gaussian_standard_deviation /
      pcalc.average_privacy_param.sensitivity)
    
    p.include_threshold = min(num_points_in_node_for_low_noise,
                               noisy_n // (2 * p.k))
    p.include_threshold = max(1, p.include_threshold)
    p.branching_threshold = 3*p.include_threshold
    print(f"Parameters used \n max depth: {p.max_depth}\n branching threshold: {p.branching_threshold} \n include_threshold: {p.include_threshold}")

    tree = LshTreeAdvanced(pcalc.count_privacy_param, p.branching_threshold, p.include_threshold, p.max_depth, X, p.dimension, noisy_n) # change to use discrete laplace
    leaves = tree.get_leaves()

    averages = []
    for (points, noisy_count) in leaves:
        averages.append(central_privacy_utils.get_private_average(points, noisy_count, pcalc.average_privacy_param, p.dimension))

    coreset_points = pd.DataFrame(averages)
    coreset_weights = pd.DataFrame([l[1] for l in leaves])

    # scale coreset points to defined radius - improves accuracy. does not violate privacy as coreset points are private
    scale = p.radius / np.maximum(
        np.linalg.norm(coreset_points, axis=-1), p.radius
    ).reshape(-1, 1)
    print(scale)
    coreset_points = coreset_points * scale

    return coreset_points, coreset_weights


def create_bucket_synopsis(X: pd.DataFrame, p: Params):

    # give half the privacy budget to computing the and half to computing weighted averages of points?
    e1, e2 = p.epsilon*0.8, p.epsilon*0.2 

    # make sure data is centered and that all points fall within provided radius

    # we compute max_depth + 1 private counts so epsilon we can use here is e1/(max_depth + 1)
    # compute a noisy count of number of rows in entire dataset
    noisy_n = len(X) + noise(1/(e1/(p.max_depth + 1)), 1)[0]
    print("Noisy total count", noisy_n)
    p.calculate_thresholds(noisy_n)
 
    # create tree : return leaf nodes pointing to all points "in" that node
    tree = LshTree(e1/(p.max_depth + 1), p.branching_threshold, p.include_threshold, p.max_depth, X, p.dimension, noisy_n)
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
    child0: Optional["TreeNode"] = None
    child1: Optional["TreeNode"] = None

class LshTree:

    def __init__(self, e_per_layer:float, branching_threshold:int, include_node_threshold: int, max_depth:int, X:np.ndarray, dimension:int, noisy_total_count: float):

        self.e_per_layer = e_per_layer
        self.branching_threshold = branching_threshold
        self.include_node_threshold = include_node_threshold
        self.max_depth = max_depth
        self.hash = SimHash(dimension, max_depth)
        self.tree = self.create_lsh_tree(X, noisy_total_count)

    def count_with_noise(self, points):
        return len(points) + noise(1 / self.e_per_layer, 1)[0]
        
    def branch(self, t : TreeNode, depth: int) -> None:

        if (t.noisy_count <= self.branching_threshold) or (depth >= self.max_depth):
            if t.noisy_count >= self.include_node_threshold:
                self.leaves.append(t)
        
        else:
            points_dict = self.hash.group_by_next_hash(t.points, t.hash_prefix)

            t.child0 = TreeNode(self.count_with_noise(points_dict["0"]), t.hash_prefix + "0", points_dict["0"])
            t.child1 = TreeNode(self.count_with_noise(points_dict["1"]), t.hash_prefix + "1", points_dict["1"])

            self.branch(t.child0, depth + 1)
            self.branch(t.child1, depth + 1)
    
    def get_leaves(self):
        l = [(t.points, t.noisy_count) for t in self.leaves]
        print("leaves", [np.sum(t.points, axis=0) for t in self.leaves])
        return l
        
    def create_lsh_tree(self, X: pd.DataFrame, noisy_total_count:float ):
        self.leaves: list[TreeNode] = []
        noisy_total_count = max(1, noisy_total_count) #always needs to be >= 1 otherwise won't branch
        print("Creating tree...")
        tree = TreeNode(noisy_total_count, "", X)
        self.branch(tree, 1)

        return tree
    
class LshTreeAdvanced(LshTree):

    @override 
    def count_with_noise(self, points):
        return central_privacy_utils.get_private_count(len(points), self.e_per_layer)


rng = np.random.default_rng(42)
data = np.concat(
    [rng.multivariate_normal(mean=[1,1], cov=[[1,0],[0,1]], size=100),
    rng.multivariate_normal(mean=[5,5], cov=[[1,0],[0,1]], size=100),
    rng.multivariate_normal(mean=[5,0], cov=[[1,0],[0,1]], size=100)]
)
# center
data = data - data.mean()

# define parameters
p = Params(epsilon=1, delta=0.0001, radius=5, dimension=2, k=3, max_depth = 20)

# set the radius to be 5 and scale so everything is inside
scale = p.radius / np.maximum(
        np.linalg.norm(data, axis=1), p.radius
    ).reshape(-1, 1)
print(data.shape)
data = data * scale

points, weights = bucket_using_privacy_accountant(data, p)
print(points)
print(weights)
plt.scatter(data[:,0], data[:,1], 1)
plt.scatter(points[0], points[1], np.abs(weights), color="red")
plt.show()