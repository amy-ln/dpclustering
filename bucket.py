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

from lloyd import lloyd_with_weights

class Params:

    def __init__(self, epsilon: float, delta: float, radius:float, dimension: int, k:int, max_depth: int = 20, branching_threshold: float = None, include_threshold: float = None):
        self.epsilon = epsilon
        self.delta = delta
        self.radius = radius
        self.dimension = dimension
        self.k=k
        self.max_depth = max_depth
        self.branching_threshold = branching_threshold
        self.include_threshold = include_threshold

def bucket_using_privacy_accountant(X: np.ndarray, p: Params, seed: int=42):

    multipliers = clustering_params.PrivacyCalculatorMultiplier()
    privacy_param = clustering_params.DifferentialPrivacyParam(p.epsilon, p.delta)
    pcalc = privacy_calculator.PrivacyCalculator(
      privacy_param, p.radius, p.max_depth, multipliers)
    
    noisy_n = central_privacy_utils.get_private_count(X.shape[0], pcalc.count_privacy_param, seed)
    #print("privacy parameters: ", pcalc.average_privacy_param.gaussian_standard_deviation, pcalc.average_privacy_param.sensitivity )
    # copy heuristic thresholds from google code 
    num_points_in_node_for_low_noise = int(
      10 * np.sqrt(X.shape[1]) *
      pcalc.average_privacy_param.gaussian_standard_deviation /
      pcalc.average_privacy_param.sensitivity)
    
    if p.include_threshold is None:
        p.include_threshold = min(num_points_in_node_for_low_noise,
                                noisy_n // (2 * p.k))
        p.include_threshold = max(1, p.include_threshold)
    if p.branching_threshold is None:
        p.branching_threshold = 3*p.include_threshold
    print(f"Parameters used \n max depth: {p.max_depth}\n branching threshold: {p.branching_threshold} \n include_threshold: {p.include_threshold}")
    print(f"pcalc", pcalc.average_privacy_param, pcalc.count_privacy_param)
    tree = LshTreeAdvanced(pcalc.count_privacy_param, p.branching_threshold, p.include_threshold, p.max_depth, X, p.dimension, noisy_n, seed) 
    leaves = tree.get_leaves()
    #print("Printing entire non private tree...")
    averages = []
    for (points, noisy_count) in leaves:
        averages.append(central_privacy_utils.get_private_average(points, noisy_count, pcalc.average_privacy_param, p.dimension, seed))

    coreset_points = averages
    coreset_weights = np.array([l[1] for l in leaves])

    # scale coreset points to defined radius - improves accuracy. does not violate privacy as coreset points are private
    scale = p.radius / np.maximum(
        np.linalg.norm(coreset_points, axis=-1), p.radius
    ).reshape(-1, 1)
    coreset_points = coreset_points * scale

    return coreset_points, coreset_weights


def create_bucket_synopsis(X: np.ndarray, p: Params, seed: int = 42, privacy_split: float = 0.8):

    # give half the privacy budget to computing the and half to computing weighted averages of points?
    e1, e2 = p.epsilon*privacy_split, p.epsilon*(1-privacy_split)

    # make sure data is centered and that all points fall within provided radius

    # we compute max_depth + 1 private counts so epsilon we can use here is e1/(max_depth + 1)
    # compute a noisy count of number of rows in entire dataset
    noisy_n = X.shape[0] + noise(1/(e1/(p.max_depth + 1)), 1)[0]
    
    # copy heuristic thresholds from google code 
    num_points_in_node_for_low_noise = int(
      10 * np.sqrt(X.shape[1]) *
      (1/e2) /
      p.radius)
    
    if p.include_threshold is None:
        p.include_threshold = min(num_points_in_node_for_low_noise,
                                noisy_n // (2 * p.k))
        p.include_threshold = max(1, p.include_threshold)
    if p.branching_threshold is None:
        p.branching_threshold = 3*p.include_threshold
    print(f"Parameters used \n max depth: {p.max_depth}\n branching threshold: {p.branching_threshold} \n include_threshold: {p.include_threshold}")
 
    # create tree : return leaf nodes pointing to all points "in" that node
    tree = LshTree(e1/(p.max_depth + 1), p.branching_threshold, p.include_threshold, p.max_depth, X, p.dimension, noisy_n, seed)
    leaves = tree.get_leaves()

    # a sum query has sensitivity d * radius
    averages = []
    for (points, noisy_count) in leaves:
        a = np.sum(points, axis=0) + noise((p.dimension * p.radius)/e2, p.dimension, seed)
        averages.append(a / noisy_count)
    #print(f"num leaves:", len(leaves))
    coreset_points = np.array(averages)
    coreset_weights = np.array([l[1] for l in leaves])

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


class LshTree:

    def __init__(self, e_per_layer:float, branching_threshold:int, include_node_threshold: int, max_depth:int, X:np.ndarray, dimension:int, noisy_total_count: float, seed:int = 42):

        self.seed = seed
        self.e_per_layer = e_per_layer
        self.branching_threshold = branching_threshold
        self.include_node_threshold = include_node_threshold
        self.max_depth = max_depth
        self.hash = SimHash(dim=dimension, max_hash_len=max_depth, seed=seed)
        self.create_lsh_tree(X, noisy_total_count)

    def count_with_noise(self, points):
        return len(points) + noise(1 / self.e_per_layer, 1, self.seed)[0]
        
    def branch(self, t : TreeNode) -> None:

        points_dict = self.hash.group_by_next_hash(t.points, t.hash_prefix)

        child0 = TreeNode(self.count_with_noise(points_dict["0"]), t.hash_prefix + "0", points_dict["0"])
        child1 = TreeNode(self.count_with_noise(points_dict["1"]), t.hash_prefix + "1", points_dict["1"])

        return [child0, child1]
    
    def get_leaves(self):
        level = 0
        leaves = []
        while level <= self.max_depth:
            nodes = self.tree.get(level, [])
            if nodes:
                leaf_nodes = list(filter(self.is_leaf, nodes))
                if leaf_nodes:
                    leaves = leaves + [(t.points, t.noisy_count) for t in leaf_nodes]
                level += 1
            else:
                break
        return leaves

    def create_lsh_tree(self, X: pd.DataFrame, noisy_total_count:float ):
        self.leaves: list[TreeNode] = []
        self.tree = {} # level index to nodes on level
        noisy_total_count = max(1, noisy_total_count) #always needs to be >= 1 otherwise won't branch
        #print("Creating tree...")
        root = TreeNode(noisy_total_count, "", X)
        self.tree[0] = [root]
        level = 0
        while level < self.max_depth:
            nodes_to_branch = list(filter(self.can_branch, self.tree.get(level, [])))
            if nodes_to_branch:
                level += 1
                self.tree[level] = np.concatenate([self.branch(node) for node in nodes_to_branch]).tolist()
            else:
                break

    def print_tree(self):
        level = 0
        nodes = self.tree.get(level, [])
        while nodes:
            print ("level", level, [n.noisy_count for n in nodes])
            level += 1 
            nodes = self.tree.get(level, [])

    def can_branch(self, node: TreeNode):
        return node.noisy_count > self.branching_threshold
    
    def is_leaf(self, node: TreeNode):
        return (node.noisy_count <= self.branching_threshold) and (node.noisy_count >= self.include_node_threshold)

    
class LshTreeAdvanced(LshTree):

    @override 
    def count_with_noise(self, points):
        return central_privacy_utils.get_private_count(len(points), self.e_per_layer, seed=self.seed)


rng = np.random.default_rng(20)
data = np.concat(
    [rng.multivariate_normal(mean=[1,1], cov=[[1,0],[0,1]], size=100),
    rng.multivariate_normal(mean=[5,5], cov=[[1,0],[0,1]], size=100),
    rng.multivariate_normal(mean=[5,0], cov=[[1,0],[0,1]], size=100)]
)
# center
data = data - data.mean()