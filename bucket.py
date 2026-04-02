import pandas as pd
import numpy as np
from dataclasses import dataclass

# use this from the diffprivlibrary for hashing 
from lsh import SimHash

# this is the laplace noise function
from util import noise

# class for storing all the necessary parameters 
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

# additionally define the gaussian noise mechanism 
def gaussian_mechanism(epsilon, delta, sensitivity, dimension, seed):
    rng = np.random.default_rng(seed)
    scale = (np.sqrt(2*np.log(1.25 / delta))*sensitivity) / epsilon
    return rng.normal(loc=0.0, scale=scale, size=dimension)

def create_bucket_synopsis(X: np.ndarray, p: Params, seed: int = 42, privacy_split: float = 0.8, use_gaussian = True) -> tuple[np.ndarray, np.ndarray]:
    """ Create the private synopsis with Tree with LSH method

    Args:
        X (np.ndarray): Data to be used, should be centered and fall within the defined radius
        p (Params): Set of parameters for building the tree
        seed (int, optional): Seed for randomness provided for reproducibility. Defaults to 42.
        privacy_split (float, optional): Describes how the privacy budget should be divided between counts and sums. Defaults to 0.8.
        use_gaussian (bool, optional): Use the Gaussian mechanism if True and otherwise use Laplace mechanism for sums. Defaults to True.

    Returns:
        (np.ndarray, np.ndarray): The private synopsis, the first array is the coordinates of the points and the second is their weights. 
    """    

    # give half the privacy budget to computing the and half to computing weighted averages of points?
    e1, e2 = p.epsilon*(1-privacy_split), p.epsilon*privacy_split

    # we compute max_depth + 1 private counts so epsilon we can use here is e1/(max_depth + 1)
    # compute a noisy count of number of rows in entire dataset
    noisy_n = X.shape[0] + noise(1/(e1/(p.max_depth + 1)), 1)[0]
    
    # some experimental heuristics - see final report for full details on choosing thresholds. 
    num_points_in_node_for_low_noise = int(
      10 * np.sqrt(X.shape[1]) * (1/e2) * 2*np.log(1.25 / p.delta))
    
    if p.include_threshold is None:
        p.include_threshold = min(num_points_in_node_for_low_noise,
                                noisy_n // (2 * p.k))
        p.include_threshold = max(1, p.include_threshold)
    if p.branching_threshold is None:
        p.branching_threshold = 3*p.include_threshold
 
    # create tree : return leaf nodes pointing to all points "in" that node
    tree = LshTree(e1/(p.max_depth + 1), p.branching_threshold, p.include_threshold, p.max_depth, X, p.dimension, noisy_n, seed)
    leaves = tree.get_leaves()

    # a sum query has sensitivity  radius
    averages = []
    if use_gaussian:
        for (points, noisy_count) in leaves:
            # assuming every entry is between [-1, 1] bound l1 by d 
            a = np.sum(points, axis=0) + gaussian_mechanism(e2, p.delta, p.radius, p.dimension, seed) #noise(p.dimension/e2, p.dimension, seed)
            averages.append(a / noisy_count)
    else:
        for (points, noisy_count) in leaves:
            # assuming every entry is between [-1, 1] bound l1 by d 
            a = np.sum(points, axis=0) + noise(p.dimension/e2, p.dimension, seed)
            averages.append(a / noisy_count)

    coreset_points = np.array(averages)
    coreset_weights = np.array([l[1] for l in leaves])

    # scale coreset points to defined radius - improves accuracthis county. does not violate privacy as coreset points are private
    scale = p.radius / np.maximum(
        np.linalg.norm(coreset_points, axis=-1), p.radius
    ).reshape(-1, 1)
    coreset_points = coreset_points * scale
    

    return coreset_points, coreset_weights



@dataclass
class TreeNode:

    noisy_count: float # private count of the number of points in this node
    hash_prefix: str # hash prefix provided by the lsh function
    points: np.ndarray # (non-private) points contained in this node


class LshTree:

    def __init__(self, e_per_layer:float, branching_threshold:int, include_node_threshold: int, max_depth:int, X:np.ndarray, dimension:int, noisy_total_count: float, seed:int = 42):

        self.seed = seed
        self.e_per_layer = e_per_layer
        self.branching_threshold = branching_threshold
        self.include_node_threshold = include_node_threshold
        self.max_depth = max_depth
        # creating the hash function 
        self.hash = SimHash(dim=dimension, max_hash_len=max_depth, seed=seed)
        # create the tree
        self.create_lsh_tree(X, noisy_total_count)

    def count_with_noise(self, points):
        # laplace noise is always used for the counts
        return len(points) + noise(1 / self.e_per_layer, 1, self.seed)[0]
        
    def branch(self, t : TreeNode) -> None:

        # group points by the next bit in the hash prefix
        points_dict = self.hash.group_by_next_hash(t.points, t.hash_prefix)

        # split into 2 child nodes
        child0 = TreeNode(self.count_with_noise(points_dict["0"]), t.hash_prefix + "0", points_dict["0"])
        child1 = TreeNode(self.count_with_noise(points_dict["1"]), t.hash_prefix + "1", points_dict["1"])

        return [child0, child1]
    
    def get_leaves(self):
        # iterate through the tree to retrieve all leaf nodes
        level = 0
        leaves = []
        while level < self.max_depth:
            nodes = self.tree.get(level, [])
            if nodes:
                leaf_nodes = list(filter(self.is_leaf, nodes))
                if leaf_nodes:
                    leaves = leaves + [(t.points, t.noisy_count) for t in leaf_nodes]
                level += 1
            else:
                break
        final_level_nodes = self.tree.get(self.max_depth, [])
        if final_level_nodes:
            to_include = list(filter(lambda node: node.noisy_count >= self.include_node_threshold, final_level_nodes))
            leaves = leaves + [(t.points, t.noisy_count) for t in to_include]
        return leaves

    def create_lsh_tree(self, X: pd.DataFrame, noisy_total_count:float ):
        self.leaves: list[TreeNode] = []
        self.tree = {} # level index to nodes on level
        noisy_total_count = max(1, noisy_total_count) #always needs to be >= 1 otherwise won't branch
        root = TreeNode(noisy_total_count, "", X)
        self.tree[0] = [root]
        level = 0
        # create the tree one level at a time
        while level < self.max_depth:
            # only branch when noisy count is larger than branching threshold
            nodes_to_branch = list(filter(self.can_branch, self.tree.get(level, [])))
            if nodes_to_branch:
                level += 1
                self.tree[level] = np.concatenate([self.branch(node) for node in nodes_to_branch]).tolist()
            else:
                break

    # for debugging purposes
    def print_tree(self):
        level = 0
        nodes = self.tree.get(level, [])
        while nodes:
            print ("level", level, [(n.noisy_count, n.hash_prefix) for n in nodes])
            level += 1 
            nodes = self.tree.get(level, [])

    def can_branch(self, node: TreeNode):
        return node.noisy_count > self.branching_threshold
    
    def is_leaf(self, node: TreeNode):
        return (node.noisy_count <= self.branching_threshold) and (node.noisy_count >= self.include_node_threshold)