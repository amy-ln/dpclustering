import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from typing import Optional

# TO IMPROVE
# not sure adding privacy correctly - how to split up privacy budget? how much noise should we add to the noisy counts?
# is the tree splitting correctly? observed many leaf nodes with large numbers of points. does normalisation affect this? 

# use this from the diffprivlibrary for hashing 
from lsh import SimHash
from util import noise, normalise

def create_bucket_synopsis(X: pd.DataFrame, e:float, d:int, branching_threshold: int, max_depth: int):

    # give half the privacy budget to computing the tree and half to computing weighted averages of points?
    e1, e2 = e/2, e/2 

    # create tree : return leaf nodes pointing to all points "in" that node
    tree = LshTree(e1, branching_threshold, max_depth, X, X.shape[1])
    leaves = tree.get_leaves()
    print("Leaves")
    print(leaves)

    # use leaf nodes to create the weighted points 
    rows = []
    for leaf in leaves:
        # assume the sum has a sensitivity of 1*d because data is normalised
        coords, weight = leaf
        average = ((coords).sum() + noise(d / e2, d)) / weight
        
        row = list(average) + [weight]
        rows.append(row)

    weighted_points = pd.DataFrame(rows)

    return weighted_points

class LshTree:

    def __init__(self, e, branching_threshold, max_depth, X, dimension):
        self.e_per_layer = e / max_depth
        self.branching_threshold = branching_threshold
        self.max_depth = max_depth
        self.hash = SimHash(dimension, max_depth)
        self.tree = self.create_lsh_tree(X)

    def branch(self, points, depth: int, hash_prefix: str):

        ## need to add noise to the counts of points here to satisfy privacy
        noisy_count = points.size + noise(1/self.e_per_layer, d=1)[0]
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

        tree = self.branch(X, 0, "")

        return tree


"""
X1 = np.random.multivariate_normal(mean=(5,10), cov=[[5,0],[0,5]], size=20)
X2 = np.random.multivariate_normal(mean=(2,3), cov=[[5,0],[0,5]], size=30)
X = normalise(pd.DataFrame(np.concatenate((X1, X2))))

create_bucket_synopsis(X, 1, 2, 5, 5)
"""