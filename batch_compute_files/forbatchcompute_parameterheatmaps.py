import numpy as np 
from bucket import create_bucket_synopsis, bucket_using_privacy_accountant, Params
from experiments.evaluation_utils import kmeans_loss
from lloyd import lloyd_with_weights

master_rng = np.random.default_rng(42)

def lsh_experiment(algo: int, data: np.ndarray, p: Params, n_trials: int = 20, privacy_split = 0.8):
    s = master_rng.integers(low=0, high=100000)
    total_loss = 0
    n_successful_trials = n_trials
    for x in range(n_trials):
        if algo == 1:
            print("starting synopsis... ")
            private_points, private_weights = create_bucket_synopsis(data, p, s+x, privacy_split=privacy_split, use_gaussian=True)
        else:
            private_points, private_weights = bucket_using_privacy_accountant(data, p, s+x)
        if private_points.shape[0] <= p.k: # if number of points is less than or equal to desired number of centers
            centers = private_points
        else:
            centers = lloyd_with_weights(k=p.k, X=private_points, weights=private_weights, n_iter=5, rs=s+x)
        try:
            loss = kmeans_loss(centers, data)
        except:
            loss = 0
            n_successful_trials -=1
        total_loss += loss
        print(f"Trial {x+1} done")
    print("Number completed trials: ", n_successful_trials)
    return total_loss / n_successful_trials

large = np.load("datasets/large-synthetic.npy")

include_values = list(range(200,1000,50)) 
depth_values = list(range(5,23,1))

loss_grid = np.zeros((len(depth_values), len(include_values)))

for i, depth in enumerate(depth_values):
    for j, include in enumerate(include_values):
        p = Params(epsilon=1, delta=1e-6, radius=4.7, dimension=100, k=10, max_depth=depth, include_threshold=include)
        loss = lsh_experiment(1, large, p, n_trials=50)
        loss_grid[i, j] = loss

np.save("results/lsh_heatmap_large.npy", loss_grid)

forest = np.load("datasets/forest.npy")

loss_grid = np.zeros((len(depth_values), len(include_values)))

for i, depth in enumerate(depth_values):
    for j, include in enumerate(include_values):
        p = Params(epsilon=1, delta=1e-6, radius=2.4, dimension=9, k=10, max_depth=depth, include_threshold=include)
        loss = lsh_experiment(1, forest, p, n_trials=50)
        loss_grid[i, j] = loss

np.save("results/lsh_heatmap_forest.npy", loss_grid)

