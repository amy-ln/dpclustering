import numpy as np 
from bucket import create_bucket_synopsis, bucket_using_privacy_accountant, Params
from experiments.evaluation_utils import kmeans_loss
from lloyd import lloyd_with_weights, dplloyd, PrivacyBudget
from grid import create_grid_synopsis_large
from sklearn.cluster import KMeans

master_rng = np.random.default_rng(42)

def lsh_experiment(algo: int, data: np.ndarray, p: Params, n_trials: int = 20):
    s = master_rng.integers(low=0, high=100000)
    total_loss = 0
    n_successful_trials = n_trials
    for x in range(n_trials):
        if algo == 1:
            print("starting synopsis... ")
            private_points, private_weights = create_bucket_synopsis(data, p, s+x)
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

# one function to apply grid synopsis and then non-private kmeans
def cluster_grid(data: np.ndarray, k: int, e:float, M:int, seed:int) -> np.ndarray:

    grid_synopsis = create_grid_synopsis_large(data, e, data.shape[1], M, seed)

    centers = lloyd_with_weights(k=k, X=grid_synopsis[:,:-1], weights=grid_synopsis[:,-1], n_iter=10, rs = seed)

    return centers

def grid_experiment(data: np.ndarray, k: int, e:float, M:int, n_trials: int = 20) -> float:

    s = master_rng.integers(low=0, high=100000)
    total_loss = 0

    for x in range(0, n_trials):
        centers = cluster_grid(data, k, e, M, s + x)
        total_loss += kmeans_loss(centers, data)

    return total_loss / n_trials

def dplloyd_experiment(X, k, epsilon, method, iterations, trials=20, output=False):
    p = PrivacyBudget(epsilon=epsilon, method=method, total_iter=iterations)
    # do 20 randomised trials
    base_seed = master_rng.integers(low=0, high=100000)
    trials = [dplloyd(k=k, X=X, n_iter=iterations, priv=p, seed=base_seed + x) for x in range(20)]
    losses = [kmeans_loss(centers, X) for centers in trials]
    avg_loss = np.mean(losses)
    if output: 
        print(f"base seed={base_seed}, average loss={avg_loss}")
    return avg_loss

def non_private_radius(x: np.ndarray) -> float:
    return np.max(np.linalg.norm(x, axis=-1))



"""LOAD DATASETS"""
small = np.load("datasets/synthetic-gaussian.npy")
print(small.shape)
airports = np.load("datasets/airports.npy")
print(airports.shape)
large = np.load("datasets/large-synthetic.npy")
print(large.shape)
concrete = np.load("datasets/concrete.npy")
print(concrete.shape)
forest = np.load("datasets/forest.npy")
print(forest.shape)


"""EXPERIMENT FUNCTION"""

def get_results(data: np.ndarray, ks: list, lloyd_iters: int, M: int, num_randomised_trials: int = 50):
    dpl = []
    grid = []
    lsh_mine = []
    lsh_google = []

    for k in ks:
        dpl.append(dplloyd_experiment(X=data, k=k, epsilon=1, method="uniform", iterations=lloyd_iters, trials=num_randomised_trials))
        grid.append(grid_experiment(data, k=k, e=1, M=M, n_trials=num_randomised_trials))
        p = Params(epsilon=1, delta=1e-6, radius=non_private_radius(data), dimension=data.shape[1], k=k, max_depth=15)
        lsh_mine.append(lsh_experiment(algo=1, data=data, p=p, n_trials=num_randomised_trials))
        lsh_google.append(lsh_experiment(algo=2, data=data, p=p, n_trials=num_randomised_trials))

    array = np.array([dpl, grid, lsh_mine, lsh_google])
    return array

small_results = get_results(small, list(range(1,8)), lloyd_iters=1, M=7)
np.save("results/small.npy", small_results)

airports_results = get_results(airports, list(range(1,11)), lloyd_iters=1, M=20)
np.save("results/airports.npy", airports_results)

concrete_results = get_results(concrete, list(range(1,16)), lloyd_iters=2, M=3)
np.save("results/concrete.npy", concrete_results)

forest_results = get_results(forest, ks = list(range(1,11)), lloyd_iters=4, M=3)
np.save("results/forest.npy", forest_results)

large_results = get_results(large, list(range(1,21)), lloyd_iters=5, M=1)
np.save("results/large.npy", large_results)


