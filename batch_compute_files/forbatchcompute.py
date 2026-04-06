import numpy as np 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from bucket import create_bucket_synopsis, bucket_using_privacy_accountant, Params
from experiments.evaluation_utils import kmeans_loss
from lloyd import lloyd_with_weights
from grid import  create_grid_synopsis_large
from lloyd import lloyd_with_weights

# d - dimension
# k - number of clusters
# n - number of points
def make_dataset(d: int, k: int, n: int=10000):
    data, _ = make_blobs(n_samples=n, n_features=d, centers=k)
    print(type(data))
    # now normalise it to (-1,1) meaning a radius of 1.4
    scaler = MinMaxScaler((-1,1))
    normalised_data = scaler.fit_transform(data)
    return normalised_data

master_rng = np.random.default_rng(42)

def cluster_grid(data: np.ndarray, k: int, e:float, M:int, seed:int) -> np.ndarray:

    grid_synopsis = create_grid_synopsis_large(data, e, data.shape[1], M, seed)

    centers = lloyd_with_weights(k=k, X=grid_synopsis[:,:-1], weights=grid_synopsis[:,-1], n_iter=10, rs = seed)

    return centers


def grid_experiment(data: np.ndarray, k: int, e:float, n_trials: int) -> float:

    n, d = data.shape
    # try maxing M^d at 100,000
    M = 10
    while M**d > 100000:
        M -= 1
    print("M=", M)
    s = master_rng.integers(low=0, high=100000)
    total_loss = 0

    for x in range(0, 20):
        centers = cluster_grid(data, k, e, M, s + x)
        total_loss += kmeans_loss(centers, data)

    return total_loss / n_trials

print("Done imports and defs")

d_vals = list(range(5, 20, 1)) 

losses_grid = np.zeros((len(d_vals), 2))

for i, d in enumerate(d_vals):
    for j, k in enumerate(k_vals):
        data = make_dataset(d, k)
        loss = grid_experiment(data, k, 1, n_trials=50)
        losses_grid[i, j] = loss

np.save("GRID_RESULTS.npy", losses_grid)