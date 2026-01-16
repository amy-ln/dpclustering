import numpy as np

def kmeans_loss(clusters: np.ndarray, datapoints: np.ndarray) -> float:
    # compute the difference between each point and each cluster center
    diffs = datapoints[:, None, :] - clusters[None, :, :]
    # sum and square these 
    distances = np.sum(diffs ** 2, axis=2)
    # take the minimum distance for each point 
    min_distances = np.min(distances, axis=1)
    # take the average
    return np.mean(min_distances)

