import numpy as np
import pandas as pd
from node2vec import Node2Vec


def is_converged(k: int, old_centroids: np.ndarray, centroids: np.ndarray) -> bool:
    distances = []

    for i in range(k):
        distances.append(calculate_euclidean_distance(old_centroids[i], centroids[i]))

    return sum(distances) == 0


def calculate_euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sqrt(np.sum((x1 - x2) ** 2))


# def transform_directed_graph_to_adjacency_matrix(data: pd.DataFrame) -> np.ndarray:
#     graph = nx.from_pandas_edgelist(
#         data, "from-node-id", "to-node-id", create_using=nx.DiGraph()
#     )
#     return nx.adjacency_matrix(graph).todense()


def get_cluster_labels(data: np.ndarray, clusters: list[list[int]]) -> list[int]:
    labels = np.zeros(data.shape[0])

    for i, cluster in enumerate(clusters):
        for j in cluster:
            labels[j] = i

    return labels


def get_centroids(
    k: int, data: np.ndarray, clusters: list[list[int]]
) -> list[np.ndarray]:
    centroids = np.zeros((k, data.shape[1]))

    for i, cluster in enumerate(clusters):
        cluster_mean = np.mean(data[cluster], axis=0)
        centroids[i] = cluster_mean

    return centroids


def create_clusters(k: int, data: np.ndarray, centroids: list[np.ndarray]) -> list[list[int]]:
    clusters = [[] for _ in range(k)]

    for i, sample in enumerate(data):
        closest_centroid = np.argmin(
            [calculate_euclidean_distance(sample, centroid) for centroid in centroids]
        )
        clusters[closest_centroid].append(i)

    return clusters


# def plot(data: np.ndarray, clusters: list[list[int]], centroids: list[np.ndarray]):
#     print("creating graph")
#     G = nx.DiGraph(data)
#     print("drawing...")
#     nx.draw(G)
#     print("showing")
#     plt.show()
#     breakpoint()


def convert_nodes_too_vectors(raw_data: pd.DataFrame):
    pass

def k_means(k: int, raw_data: pd.DataFrame):
    print("Running K-means algorithm")
    print(raw_data)
    data = convert_nodes_too_vectors(raw_data)

    breakpoint()
    # data = transform_directed_graph_to_adjacency_matrix(raw_data)

    # number_of_samples, number_of_features = data.shape

    random_centroid_indexes = np.random.choice(data.shape[0], k, replace=False)
    centroids = [data[i] for i in random_centroid_indexes]

    while True:
        clusters = create_clusters(k, data, centroids)

        # plot(data, clusters, centroids)

        old_centroids = centroids
        centroids = get_centroids(k, data, clusters)

        if is_converged(k, old_centroids, centroids):
            break

        # plot(clusters, centroids)

        cluster_labels = get_cluster_labels(data, clusters)

        print(f"Cluster labels: {cluster_labels}")