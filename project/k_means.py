import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def convert_to_dataframe(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as file:
        raw_data = file.readlines()

    data = {"from-node-id": [], "to-node-id": []}

    for row in raw_data[4:]:
        data["from-node-id"].append(row.split("\t")[0].strip())
        data["to-node-id"].append(row.split("\t")[1].strip())

    return pd.DataFrame(data)


def plot_clusters(
    k: int,
    points: np.ndarray,
    centroids: np.ndarray,
    clusters: list[list[int]],
    iteration: int,
):
    plt.figure(figsize=(10, 7))
    colors = plt.cm.hsv(np.linspace(0, 1, k + 1))

    for i, cluster in enumerate(clusters):
        cluster_points = np.array([points[key] for key in cluster])
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=30,
            color=colors[i % k],
            label=f"Cluster {i}",
        )

    for centroid in centroids:
        plt.scatter(
            centroid[0], centroid[1], s=100, color="black", marker="x", linewidths=3
        )

    plt.title(f"Clusters and Centroids ITERATION #{iteration}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()


def calculate_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


def is_converged(
    k: int, old_centroids: np.ndarray, current_centroids: np.ndarray
) -> bool:
    distances = [
        calculate_euclidean_distance(old_centroids[i], current_centroids[i])
        for i in range(k)
    ]
    sum_distances = sum(distances)
    print(f"Sum of cluster movement: {sum_distances}\n")
    return sum_distances <= 0.001


def calculate_new_centroids(
    k: int, points: np.ndarray, clusters: list[list[int]]
) -> np.ndarray:
    centroids = np.zeros((k, 2))

    for i, cluster in enumerate(clusters):
        cluster_vectors = np.array([points[key] for key in cluster])
        cluster_mean = np.mean(cluster_vectors, axis=0)
        centroids[i] = cluster_mean

    return centroids


def create_clusters(
    k: int, points: np.ndarray, centroids: np.ndarray
) -> list[list[int]]:
    clusters = [[] for _ in range(k)]

    for i, point in enumerate(points):
        closest_centroid = np.argmin(
            [calculate_euclidean_distance(point, centroid) for centroid in centroids]
        )
        clusters[closest_centroid].append(i)

    return clusters


def convert_nodes_to_vectors(raw_data: pd.DataFrame) -> KeyedVectors:
    print("Creating digraph...")
    G = nx.from_pandas_edgelist(
        raw_data, "from-node-id", "to-node-id", create_using=nx.DiGraph()
    )

    print("Initializing node2vec model...")
    node2vec = Node2Vec(G, dimensions=64, walk_length=50, num_walks=50, workers=32)

    print("Training node2vec model...")
    model = node2vec.fit(window=5, min_count=1, batch_words=10000)

    print("Generating vector embeddings...")
    keyed_vectors = model.wv

    return keyed_vectors


def convert_vectors_to_2d(keyed_vectors: KeyedVectors):
    breakpoint()
    vectors_array = np.array([v for v in keyed_vectors.vectors])
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors_array)


def get_cluster_labels(clusters, num_samples) -> np.ndarray:
    labels = np.zeros(num_samples, dtype=int)

    for cluster_number, cluster_indices in enumerate(clusters):
        for index in cluster_indices:
            labels[index] = cluster_number

    return labels


def k_means(
    k: int, raw_data: pd.DataFrame
) -> tuple[KeyedVectors, list[list[int]], np.ndarray]:
    print("Running K-means algorithm")
    start_time = time.time()
    points = convert_vectors_to_2d(convert_nodes_to_vectors(raw_data))
    breakpoint()
    indices = np.random.choice(len(points), k, replace=False)
    current_centroids = points[indices]

    iteration = 1
    while iteration <= 1000:
        print(f"Iteration #{iteration}")
        clusters = create_clusters(k, points, current_centroids)

        old_centroids = current_centroids
        current_centroids = calculate_new_centroids(k, points, clusters)

        if is_converged(k, old_centroids, current_centroids):
            break

        iteration += 1

    end_time = time.time()
    wcss = calculate_wcss(k, points, clusters, current_centroids)
    silhouette_avg = silhouette_score(points, get_cluster_labels(clusters, len(points)))
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"WCSS = {wcss}")
    print(f"Silhouette Score: {silhouette_avg}")

    plot_clusters(k, points, current_centroids, clusters, iteration)

    return wcss


def calculate_wcss(
    k: int,
    keyed_vectors: KeyedVectors,
    clusters: list[list[int]],
    centroids: np.ndarray,
) -> float:
    wcss = 0

    for cluster_id in range(k):
        cluster_indices = clusters[cluster_id]
        cluster_vectors = keyed_vectors[cluster_indices]

        for vector in cluster_vectors:
            squared_distance = (
                calculate_euclidean_distance(vector, centroids[cluster_id]) ** 2
            )
            wcss += squared_distance

    return wcss


def plot_optimal_k(k_range: range, values: list[list[int]], method: str):
    plt.plot(k_range, values, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(method)
    plt.title("Find Optimal k")
    plt.xticks(range(min(k_range), max(k_range) + 1))
    plt.show()


def find_optimal_k(data: pd.DataFrame, k_range: range):
    wcss_values = []

    for k in k_range:
        print(f"k = {k}")
        wcss = k_means(k, data)
        wcss_values.append(wcss)

    plot_optimal_k(k_range, wcss_values, "WCSS")


if __name__ == "__main__":
    test_data = convert_to_dataframe("./datasets/test_data.txt")
    arxiv_small_dataset = convert_to_dataframe("./datasets/CA-GrQc.txt")
    dblp_large_dataset = convert_to_dataframe("./datasets/com-dblp.ungraph.txt")

    if input("Find optimal k?\nInput (y/n): ") in ["yes", "y"]:
        find_optimal_k(arxiv_small_dataset, range(3, 15))
    else:
        # k_means(5, arxiv_small_dataset)
        k_means(3, arxiv_small_dataset)
        # dblp_large_dataset
        # k_means(12, dblp_large_dataset)
