import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors
import time

import matplotlib

matplotlib.use("TkAgg")
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
    keyed_vectors: KeyedVectors,
    centroids: np.ndarray,
    clusters: list[list[int]],
    iteration: int,
):
    plt.figure(figsize=(10, 7))
    colors = plt.cm.hsv(np.linspace(0, 1, k + 1))

    all_vectors = np.array(
        [keyed_vectors[key] for cluster in clusters for key in cluster]
    )

    pca = PCA(n_components=2)
    pca.fit(all_vectors)

    for i, cluster in enumerate(clusters):
        points = np.array([keyed_vectors[key] for key in cluster])
        points_2d = pca.transform(points)
        plt.scatter(
            points_2d[:, 0],
            points_2d[:, 1],
            s=30,
            color=colors[i % k],
            label=f"Cluster {i}",
        )

    centroids_2d = pca.transform(centroids)
    for centroid in centroids_2d:
        plt.scatter(
            centroid[0], centroid[1], s=100, color="black", marker="x", linewidths=3
        )

    plt.title(f"Clusters and Centroids ITERATION #{iteration}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()


def calculate_squared_diffs(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return [(vec1[i] - vec2[i]) ** 2 for i in range(len(vec1))]


def calculate_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.sqrt(sum(calculate_squared_diffs(vec1, vec2)))


def is_converged(
    k: int, old_centroids: np.ndarray, current_centroids: np.ndarray
) -> bool:
    distances = [
        calculate_euclidean_distance(old_centroids[i], current_centroids[i])
        for i in range(k)
    ]
    sum_distances = sum(distances)
    # print(f"Sum of cluster movement: {sum_distances}\n")
    return sum_distances <= 0.001


def get_centroids(
    k: int, keyed_vectors: KeyedVectors, clusters: list[list[int]]
) -> np.ndarray:
    vector_dimensions = keyed_vectors.vector_size
    centroids = np.zeros((k, vector_dimensions))

    for i, cluster in enumerate(clusters):
        cluster_vectors = np.array([keyed_vectors[key] for key in cluster])
        cluster_mean = np.mean(cluster_vectors, axis=0)
        centroids[i] = cluster_mean

    return centroids


def create_clusters(
    k: int, keyed_vectors: KeyedVectors, centroids: np.ndarray
) -> list[list[int]]:
    clusters = [[] for _ in range(k)]

    for i, vector in enumerate(keyed_vectors.vectors):
        closest_centroid = np.argmin(
            [calculate_euclidean_distance(vector, centroid) for centroid in centroids]
        )
        clusters[closest_centroid].append(i)

    return clusters


def convert_nodes_to_vectors(raw_data: pd.DataFrame) -> KeyedVectors:
    print("Creating digraph...")
    G = nx.from_pandas_edgelist(
        raw_data, "from-node-id", "to-node-id", create_using=nx.DiGraph()
    )

    print("Initializing node2vec model...")
    node2vec = Node2Vec(G, dimensions=64, walk_length=100, num_walks=100, workers=32)

    print("Training node2vec model...")
    model = node2vec.fit(window=5, min_count=1, batch_words=10000)

    print("Generating vector embeddings...")
    keyed_vectors = model.wv

    return keyed_vectors


def get_cluster_labels(clusters, num_samples):
    labels = np.zeros(num_samples, dtype=int)

    for cluster_number, cluster_indices in enumerate(clusters):
        for index in cluster_indices:
            labels[index] = cluster_number

    return labels



def k_means(k: int, raw_data: pd.DataFrame):
    print("Running K-means algorithm")
    start_time = time.time()
    keyed_vectors = convert_nodes_to_vectors(raw_data)

    indices = np.random.choice(len(keyed_vectors), k, replace=False)
    current_centroids = keyed_vectors[indices]

    iteration = 1
    while iteration <= 1000:
        # print(f"Iteration #{iteration}")
        clusters = create_clusters(k, keyed_vectors, current_centroids)

        old_centroids = current_centroids
        current_centroids = get_centroids(k, keyed_vectors, clusters)

        if is_converged(k, old_centroids, current_centroids):
            break

        # plot_clusters(k, keyed_vectors, current_centroids, clusters, iteration)
        iteration += 1

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    plot_clusters(k, keyed_vectors, current_centroids, clusters, iteration)

    return keyed_vectors, clusters, current_centroids


def calculate_wcss(k, keyed_vectors, clusters, centroids):
    wcss = 0

    for cluster_id in range(k):
        cluster_indices = clusters[cluster_id]
        cluster_vectors = keyed_vectors[cluster_indices]

        for vector in cluster_vectors:
            squared_distance = calculate_euclidean_distance(vector, centroids[cluster_id]) ** 2
            wcss += squared_distance

    return wcss


def plot_optimal_k(k_range, values, method):
    plt.plot(k_range, values, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(method)
    plt.title('Elbow Method for Optimal k')
    plt.xticks(range(min(k_range), max(k_range)+1))
    plt.show()


def find_optimal_k(data, k_range):
    wcss_values, silh_values = [], []

    for k in k_range:
        print(f"k = {k}")
        keyed_vectors, clusters, current_centroids = k_means(k, data)

        wcss = calculate_wcss(k, keyed_vectors, clusters, current_centroids)
        silhouette_avg = silhouette_score(keyed_vectors.vectors, get_cluster_labels(clusters, len(keyed_vectors)))
        print(f"WCSS = {wcss}")
        print(f"Silhouette Score: {silhouette_avg}")

        wcss_values.append(wcss)
        silh_values.append(silhouette_avg)
        # k_means(3, arxiv_small_dataset)
        # k_means(10, dblp_large_dataset)

    plot_optimal_k(k_range, wcss_values, "WCSS")
    plot_optimal_k(k_range, silh_values, "Silhouette Score")



if __name__ == "__main__":
    test_data = convert_to_dataframe("./datasets/test_data.txt")
    arxiv_small_dataset = convert_to_dataframe("./datasets/CA-GrQc.txt")
    dblp_large_dataset = convert_to_dataframe("./datasets/com-dblp.ungraph.txt")

    
    if input("Find optimal k?\nInput (y/n): ") in ["yes", "y"]:
        find_optimal_k(arxiv_small_dataset, range(3, 34))
    else:
        # k_means(34, arxiv_small_dataset)
        k_means(3, test_data)
        # k_means(10, dblp_large_dataset)