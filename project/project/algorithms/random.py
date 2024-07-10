import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_clusters(
    k: int,
    keyed_vectors: KeyedVectors,
    centroids: np.ndarray,
    clusters: list[list[int]],
    iteration: int,
):
    plt.figure(figsize=(10, 7))
    colors = plt.cm.hsv(np.linspace(0, 1, k))

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
            centroid[0], centroid[1], s=100, color="black", marker="x", linewidths=2
        )

    plt.title(f"Clusters and Centroids ITERATION #{iteration}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()


def calculate_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.sqrt(sum([(vec1[i] - vec2[i]) ** 2 for i in range(len(vec1))]))


def is_converged(k: int, centroids1: np.ndarray, centroids2: np.ndarray) -> bool:
    distances = [
        calculate_euclidean_distance(centroids1[i], centroids2[i]) for i in range(k)
    ]
    sum_distances = sum(distances)
    print(sum_distances)
    return sum_distances == 0


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
    node2vec = Node2Vec(G, dimensions=64, walk_length=60, num_walks=40, workers=32)

    print("Training node2vec model...")
    model = node2vec.fit(window=10, min_count=1, batch_words=1000)

    print("Generating embeddings...")
    keyed_vectors = model.wv

    return keyed_vectors


def get_cluster_labels(
    keyed_vectors: np.ndarray, clusters: list[list[int]]
) -> list[int]:
    labels = np.zeros(len(keyed_vectors))

    for i, cluster in enumerate(clusters):
        for j in cluster:
            labels[j] = i

    return labels


def k_means(k: int, raw_data: pd.DataFrame):
    print("Running K-means algorithm")

    keyed_vectors = convert_nodes_to_vectors(raw_data)
    indicies = np.random.choice(len(keyed_vectors), k, replace=False)
    centroids = keyed_vectors[indicies]

    iteration = 1
    while iteration <= 1000:
        clusters = create_clusters(k, keyed_vectors, centroids)

        old_centroids = centroids
        new_centroids = get_centroids(k, keyed_vectors, clusters)

        if is_converged(k, old_centroids, new_centroids):
            break

        print(f"Cluster labels #{iteration}")
        iteration += 1

    plot_clusters(k, keyed_vectors, centroids, clusters, iteration)
