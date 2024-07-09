import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def plot_clusters(keyed_vectors: KeyedVectors, centroids: np.ndarray, clusters: list[list[int]], iteration: int):
    plt.figure(figsize=(10, 7))  # Set the figure size
    colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Define colors for different clusters

    pca = PCA(n_components=2)
    
    for i, cluster in enumerate(clusters):
        points = np.array([keyed_vectors[key] for key in cluster])
        points_2d = pca.fit_transform(points)
        plt.scatter(points_2d[:, 0], points_2d[:, 1], s=30, c=colors[i % len(colors)], label=f'Cluster {i}')
    
    centroids_2d = pca.fit_transform(centroids)
    for centroid in centroids_2d:
        plt.scatter(centroid[0], centroid[1], s=100, c='black', marker='x', linewidths=3)
    
    plt.title(f'Clusters and Centroids ITERATION #{iteration}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.show()


def calculate_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length.")

    squared_diffs = [(vec1[i] - vec2[i]) ** 2 for i in range(len(vec1))]
    distance = np.sqrt(sum(squared_diffs))

    return distance


def is_converged(k: int, old_centroids: np.ndarray, centroids: np.ndarray) -> bool:
    distances = []

    for i in range(k):
        distances.append(calculate_euclidean_distance(old_centroids[i], centroids[i]))

    return sum(distances) == 0


def get_cluster_labels(keyed_vectors: np.ndarray, clusters: list[list[int]]) -> list[int]:
    labels = np.zeros(len(keyed_vectors))

    for i, cluster in enumerate(clusters):
        for j in cluster:
            labels[j] = i

    return labels


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
    node2vec = Node2Vec(G, dimensions=64, walk_length=40, num_walks=20, workers=4)

    print("Training node2vec model...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    print("Generating embeddings...")
    keyed_vectors = model.wv

    return keyed_vectors


def k_means(k: int, raw_data: pd.DataFrame):
    print("Running K-means algorithm")
    keyed_vectors = convert_nodes_to_vectors(raw_data)

    indices = np.random.choice(len(keyed_vectors), k, replace=False)
    centroids = keyed_vectors[indices]
    
    iteration = 1
    while iteration <= 100:
        clusters = create_clusters(k, keyed_vectors, centroids)
        # plot(data, clusters, centroids)

        old_centroids = centroids
        centroids = get_centroids(k, keyed_vectors, clusters)

        if is_converged(k, old_centroids, centroids):
            break
        
        plot_clusters(keyed_vectors, centroids, clusters, iteration)

        cluster_labels = get_cluster_labels(keyed_vectors, clusters)
        print(f"Cluster labels: {cluster_labels}")
        iteration += 1
