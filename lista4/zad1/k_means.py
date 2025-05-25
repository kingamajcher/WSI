from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# loading data
(X, Y), (_, _) = keras.datasets.mnist.load_data()

# normalising data from (0, 255) to (0.0, 1.0)
X = X / 255.0

# reshaping data to vectors
X = X.reshape(X.shape[0], -1)

# squared Euclidean distance
def squared_euclidean_distance(x, y):
    return np.linalg.norm(x - y) ** 2

# minimum squared Euclidean distance to centroids
def min_squared_euclidean_distance(x, centroids):
    return min(squared_euclidean_distance(x, c) ** 2 for c in centroids)


# initializing centroids
def initialize_centroids(X, k):
    # number of samples, size of each sample
    n_samples, n_features = X.shape
    centroids = []

    # first centroid is random
    idx = np.random.randint(0, n_samples)   
    centroids.append(X[idx])

    for _ in range(1, k):
        distances = np.array([min_squared_euclidean_distance(x, centroids) for x in X])

        probabilities = distances / np.sum(distances)
        idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(X[idx])

    return np.array(centroids)

# k-means algorithm
def k_means(X, k, max_iterations = 100):
    centroids = initialize_centroids(X, k)

    for _ in range(max_iterations):
        distances = np.array([
            [squared_euclidean_distance(x, c) for c in centroids] 
            for x in X
        ])

        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# choose clusters with best inertia
def clusters_with_best_inertia(X, k, max_iterations=100, number_of_runs=10):
    best_inertia = float('inf')
    best_centroids = None
    best_labels = None

    for _ in range(number_of_runs):
        centroids, labels = k_means(X, k, max_iterations)

        inertia = sum(squared_euclidean_distance(X[i], centroids[labels[i]]) for i in range(len(X)))

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels, best_inertia

        

def cluster_distribution_matrix(labels, y_true, k):
    matrix = np.zeros((10, k))
    for true_label, cluster_label in zip(y_true, labels):
        matrix[true_label, cluster_label] += 1

    percent_matrix = (matrix.T / matrix.sum(axis=1)).T * 100
    return percent_matrix


def plot_distribution_matrix(matrix, k, filename=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label="Procent")
    plt.title(f"Macierz procentowa: cyfry vs klastry (k={k})")
    plt.xlabel("Klastry")
    plt.ylabel("Rzeczywiste cyfry")
    plt.xticks(range(matrix.shape[1]))
    plt.yticks(range(10))
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_centroids(centroids, filename=None):
    k = len(centroids)
    n_cols = min(k, 10)
    n_rows = (k + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i, centroid in enumerate(centroids):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(centroid.reshape(28, 28), cmap='gray')
        plt.title(f"Centroid {i}")
        plt.axis('off')
    plt.suptitle("Centroidy klastrów")
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def save_clustering_results(filename, centroids, labels, inertia):
    np.savez(filename, centroids=centroids, labels=labels, inertia=inertia)

def load_clustering_results(filename):
    data = np.load(filename)
    return data['centroids'], data['labels'], data['inertia']

def main():
    for k in [10, 15, 20, 30]:
        print(f"\nKlastry: {k}")
        filename = f"clustering_k_{k}.npz"

        if os.path.exists(filename):
            centroids, labels, inertia = load_clustering_results(filename)
            print(f"Wczytano dane z pliku: {filename}")
        else:
            centroids, labels, inertia = clusters_with_best_inertia(X, k, number_of_runs=3)
            save_clustering_results(filename, centroids, labels, inertia)
            print(f"Zapisano dane do pliku: {filename}")

        print(f"Inercja: {float(inertia):.2f}")

        matrix = cluster_distribution_matrix(labels, Y, k)

        dist_plot_file = f"distribution_k_{k}.png"
        centroids_plot_file = f"centroids_k_{k}.png"
        plot_distribution_matrix(matrix, k, filename=dist_plot_file)
        plot_centroids(centroids, filename=centroids_plot_file)

        print(f"Zapisano wykresy: {dist_plot_file}, {centroids_plot_file}")


if __name__ == "__main__":
    main()