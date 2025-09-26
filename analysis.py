from symnmf import proccess_input_file, generate_initial_H, symnmf_c
from sklearn.metrics import silhouette_score
import numpy as np

EPSILON = 1e-4
MAX_ITER = 300

np.random.seed(1234)


def symnmf(k, X):
    W = symnmf_c.norm(X)
    H = generate_initial_H(W, k)
    result = symnmf_c.symnmf(H, W)

    return result


def get_distance(list1, list2):
	# Returns the distance between two vectors
	return sum([(list1[i] - list2[i]) ** 2 for i in range(len(list1))]) ** 0.5


def is_good(number):
	# Returns true if the string input represents a whole number
	try:
		num = float(number)
		return num.is_integer()
	except ValueError:
		return False


def kmeans(k, X):
    n = len(X)
    d = len(X[0])

    # Initialize the centroids and structures / variables to support the algorithm
	centroids = [X[i].copy() for i in range(k)]
	value_to_centroid = {}
	nodes_in_centroid = [0 for i in range(k)]
	count = 0
	max_change = 1

	# Run the algorithm
	while count < MAX_ITER and max_change > EPSILON:
		# Save the closest centroid for each node
		for i in range(n):
			distances = [get_distance(X[i], centroids[j]) for j in range(k)]
			value_to_centroid[i] = distances.index(min(distances))

		# Save the current values of the centroids
		old_centroids = centroids.copy()

		# Initialize the centroid values before setting them to the new ones
		for i in range(k):
			centroids[i] = [0 for j in range(d)]
			nodes_in_centroid[i] = 0

		# calculate the new centroid values
		for i in range(n):
			nodes_in_centroid[value_to_centroid[i]] += 1
			for j in range(d):
				centroids[value_to_centroid[i]][j] += X[i][j]

		# Divide by the number of data points in each centroid
		for i in range(k):
			centroids[i] = [j / nodes_in_centroid[i] for j in centroids[i]]

		# Get the max change
		max_change = max([get_distance(old_centroids[i], centroids[i]) for i in range(k)])
		count += 1
    
    return centroids


def labels_from_symnmf(results):
    labels = []

    for row in results:
        labels.append(row.index(max(row)))
    
    return labels


def labels_from_kmeans(X, results):
    labels = []

    for point in X:
        distances = [distance(point, cluster) for cluster in results]
        labels.append(distances.index(min(distances)))
    
    return labels


def main():
    if len(sys.argv) != 3:
        print("Usage: python symnmf.py <k> <file_name>")
        sys.exit(1)
    
    k = int(float(sys.argv[1]))
    file_name = sys.argv[2]

    X = proccess_input_file(file_name)

    symnmf_results = symnmf(k, X)
    kmeans_results = kmeans(k, X)

    symnmf_labels = labels_from_symnmf(X, symnmf_restuls)
    kmeans_labels = labels_from_kmeans(X, kmeans_restuls)

    symnmf_score = silhouette_score(X, symnmf_labels)
    kmeans_score = silhouette_score(X, kmeans_labels)

    print(f"nmf: {symnmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")


if __name__ == '__main__':
    main()
