from symnmf import proccess_input_file, init_H
import symnmf_module
from sklearn.metrics import silhouette_score
import numpy as np
import sys

EPSILON = 1e-4
MAX_ITER = 300

np.random.seed(1234)


def symnmf(k, X):
	"""
	Perform the symnmf algorithm
	k: number of clusters
	X: input matrix
	return: symnmf algorithm result
	"""
	W = symnmf_module.norm(X)
	H = init_H(W, k)
	result = symnmf_module.symnmf(H, W)

	return result


def get_distance(list1, list2):
	"""
	Returns the distance between two vectors
	list1: vector
	list2: vector
	return: distance
	"""
	return sum([(list1[i] - list2[i]) ** 2 for i in range(len(list1))]) ** 0.5


def kmeans_step(X, k, n, d, centroids, value_to_centroid, nodes_in_centroid):
	"""
	Perform a step in the kmeans algorithm
	X: input matrix
	k: number of clusters
	n: number of points
	d: size of points
	centroids, value_to_centroid, nodes_in_centroid: kmeans lists
	return: kmeans lists and max_change
	"""
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
		if nodes_in_centroid[i] != 0:
			centroids[i] = [j / nodes_in_centroid[i] for j in centroids[i]]
		else:
			centroids[i] = old_centroids[i]

	# Get the max change
	max_change = max([get_distance(old_centroids[i], centroids[i]) for i in range(k)])

	return centroids, value_to_centroid, nodes_in_centroid, max_change


def kmeans(k, X):
	"""
	Perform the kmeans algorithm
	k: number of clusters
	X: input matrix
	return: kmeans algorithm result
	"""
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
		# Perform a step in the kmeans algorithm
		centroids, value_to_centroid, nodes_in_centroid, max_change = kmeans_step(X, k, n, d, centroids, value_to_centroid, nodes_in_centroid)
		count += 1

	return centroids


def labels_from_symnmf(results):
	"""
	Get labels from symnmf results
	results: symnmf algorithm results
	return: labels
	"""
	labels = []

	for row in results:
		labels.append(row.index(max(row)))

	return labels


def labels_from_kmeans(X, results):
	"""
	Get labels from kmeans results
	X: input matrix
	results: kmeans algorithm results
	return: labels
	"""
	labels = []

	for point in X:
		distances = [get_distance(point, cluster) for cluster in results]
		labels.append(distances.index(min(distances)))

	return labels


def main():
	# Check correct number of args
	if len(sys.argv) != 3:
		print("Usage: python symnmf.py <k> <file_name>")
		sys.exit(1)

	# Get args
	k = int(float(sys.argv[1]))
	file_name = sys.argv[2]

	# Get matrix from input file
	X = proccess_input_file(file_name)

	# Perform both algorithms
	try:
		symnmf_results = symnmf(k, X)
	except RuntimeError as e:
		print(e)
		sys.exit(1)
	kmeans_results = kmeans(k, X)

	# Get labels from algorithm results
	symnmf_labels = labels_from_symnmf(symnmf_results)
	kmeans_labels = labels_from_kmeans(X, kmeans_results)

	# Calculate silhouette scores for both algorithms
	symnmf_score = silhouette_score(X, symnmf_labels)
	kmeans_score = silhouette_score(X, kmeans_labels)

	# Print scores
	print(f"nmf: {symnmf_score:.4f}")
	print(f"kmeans: {kmeans_score:.4f}")


if __name__ == '__main__':
    main()
