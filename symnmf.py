import sys
import symnmf_module
import numpy as np
import math

np.random.seed(1234)


def init_H(W, k):
    """
    Initialize a random matrix H, of size n x k
    W: normalized similarity matrix
    k: number of clusters
    return: H
    """
    n = len(W)

    # Calculate average entry of W
    m = 0
    for row in W:
        for value in row:
            m += value
    m /= n * n

    # Initialize H with random values between 0 and 2 * math.sqrt(m / k)
    H = np.zeros((n, k), dtype=float)
    for i in range(n):
        for j in range(k):
            H[i, j] = np.random.uniform(0, 2 * math.sqrt(m / k))

    return H


def proccess_input_file(file_name):
    """
    Proccess and input file
    file_name: file to load input from
    return: X, matrix read from file
    """
    # Open the file and check for errors
    try:
        f = open(file_name, "r")
    except:
        print("An Error Has Occurred")
        sys.exit(1)

    # Read from the file
    lines = f.readlines()

    # Initialize the X matrix
    X = []

    # Go through each line in the file (each line is a different data point)
    for line in lines:
        # Check if end of line and need to stop
        if line == "\n":
            break

        # Get the values from the given format and save it as a data point
        row = [float(value) for value in line[:-1].split(",")]
        X.append(row)

    return X


def print_matrix(A):
    """
    Print matrix A
    A: matrix to print
    """
    for row in A:
        print(",".join([f"{value:.4f}" for value in row]))


def main():
    # Check if the number of arguments is correct
    if len(sys.argv) != 4:
        print("Usage: python symnmf.py <k> <goal> <file_name>")
        sys.exit(1)

    # Load variables from args
    k = int(float(sys.argv[1]))
    goal = sys.argv[2]
    file_name = sys.argv[3]

    # Get matrix from file
    X = proccess_input_file(file_name)
    result = None

    # Calculate the result based on the value of goal
    try:
        if goal == "symnmf":
            # Calculate symNMF and output the final result for H
            W = symnmf_module.norm(X)
            H = init_H(W, k)
            result = symnmf_module.symnmf(H, W)
        elif goal == "sym":
            # Calculate the similarity matrix A
            result = symnmf_module.sym(X)
        elif goal == "ddg":
            # Calculate the diagonal degree matrix D
            result = symnmf_module.ddg(X)
        elif goal == "norm":
            # Calculate the normalized similarity matrix W
            result = symnmf_module.norm(X)
        else:
            # Print error message and exit
            print("An Error Has Occurred")
            sys.exit(1)
    except RunRuntimeError as e:
		print(e)
		sys.exit(1)

    # Print the result
    print_matrix(result)


if __name__ == '__main__':
    main()
