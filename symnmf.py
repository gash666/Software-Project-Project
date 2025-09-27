import sys
import symnmf_module
import numpy as np
import math

np.random.seed(1234)


def init_H(W, k):
    n = len(W)
    
    m = 0
    for row in W:
        for value in row:
            m += value
    m /= n * k

    H = np.zeros((n, k), dtype=float)

    for i in range(n):
        for j in range(k):
            H[i, j] = np.random.uniform(0, 2 * math.sqrt(m / k))
    
    return H


def proccess_input_file(file_name):
    try:
        f = open(file_name, "r")
    except:
        print("An Error Has Occurred")
        sys.exit(1)
    
    lines = f.readlines()

    X = []

    for line in lines:
        if line == "\n":
            break
        
        row = [float(value) for value in line[:-1].split(",")]
        X.append(row)
    
    return X


def print_matrix(A):
    for row in A:
        print(",".join([f"{value:.4f}" for value in row]))


def main():
    if len(sys.argv) != 4:
        print("Usage: python symnmf.py <k> <goal> <file_name>")
        sys.exit(1)
    
    k = int(float(sys.argv[1]))
    goal = sys.argv[2]
    file_name = sys.argv[3]

    X = proccess_input_file(file_name)
    result = None

    if goal == "symnmf":
        W = symnmf_module.norm(X)
        H = init_H(W, k)
        result = symnmf_module.symnmf(H, W)
    elif goal == "sym":
        result = symnmf_module.sym(X)
    elif goal == "ddg":
        result = symnmf_module.ddg(X)
    elif goal == "norm":
        result = symnmf_module.norm(X)
    else:
        print("An Error Has Occurred")
        sys.exit(1)

    print_matrix(result)


if __name__ == '__main__':
    main()