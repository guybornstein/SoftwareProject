import pandas as pd
import numpy as np
import sys


DEFAULT_MAX_ITER = 200


def invalid_input():
    print("Invaid Input!")
    sys.exit(1)


def exception_handler():
    print("An Error Has Occurred")
    sys.exit(1)


def kmeans_pp(datapoints, k):
    n, m = datapoints.shape
    if k >= n:
        invalid_input()
    
    np.random.seed(0)
    probs = np.ones(n) / n
    starting_centroids = np.empty((k, m))
    dists = np.empty(n)

    starting_centroids[0] = datapoints[np.random.choice(n, 1, p=probs)]
    for i in range(1, k):
        for l in range(n):
            dists[l] = min([np.vdot(datapoints[l] - starting_centroids[j], datapoints[l] - starting_centroids[j]) for j in range(i)])
        probs = dists / np.sum(dists)
        starting_centroids[i] = datapoints[np.random.choice(n, 1, p=probs)]


    return starting_centroids


def main():
    # parse commandline arguments
    if len(sys.argv) == 6:
        _, k, max_iter, epsilon, input_path1, input_path2 = sys.argv
    elif len(sys.argv) == 5:
        _, k, epsilon, input_path1, input_path2 = sys.argv
        max_iter = str(DEFAULT_MAX_ITER)
    else:
        invalid_input()
    
    if not k.isdigit() or not max_iter.isdigit():
        invalid_input()

    try:
       epsilon = float(epsilon)
    except ValueError:
        invalid_input()

    
    k = int(k)
    max_iter = int(max_iter)

    if (k <= 1 or max_iter <= 0):
        invalid_input()
    
    df1 = pd.read_csv(input_path1, header=None)
    df2 = pd.read_csv(input_path1, header=None)
    datapoints = pd.merge(df1, df2, on=0, how='inner')[1:].to_numpy()
    kmeans_pp(datapoints, k)



if __name__ == "__main__":
    main()