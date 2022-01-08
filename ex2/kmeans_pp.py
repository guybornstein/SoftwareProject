import pandas as pd
import numpy as np
import sys
import os
import mykmeanssp


DEFAULT_MAX_ITER = 300
DATAFILE = "rami_patsuts.txt"


def invalid_input():
    print("Invaid Input!")
    sys.exit(1)


def exception_handler():
    print("An Error Has Occurred")
    sys.exit(1)


def kmeans_pp(datapoints, k):
    m, n = datapoints.shape
    if k>=n:
        invalid_input()
    np.random.seed(0)
    observations = [np.random.choice(m, 1)[0]]
    for _ in range(1, k):
        diffs = datapoints.reshape(m, 1, n) - datapoints[observations].reshape(1, -1, n)
        dists = np.min(np.sum(diffs ** 2, axis=2), axis=1)
        probs = dists / np.sum(dists)
        observations.append(np.random.choice(m, 1, p=probs)[0])
    return observations


def write_data_to_file(datapoints, starting_centriods, k, max_iter, epsilon):
    m, n = datapoints.shape
    with open(DATAFILE, "w") as f:
        f.write(f"{m},{n},{k},{max_iter},{epsilon}\n")
        f.write(','.join([str(num) for num in starting_centriods]) + '\n')
        for i in range(len(datapoints)):
            f.write(','.join([str(arr) for arr in datapoints[i]]) + '\n')


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
    except Exception:
        invalid_input()

    
    k = int(k)
    max_iter = int(max_iter)

    if (k <= 1 or max_iter <= 0 or epsilon < 0):
        invalid_input()
  
    df1 = pd.read_csv(input_path1, header=None)
    df2 = pd.read_csv(input_path2, header=None)
    merged = pd.merge(df1, df2, on=0, how='inner')
    merged.sort_values(0, inplace=True)
    
    datapoints = merged.iloc[:, 1:].to_numpy()
    observations = kmeans_pp(datapoints, k)

    write_data_to_file(datapoints, observations, k, max_iter, epsilon)
    mykmeanssp.fit(DATAFILE)
    centroids = pd.read_csv(DATAFILE, header = None).to_numpy()
    print(','.join([str(i) for i in observations]))
    for centroid in centroids:
        print(','.join([f'{val:.4f}' for val in centroid]))
    os.remove(DATAFILE)


if __name__ == "__main__":
    main()