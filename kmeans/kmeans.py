import sys


DEFAULT_MAX_ITER = 200


def invalid_input():
    print("Invaid Input!")
    sys.exit(1)


def exception_handler():
    print("An Error Has Occurred")
    sys.exit(1)


def load_csv(input_path: str) -> list[tuple[float]]:
    datapoints = []
    with open(input_path) as file:
        for line in file:
            points = line[:-1].split(',')
            datapoints.append(tuple([float(p) for p in points]))
    return datapoints


def save_csv(centroids: list[tuple[float]], output_path: str):
    with open(output_path, "w") as file:
        for tup in centroids:
            file.write(','.join([f'{x:.4f}' for x in tup]) + '\n')


def distance(point1: tuple[float], point2: tuple[float]) -> float:
    s = 0
    for i in range(len(point1)):
        s += pow((point1[i]-point2[i]),2)

    return pow(s, 0.5)


def vector_sum(vector1: tuple[float], vector2: tuple[float]) -> tuple[float]:
    assert(len(vector1) == len(vector2))
    return tuple([vector1[i] + vector2[i] for i in range(len(vector1))])


def scalar_product(vector: tuple[float], a: float) -> tuple[float]:
    return tuple([vector[i] *a for i in range(len(vector))])


def kmeans(datapoints: list[tuple[float]], k: int, max_iter: int, epsilon: float = 0.001) -> list[tuple[float]]:
    centroids = datapoints[:k]
    dim = len(datapoints[0])
    iteration = 0
    
    while iteration <= max_iter:
        iteration += 1

        clusters_mapping = []
        prev_centroids = centroids[:]
        
        # find the closest cluster to each datapoint
        for point in datapoints:
            cluster = min(range(k), key=lambda cluster: distance(point, centroids[cluster]))
            clusters_mapping.append(cluster)

        # calculate the new centroids by averging the points in each cluster
        cluster_sum = [tuple([0] * dim) for i in range(k)]
        cluster_size = [0 for i in range(k)]

        for point_index, cluster in enumerate(clusters_mapping):
            cluster_sum[cluster] = vector_sum(cluster_sum[cluster], datapoints[point_index])
            cluster_size[cluster] += 1

        for cluster in range(k):
            centroids[cluster] = scalar_product(cluster_sum[cluster], 1 / cluster_size[cluster])

        # check for convergence
        for cluster in range(k):
            if distance(centroids[cluster], prev_centroids[cluster]) >= epsilon:
                break
        else:
            return centroids
    
    return centroids


def main():
    # parse commandline arguments
    if len(sys.argv) == 5:
        _, k, max_iter, input_path, output_path = sys.argv
    elif len(sys.argv) == 4:
        _, k, input_path, output_path = sys.argv
        max_iter = str(DEFAULT_MAX_ITER)
    else:
        invalid_input()
    
    if not k.isdigit() or not max_iter.isdigit():
        invalid_input()

    k = int(k)
    max_iter = int(max_iter)
    
    # load, calculate, and save kmeans
    datapoints = load_csv(input_path)
    centroids = kmeans(datapoints=datapoints, k=k, max_iter=max_iter)
    save_csv(centroids, output_path)


if __name__ == "__main__":
    main()
