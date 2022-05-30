import pandas as pd
import numpy as np
import sys

args = sys.argv[1:]
if len(args) == 5:  # max_iteration given
    if not args[0].isnumeric() or not args[1].isnumeric() or not args[2].isnumeric():  # k, max_iteration ot eps not int
        print("Invalid Input!")
        quit()
    find_centrids()
    kmeans(int(args[0]), args[2], int(args[1]))
elif len(args) == 4:  # max_iteration not given
    if not args[0].isnumeric() or not args[1].isnumeric():  # k not int
        print("Invalid Input!")
        quit()
    kmeans(int(args[0]), args[1])
else:  # wrong number of args
    print("Invalid Input!")
    quit()


def find_centroids(k, file_name_1, file_name_2):
    # prepare DataFrame
    file_1 = pd.read_csv(file_name_1, header=None)
    file_2 = pd.read_csv(file_name_2, header=None)
    vectors = pd.merge(file_1, file_2, on=0)
    index_column = vectors.columns[0]
    vectors.sort_values(by=[index_column], inplace=True)
    vectors[index_column] = pd.to_numeric(vectors[index_column], downcast='signed')
    vectors.set_index(index_column, inplace=True)
    index_lst = vectors.index.values.tolist()
    vectors = vectors.to_numpy()
    # choose random centroid
    np.random.seed(0)
    n = len(vectors)
    random_index = np.random.choice(n)
    i = 1
    centroids = [vectors[random_index]]
    centroids_index = [index_lst[random_index]]
    while i != k:
        d = np.array([0.0 for j in range(n)])
        for l in range(n):
            min_distance = np.linalg.norm(vectors[l]-centroids[0])
            for centroid in centroids:
                distance = np.linalg.norm(vectors[l]-centroid)
                min_distance = min(distance, min_distance)
            d[l] = min_distance**2
        d_sum = d.sum()
        p = np.array([d[j]/d_sum for j in range(n)])
        i += 1
        index = np.random.choice(n, p=p)
        centroids_index.append(index_lst[index])
        centroids.append(vectors[index])
    return centroids, centroids_index


