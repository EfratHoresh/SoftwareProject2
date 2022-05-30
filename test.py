import pandas as pd
import numpy as np
import sys


file_1 = pd.read_csv("input_1_db_1.txt", header=None)
file_2 = pd.read_csv("input_1_db_2.txt", header=None)
vectors = pd.merge(file_1, file_2, on=0)
print(vectors.iloc[[5]])


def oclidDistance(x1, x2):
    distance = 0
    for i in range(len(x1.columns)):
        x1_val = x1.iloc[0, i]
        x2_val = x2.iloc[0, i]
        distance += (x1_val-x2_val)**2
    return distance


def find_centroids(k, file_name_1, file_name_2):
    # prepare DataFrame
    file_1 = pd.read_csv(file_name_1, header=None)
    file_2 = pd.read_csv(file_name_2, header=None)
    vectors = pd.merge(file_1, file_2, on=0)
    vectors.drop([0], axis=1, inplace=True)
    # choose random centroid
    np.random.seed(0)
    random_index = np.random.choice(vectors.index)
    n = len(vectors)
    i = 1
    centroids = [vectors.iloc[[random_index]]]
    index_lst = [random_index]
    while i != k:
        d = np.array([0 for j in range(n)])
        for l in range(n):
            min_distance = oclidDistance(vectors.iloc[[l]], centroids[0])
            for centroid in centroids:
                distance = oclidDistance(vectors.iloc[[l]], centroid)
                min_distance = min(distance, min_distance)
            d[l] = min_distance
        d_sum = d.sum()
        p = np.array([d[j]/d_sum for j in range(n)])
        i += 1
        index = np.random.choice(vectors.index, p=p)
        index_lst.append(index)
        centroids.append(vectors.iloc[[index]])
    return centroids, index_lst


centroids, index_lst = find_centroids(3, "input_1_db_1.txt", "input_1_db_2.txt")

print(centroids)
print(index_lst)