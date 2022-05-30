import pandas as pd
import numpy as np
import sys

#
file_1 = pd.read_csv("input_2_db_1.txt", header=None)
file_2 = pd.read_csv("input_2_db_2.txt", header=None)
# vectors = pd.merge(file_1, file_2, on=0)
# print(vectors.dtypes)
# index_column = vectors.columns[0]
# vectors.sort_values(by=[index_column], inplace=True)
# vectors[index_column] = pd.to_numeric(vectors[index_column], downcast='signed')
# vectors.set_index(index_column, inplace=True)
# print(vectors.index.values.tolist())
# vectors = vectors.to_numpy()
# print(vectors)
# vector1 = vectors[0]
# vector2 = vectors[1]
#
#

#
# def oclidDistance(x1, x2):
#     distance = 0
#     for i in range(len(x1.columns)):
#         x1_val = x1.iloc[0, i]
#         x2_val = x2.iloc[0, i]
#         distance += (x1_val-x2_val)**2
#     return distance

def oclidDistance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i]-x2[i])**2
    return distance


# print(oclidDistance(vector1, vector2))
# print(np.linalg.norm(vector1-vector2)**2)

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


centroids, centroids_index = find_centroids(15, "input_3_db_1.txt", "input_3_db_2.txt")

print(centroids)
print(centroids_index)