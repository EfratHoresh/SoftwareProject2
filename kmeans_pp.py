import pandas as pd
import numpy as np
import sys
import mykmeanssp
import os


def prepare_data(k, file_name_1, file_name_2):
    # prepare DataFrame
    try:  #  read from files
        file_1 = pd.read_csv(file_name_1, header=None)
        file_2 = pd.read_csv(file_name_2, header=None)
    except Exception:
        print("An Error Has Occurred")
        quit()
    vectors = pd.merge(file_1, file_2, on=0)  # inner join 2 data frames
    index_column = vectors.columns[0]
    vectors.sort_values(by=[index_column], inplace=True)
    vectors[index_column] = pd.to_numeric(vectors[index_column], downcast='signed')
    vectors.set_index(index_column, inplace=True)
    vectors.to_csv('vectors_tmp_file.txt', header=False, index=False)  # create a new file of all vectors
    index_lst = vectors.index.values.tolist()
    vectors = vectors.to_numpy()
    # choose random centroid
    np.random.seed(0)
    n = len(vectors)
    if k >= n or k < 2:
        print("Invalid Input!")
        quit()
    random_index = np.random.choice(n)
    i = 1
    centroids = [vectors[random_index]]
    centroids_index = [index_lst[random_index]]
    while i < k:
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
        np.savetxt('centroids_tmp_file.txt', centroids, fmt='%.4f', delimiter=',')  # create new file for centroids
    return centroids_index


def is_e_legal(e):
    try:
        e = float(e)
        if e >= 0:
            return True
        else:
            return False
    except ValueError:
        return False


def is_max_iter_legal(max_iter):
    return max_iter.isnumeric() and int(max_iter)>0


args = sys.argv[1:]
if len(args) == 5:  # max_iteration given
    if not args[0].isnumeric() or not is_max_iter_legal(args[1]) or not is_e_legal(args[2]):  # k, max_iteration or eps not int
        print("Invalid Input!")
        quit()
    index_lst = prepare_data(int(args[0]), args[3], args[4])
    mykmeanssp.fit(int(args[0]), int(args[1]), float(args[2]))  # run c module
elif len(args) == 4:  # max_iteration not given
    if not args[0].isnumeric() or not is_e_legal(args[1]):  # k not int
        print("Invalid Input!")
        quit()
    index_lst = prepare_data(int(args[0]), args[2], args[3])
    mykmeanssp.fit(int(args[0]), 300, float(args[1]))  # run c module
else:  # wrong number of args
    print("Invalid Input!")
    quit()
index_str = ','.join([str(num) for num in index_lst])  # add indexes to file
try:
    f = open('results_tmp_file.txt', 'r+')
    content = f.read()
    f.close()
except Exception:
    print("An Error Has Occurred")
    quit()
# f.seek(0, 0)
# f.write(index_str+'\n'+content)
# f.seek(0, 0)  # print file
# content = f.read()
os.remove('vectors_tmp_file.txt')
os.remove('centroids_tmp_file.txt')
os.remove('results_tmp_file.txt')
print(index_str+'\n'+content)
