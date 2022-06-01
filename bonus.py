# import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
import numpy as np


data = load_iris()
kmeans_results = []
for i in range(1, 11):
    centroids = sklearn.cluster.KMeans(i, random_state=0)
    kmeans_results.append(centroids)
