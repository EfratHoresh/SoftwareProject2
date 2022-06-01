import matplotlib.pyplot as plt
import matplotlib.patches as pat
import sklearn.cluster
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd


data = load_iris()
kmeans_results = []
df = pd.DataFrame(data.data, columns=data.feature_names)
for i in range(1, 11):
    centroids = sklearn.cluster.KMeans(i, random_state=0).fit(df)
    kmeans_results.append(centroids.inertia_)
plt.plot([i for i in range(1, 11)], kmeans_results, ms=15*2, mec="b", mfc="none", mew=2)
plt.annotate("Elbow Point", xy=(2.5, 130), xytext=(5, 300), arrowprops={"arrowstyle":"->","color":"black", "shrinkB":15*1.2})
plt.show()
