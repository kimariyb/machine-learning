# 聚类

import matplotlib.pyplot as plt
import mglearn as mg

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 生成模拟的二维数据
X, y = make_blobs(random_state=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 使用 2 个簇中心
kmeans = KMeans(n_clusters=2)
kmeans.fit(X=X)
assignments = kmeans.labels_

mg.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

# 使用 5 个簇中心
kmeans = KMeans(n_clusters=5)
kmeans.fit(X=X)
assignments = kmeans.labels_

mg.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
