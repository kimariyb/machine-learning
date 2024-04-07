# 聚类

import matplotlib.pyplot as plt
import numpy as np
import mglearn as mg

from sklearn.datasets import make_blobs, fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans

def cluster_blobs():
    # 生成模拟的二维数据
    X, y = make_blobs(random_state=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 使用 2 个簇中心
    kmeans = KMeans(n_clusters=2, n_init='auto')
    kmeans.fit(X=X)
    assignments = kmeans.labels_

    mg.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

    # 使用 5 个簇中心
    kmeans = KMeans(n_clusters=5, n_init='auto')
    kmeans.fit(X=X)
    assignments = kmeans.labels_

    mg.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

    fig.savefig('day05_01.png', bbox_inches='tight', dpi=300)
    
    
def cluster():
    ...
     
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

mask = np.zeros(people.target.shape, dtype=np.bool_)

for target in np.unique(people.target):
    mask[
        np.where(people.target == target)[0][:50]
    ] = 1

X_people = (people.data[mask]) / 255.
y_people = people.target[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0
)

nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)

pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)

kmeans = KMeans(n_clusters=100, random_state=0, n_init='auto')
kmeans.fit(X_train)

X_reconstruct_nmf = np.dot(nmf.transform(X_test), nmf.components_)
X_reconstruct_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstruct_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]

image_shape = people.images[0].shape

fig, axes = plt.subplots(
    nrows=3,
    ncols=5,
    figsize=(8, 8),
    subplot_kw={
        'xticks': (),
        'yticks': (),
    }
)

fig.suptitle('Extracted Components')

for ax, com_kmeas, comp_pca, comp_nmf in zip(
    axes.T,
    kmeans.cluster_centers_,
    pca.components_,
    nmf.components_
):
    ax[0].imshow(com_kmeas.reshape(image_shape))
    ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
    ax[2].imshow(comp_nmf.reshape(image_shape))

axes[0, 0].set_ylabel('kmeans')
axes[1, 0].set_ylabel('pca')
axes[2, 0].set_ylabel('nmf')

fig.savefig('day05_02.png', dpi=300, bbox_inches='tight')