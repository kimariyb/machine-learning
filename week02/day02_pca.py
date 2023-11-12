# 主成分分析 PCA

import matplotlib.pyplot as plt
import mglearn as mg
import numpy as np

from sklearn.datasets import load_breast_cancer, fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_cancer_hist():
    cancer = load_breast_cancer()
    fig, axes = plt.subplots(15, 2, figsize=(10, 20))
    maligant = cancer.data[cancer.target == 0]
    benign = cancer.data[cancer.target == 1]

    ax = axes.ravel()

    for i in range(30):
        _, bins = np.histogram(cancer.data[:, i], bins=50)
        ax[i].hist(maligant[:, i], bins=bins, color=mg.cm3(0), alpha=.5)
        ax[i].hist(benign[:, i], bins=bins, color=mg.cm3(2), alpha=.5)
        ax[i].set_title(cancer.feature_names[i])
        ax[i].set_yticks(())
    ax[0].set_xlabel('Feature magnitude')
    ax[0].set_ylabel('Frequency')
    ax[0].legend(['maligant', 'benign'], loc='best')
    fig.tight_layout()
    fig.savefig('day02.png', dpi=300)


def pca_cancer():
    cancer = load_breast_cancer()

    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cancer.data)

    # 保留数据的前两个主成分
    pca = PCA(n_components=2)
    # 对乳腺癌数据拟合 PCA 模型
    pca.fit(X_scaled)

    # 将数据变换到前两个主成分的方向上
    X_pca = pca.transform(X_scaled)
    print(f'Original shape: {X_scaled.shape}')
    print(f'Reduced shape: {X_pca.shape}')

    # 对第一个和第二个主成分作图
    plt.figure(figsize=(8, 8))
    mg.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
    plt.legend(cancer.target_names, loc='best')
    plt.gca().set_aspect('equal')
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    # plt.show()

    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ['First component', 'Second component'])
    plt.colorbar()
    plt.xticks(
        range(len(cancer.feature_names)),
        cancer.feature_names, 
        rotation=60, 
        ha='left'
    )

    plt.xlabel('Feature')
    plt.ylabel('Principal component')
    plt.show()
    
def draw_people():
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(15, 8),
        subplot_kw={
            'xticks': (),
            'yticks': (),
        }
    )

    for target, image, ax in zip(people.target, people.images, axes.ravel()):
        ax.imshow(image)
        ax.set_title(people.target_names[target])
        
    fig.savefig('day_02.png', bbox_inches='tight', dpi=300)

    print(f'people.images.shape: {people.images.shape}') 
    print(f'Number of classes: {len(people.target_names)}')

def pca_people():
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool_)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[mask]
    y_people = people.target[mask]

    # 将灰度值缩放到 0-1 之间，而不是 0-255 之间
    # 以得到更好的数据稳定性
    X_people = X_people / 255.

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_people,
        y_people,
        stratify=y_people,
        random_state=0
    )
    # 使用一个邻居构建 KNeighborsClassifer
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print(f'Test set score of 1-nn: {knn.score(X_test, y_test):.2f}')

    pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f'X_train_pca.shape: {X_train_pca.shape}')

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_pca, y_train)
    print(f'Test set accuracy: {knn.score(X_test_pca, y_test):.2f}')

    print(f'pca.components_.shape: {pca.components_.shape}')

    fig, axes = plt.subplots(
        nrows=3,
        ncols=5,
        figsize=(15, 12),
        subplot_kw={
            'xticks': (),
            'yticks': (),
        }
    )

    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title('{}. component'.format((i + 1)))

    fig.savefig('day02.png', bbox_inches='tight', dpi=300)