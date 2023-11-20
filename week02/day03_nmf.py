# 非负矩阵分解 NMF 

import numpy as np
import mglearn as mg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, PCA
from sklearn.datasets import fetch_lfw_people

def people_nmf():
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

    nmf = NMF(n_components=15, random_state=0)
    X_train_nmf = nmf.fit_transform(X_train)
    X_test_nmf = nmf.fit_transform(X_test)

    # fig, axes = plt.subplots(
    #     nrows=3,
    #     ncols=5,
    #     figsize=(15, 12),
    #     subplot_kw={
    #         'xticks': (),
    #         'yticks': (),
    #     }
    # )

    # for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    #     ax.imshow(component.reshape(image_shape))
    #     ax.set_title('{}. component'.format(i))

    # fig.savefig('day03.png', dpi=300, bbox_inches='tight')

    compn = 3
    # 按第 3 个分量排序，绘制前 10 张图像
    inds = np.argsort(X_train_nmf[:, compn])[::-1]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(15, 8),
        subplot_kw={
            'xticks': (),
            'yticks': (),
        }
    )

    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(image_shape))
        
    fig.savefig('day03_02.png', dpi=300, bbox_inches='tight')

    compn = 7
    # 按第 7 个分量排序，绘制前 10 张图像
    inds = np.argsort(X_train_nmf[:, compn])[::-1]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(15, 8),
        subplot_kw={
            'xticks': (),
            'yticks': (),
        }
    )

    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(image_shape))
        
    fig.savefig('day03_03.png', dpi=300, bbox_inches='tight')

def signal():
    S = mg.datasets.make_signals()
    # 将数据混合成 100 维的状态
    A = np.random.RandomState(0).uniform(size=(100, 3))
    X = np.dot(S, A.T)
    print('Shape of measurements: {}'.format(X.shape))

    nmf = NMF(n_components=3, random_state=42)
    S_ = nmf.fit_transform(X)
    print('Shape of measurements: {}'.format(S_.shape))

    pca = PCA(n_components=3)
    H = pca.fit_transform(X)

    models = [X, S, S_, H]
    names = [
        'Observations (First three measurements)',
        'True sources',
        'NMF recovered signals',
        'PCA recovered signals'
    ]

    fig, axes = plt.subplots(
        nrows=4,
        figsize=(8, 4),
        gridspec_kw={
            'hspace': .5
        },
        subplot_kw={
            'xticks': (),
            'yticks': (),
        }
    )


    for model, name, ax in zip(models, names, axes):
        ax.set_title(name)
        ax.plot(model[:, :3], '-')

    fig.savefig('day03_04.png', dpi=300, bbox_inches='tight')
