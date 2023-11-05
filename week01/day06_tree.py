# 随机森林

import matplotlib.pyplot as plt
import mglearn as mg
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_moons, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

def moons_forest():
    """随机森林"""
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    forest = RandomForestClassifier(n_estimators=5, random_state=2)
    forest.fit(X_train, y_train)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title(f'Tree {i}')
        mg.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

    mg.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
    axes[-1, -1].set_title('Random Forest')
    mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.show()

def cancer_forest(): 
    """cancer 随机森林"""
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0
    )
    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(X_train, y_train)
    print(f'Accuracy on training set: {forest.score(X_train, y_train):.3f}')
    print(f'Accuracy on test set: {forest.score(X_test, y_test):.3f}')

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0
    )
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train)
    print(f'Accuracy on training set: {gbrt.score(X_train, y_train):.3f}')
    print(f'Accuracy on test set: {gbrt.score(X_test, y_test):.3f}')

def draw_blobs():
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    # 添加第二个特征的平方
    X_new = np.hstack([X, X[:, 1:] ** 2])
    fig = plt.figure(figsize=(10, 9))
    # 3D 可视化
    ax = fig.add_subplot(projection='3d')
    # 首先画出所有 y == 0 的点，然后画出所有 y == 1 的点
    mask = y == 0
    # Set the elevation and azimuth angles
    ax.view_init(elev=-152, azim=-26)

    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
                s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r',
                marker='^', s=60)
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_zlabel('feature 1 ** 2')

    fig.savefig('day06.png', dpi=300)

def line_svm_blobs():
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    # 添加第二个特征的平方，作为第三个特征
    X_new = np.hstack([X, X[:, 1:] ** 2])
    linear_svm_3d = LinearSVC().fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    # 显示线性决策边界
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(projection='3d')
    # Set the elevation and azimuth angles
    ax.view_init(elev=-152, azim=-26)
    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
    # 首先画出所有 y == 0 的点，然后画出所有 y == 1 的点
    mask = y == 0
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
                s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r',
                marker='^', s=60)
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set_zlabel('feature 1 ** 2')

    fig.savefig('day06.png', dpi=300)
    
def svm():
    X, y = mg.tools.make_handcrafted_dataset()
    svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
    # mg.plots.plot_2d_separator(svm, X, eps=0.5)
    # mg.discrete_scatter(X[:, 0], X[:, 1], y)
    # # 画出支持向量
    # sv = svm.support_vectors_
    # # 支持向量的类别标签由 dual_coef_ 的正负号给出
    # sv_labels = svm.dual_coef_.ravel() > 0
    # mg.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 1')

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    for ax, C in zip(axes, [-1, 0, 3]):
        for a, gamma in zip(ax, range(-1, 2)):   
            mg.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

    axes[0, 0].legend(
        ['class 0', 'class 1', 'sv class 0', 'sv class 1'], 
        ncol=4,
        loc=(.9, 1.2)
    )
    fig.savefig('day06.png', bbox_inches='tight', dpi=300)