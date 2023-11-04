# 监督学习，分为分类和回归两大问题

import numpy as np
import mglearn as mg
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def classification_forge():
    """使用简单的 forge 数据集"""
    
    # K 近邻分类算法
    # 生成 forge 数据集，同时生成测试集和训练集
    X, y = mg.datasets.make_forge()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 搭建 K 近邻模型，同时拟合数据
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    # 使用测试集评估模型
    print(f'Test set accuracy: {knn.score(X_test, y_test):.2f}')

    # 使用可视化的方法查看决策边界
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for n_neighbors, ax in zip([1, 3, 9], axes):
        # fit 方法返回对象本身，所以可以将实例化和拟合放在一行中
        knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mg.plots.plot_2d_separator(knn, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mg.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title(f'{n_neighbors} neighbor(s)')
        ax.set_xlabel('feature 0')
        ax.set_ylabel('feature 1')
    axes[0].legend(loc=3)
    fig.savefig('neighbors.png', dpi=300)
    
    
def classification_cancer():   
    """使用复杂的 breast_cancer 数据集""" 
    
    # 读取乳腺癌数据集，并打印些基本信息，benign 为良性；malignant 为恶性
    cancer = load_breast_cancer()
    print(f'Shape of cancer data: {cancer.data.shape}')
    print('Sample count per class: ')
    for class_name, count in zip(cancer.target_names, 
                                    np.bincount(cancer.target)):
        print(f'{class_name}, {count}')
        
    # 得到每个特征的语义说明
    print(f'Feature names: \n{cancer.feature_names}')

    # 获取测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66
    )

    training_accuracy = []
    test_accuracy = []

    # n_neighbors 取值从 1 到 10
    neighbors_settings = range(1, 11)

    for n_neighbors in neighbors_settings:
        # 构建模型
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        # 记录训练精度
        training_accuracy.append(knn.score(X_train, y_train))
        # 记录测试精度
        test_accuracy.append(knn.score(X_test, y_test))
        
    # 可视化，可以看见仅考虑单一近邻时，训练集上的预测结果十分完美，随着邻居个数的增多，精度下降
    # 单一近邻时，测试集的预测结果非常差，随着邻居个数的增多，预测性能先增大后减小    
    fig, ax= plt.subplots(figsize=(7, 5))
    ax.plot(neighbors_settings, training_accuracy, label='training accuracy')
    ax.plot(neighbors_settings, test_accuracy, label='test accuracy', linestyle='--')
    ax.set_xlabel('n_neighbors')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='upper right')
    fig.savefig('neighbors.png', bbox_inches='tight', dpi=300)


def regressor_wave():
    """K 近邻回归代码"""
    # K 近邻回归，使用 wave 数据集
    X, y = mg.datasets.make_wave(n_samples=40)
    # 将 Wave 数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # 模型实例化，并将邻居设置为 3
    reg = KNeighborsRegressor(n_neighbors=3)
    # 利用训练集训练模型
    reg.fit(X_train, y_train)

    # 对测试集进行预测
    print(f'Test set prediction: \n{reg.predict(X_test)}') 
    # 对模型进行评估，R^2 > 0.8 就说明拟合很好了
    print(f'Test set R^2: {reg.score(X_test, y_test):.2f}')

    # K 近邻回归的缺陷，首先创建 1000 个数据点，在 -3 到 3 之间均匀分布
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # 利用 1、3、9 个邻居进行预测
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mg.cm2(0), markersize=8)
        ax.plot(X_test, y_test, '^', c=mg.cm2(1), markersize=8)
        ax.set_title(
            f'{n_neighbors} neighbors\n train score: {reg.score(X_train, y_train):.2f} test score: {reg.score(X_test, y_test):.2f}'
        )
        ax.set_xlabel('Feature')
        ax.set_ylabel('Target')
    axes[0].legend(['Model predictions', 'Training data/target', 'Test data/target'], loc='best')
    fig.savefig('neighbors.png', bbox_inches='tight', dpi=300)
    
