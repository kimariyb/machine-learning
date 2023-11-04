# 朴素贝叶斯分类器

import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.linear_model import LinearRegression

def bayes():
    """朴素贝叶斯分类器"""
    X = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 1, 0]
    ])

    y = np.array([0, 1, 0, 1])
    counts = {}
    for label in np.unique(y):
        # 对每个类别进行遍历
        # 计算（求和）中每个特征 1 的个数
        counts[label] = X[y == label].sum(axis=0)
    print(f'Feature counts: \n{counts}')
    
def tree_cancer():
    # 决策树
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42
    )
    tree1 = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    print(f'Accuracy on training set: {tree1.score(X_train, y_train):.3f}')
    print(f'Accuracy on test set: {tree1.score(X_test, y_test):.3f}')
    tree2 = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
    print(f'Accuracy on training set: {tree2.score(X_train, y_train):.3f}')
    print(f'Accuracy on test set: {tree2.score(X_test, y_test):.3f}')

    # export_graphviz(
    #     tree2, 
    #     out_file='tree.dot', 
    #     class_names=['malignat', 'benign'],
    #     feature_names=cancer['feature_names'],
    #     impurity=False,
    #     filled=True
    # )

    # graph = graphviz.Source.from_file('tree.dot')
    # graph.render(filename='tree', format='png')

    print(f'Features importance: \n{tree2.feature_importances_}')
    
def tree_ram_price():
    """决策树 ram 价格"""
    ram_prices = pd.read_csv('data/ram_price.csv')
    # 利用历史数据预测 2000 年之后的价格
    data_train = ram_prices[ram_prices.date < 2000]
    data_test = ram_prices[ram_prices.date >= 2000]

    # 基于日期来预测价格
    X_train = np.expand_dims(data_train.date, axis=1)
    # 我们利用对数变换得到数据和目标之间更简单的关系
    y_train = np.log(data_train.price)

    tree = DecisionTreeRegressor().fit(X_train, y_train)
    linear_reg = LinearRegression().fit(X_train, y_train)

    # 对所有数据进行预测
    X_all = np.expand_dims(ram_prices.date, axis=1)

    pred_tree = tree.predict(X_all)
    pred_lr = linear_reg.predict(X_all)

    # 对数变换逆运算
    pred_tree = np.exp(pred_tree)
    pred_lr = np.exp(pred_lr)

    fig, ax = plt.subplots()

    ax.semilogy(data_train.date, data_train.price, label='Training data')
    ax.semilogy(data_test.date, data_test.price, label='Test data')
    ax.semilogy(ram_prices.date, pred_tree, label='Tree prediction')
    ax.semilogy(ram_prices.date, pred_lr, label='Linear prediction')
    ax.legend()

    fig.savefig('day05_02.png', dpi=300, bbox_inches='tight')