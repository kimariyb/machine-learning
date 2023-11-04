# 线性回归问题

import mglearn as mg
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

def regression_wave():
    """wave 数据集的线性回归模型"""
    # 生成数据集
    X, y = mg.datasets.make_wave(n_samples=60)
    # 生成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 训练线性回归模型
    lr = LinearRegression().fit(X_train, y_train)
    # 斜率参数为 coef_，截距参数为 inercept_
    print(f'lr.coef_: {lr.coef_}')
    print(f'lr.intercept_: {lr.intercept_}')
    # 训练集和测试集的性能
    print(f'Training set score: {lr.score(X_train, y_train):.2f}')
    print(f'Test set score: {lr.score(X_test, y_test):.2f}')

def regression_boston():
    # 构建波士顿数据集
    X, y = mg.datasets.load_extended_boston()
    # 生成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 构建线性回归模型，并训练模型
    lr = LinearRegression().fit(X_train, y_train)
    # 训练集和测试集的性能
    print(f'Training set score: {lr.score(X_train, y_train):.2f}')
    print(f'Test set score: {lr.score(X_test, y_test):.2f}')
    # 构建岭回归模型，并训练模型
    ridge = Ridge().fit(X_train, y_train)
    # 训练集和测试集的性能
    print(f'Training set score: {ridge.score(X_train, y_train):.2f}')
    print(f'Test set score: {ridge.score(X_test, y_test):.2f}')
    # 使用 alpha 参数处理模型，默认为 1
    # 增加 alpha 参数值会使得训练集性能降低
    ridge_10 = Ridge(alpha=10).fit(X_train, y_train)
    # 训练集和测试集的性能
    print(f'Training set score: {ridge_10.score(X_train, y_train):.2f}')
    print(f'Test set score: {ridge_10.score(X_test, y_test):.2f}')

def ridge_boston():
    # 构建波士顿数据集
    X, y = mg.datasets.load_extended_boston()
    # 生成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 分别训练线性回归、岭回归 alpha=0.1、1、10 四个模型
    lr = LinearRegression().fit(X_train, y_train)
    rid1 = Ridge(alpha=0.1).fit(X_train, y_train)
    rid2 = Ridge(alpha=1).fit(X_train, y_train)
    rid3 = Ridge(alpha=10).fit(X_train, y_train)
    # 生成画布
    fig, ax = plt.subplots()
    ax.plot(rid1.coef_, 's', label='Ridge alpha=0.1')
    ax.plot(rid2.coef_, '^', label='Ridge alpha=1')
    ax.plot(rid3.coef_, 'v', label='Ridge alpha=10')
    ax.plot(lr.coef_, 'o', label='LinearRegression')
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('Coefficient magnitude')
    ax.set_ylim(-25, 25)
    ax.hlines(0, 0, len(lr.coef_))
    ax.legend()
    fig.savefig('reg.png', bbox_inches='tight', dpi=300)

def lasso_boston():
    # 构建波士顿数据集
    X, y = mg.datasets.load_extended_boston()
    # 生成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 训练 Lasso 模型
    lasso1 = Lasso().fit(X_train, y_train)
    print(f'Training set score: {lasso1.score(X_train, y_train):.2f}')
    print(f'Test set score: {lasso1.score(X_test, y_test):.2f}')
    print(f'Number of features used: {np.sum(lasso1.coef_ != 0)}\n')
    # 如果减小 alpha 和增大 max_iter
    lasso2 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    print(f'Training set score: {lasso2.score(X_train, y_train):.2f}')
    print(f'Test set score: {lasso2.score(X_test, y_test):.2f}')
    print(f'Number of features used: {np.sum(lasso2.coef_ != 0)}\n')
    # 如果把 alpha 设置得太小，就会出现过拟合的情况
    lasso3 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
    print(f'Training set score: {lasso3.score(X_train, y_train):.2f}')
    print(f'Test set score: {lasso3.score(X_test, y_test):.2f}')
    print(f'Number of features used: {np.sum(lasso3.coef_ != 0)}\n')
    # 生成画布
    fig, ax = plt.subplots()
    ax.plot(lasso1.coef_, 's', label='Lasso alpha=1')
    ax.plot(lasso2.coef_, '^', label='Lasso alpha=0.1')
    ax.plot(lasso3.coef_, 'v', label='Lasso alpha=0.0001')
    rid1 = Ridge(alpha=0.1).fit(X_train, y_train)
    ax.plot(rid1.coef_, 'o', label='Ridge alpha=0.1')
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('Coefficient magnitude')
    ax.set_ylim(-25, 25)
    ax.legend(ncol=2, loc=(0, 1.05))
    fig.savefig('reg.png', bbox_inches='tight', dpi=300)    
    
