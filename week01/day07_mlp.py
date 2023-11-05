# 神经网络，前馈神经网络 MLP

import mglearn as mg
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split


def network_moons():
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )

    # mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
    # mg.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    # mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 1')
    # plt.show()

    # mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
    # mlp.fit(X_train, y_train)
    # mg.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    # mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 1')
    # plt.show()

    # 使用 2 个隐层，每个层包含 10 个单元
    # mlp = MLPClassifier(solver='lbfgs', random_state=0, activation='relu', hidden_layer_sizes=[10, 10])
    # mlp.fit(X_train, y_train)
    # mg.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    # mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 1')
    # plt.show()

    # 使用 2 个隐层，每个层包含 10 个单元，但是使用 tanh 非线性激活函数
    # mlp = MLPClassifier(solver='lbfgs', random_state=0, activation='tanh', hidden_layer_sizes=[10, 10])
    # mlp.fit(X_train, y_train)
    # mg.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    # mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 1')
    # plt.show()

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for axx, n_hidden_nodes in zip(axes, [10, 100]):
        for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
            mlp = MLPClassifier(
                solver='lbfgs', 
                random_state=0,
                hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                alpha=alpha
            )
            mlp.fit(X_train, y_train)
            mg.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
            mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
            ax.set_title(f'n_hidden=[{n_hidden_nodes}, {n_hidden_nodes}]\nalpha={alpha:.4f}')

    fig.savefig('day07.png', bbox_inches='tight', dpi=300)
    
    
def network_cancer():
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data,
        cancer.target,
        random_state=0
    )

    # mlp = MLPClassifier(random_state=42)
    # mlp.fit(X_train, y_train)

    # print(f'Accuracy on training set: {mlp.score(X_train, y_train):.2f}')
    # print(f'Accuracy on test set: {mlp.score(X_test, y_test):.2f}')

    # 计算训练集中每个特征的平均值
    mean_on_train = X_train.mean(axis=0)
    # 计算训练集中每个特征的标准差
    std_on_train = X_train.std(axis=0)

    # 减去平均值，然后乘上标准差的倒数
    # 如此计算后 mean=0，std=1
    X_train_scaled = (X_train - mean_on_train) / std_on_train
    # 对测试集做相同的变换
    X_test_scaled = (X_test - mean_on_train) / std_on_train

    mlp = MLPClassifier(max_iter=1000, random_state=0)
    mlp.fit(X_train_scaled, y_train)

    print(f'Accuracy on training set: {mlp.score(X_train_scaled, y_train):.3f}')
    print(f'Accuracy on test set: {mlp.score(X_test_scaled, y_test):.3f}')


    plt.figure(figsize=(20, 5))
    plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
    plt.yticks(range(30), cancer.feature_names)
    plt.xlabel('Colums in weight matrix')
    plt.ylabel('Input feature')
    plt.colorbar()
    plt.show()

