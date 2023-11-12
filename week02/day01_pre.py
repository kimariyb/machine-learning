# 数据预处理

import matplotlib.pyplot as plt
import mglearn as mg

from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

def scaled_cancer():
    """缩放 cancer 数据"""
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=1
    )

    print(X_train.shape)
    print(X_test.shape)

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # 变化数据
    X_train_scaled = scaler.transform(X_train)
    # 在缩放之前和之后分别打印数据集属性
    print(f'transformed shape: {X_train_scaled.shape}')
    print(f'per-feature minimum before scaling: \n{X_train.min(axis=0)}')
    print(f'per-feature maximum before scaling: \n{X_train.max(axis=0)}')
    print(f'per-feature minimum after scaling: \n{X_train_scaled.min(axis=0)}')
    print(f'per-feature maximum after scaling: \n{X_train_scaled.max(axis=0)}')

    # 对测试集进行变换
    X_test_scaled = scaler.transform(X_test)
    # 在缩放之后打印测试数据的属性
    print(f'per-feature minimum after scaling: \n{X_test_scaled.min(axis=0)}')
    print(f'per-feature maximum after scaling: \n{X_test_scaled.max(axis=0)}')


def scaled_blob():
    """数据缩放 blob 数据集"""
    # 构造数据
    X, _ = make_blobs(
        n_samples=50,
        centers=5,
        random_state=4,
        cluster_std=2
    )

    # 将其分为测试集和训练集
    X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

    # 绘制训练集和测试集
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=mg.cm2(0),
        label='Training set',
        s=60
    )
    axes[0].scatter(
        X_test[:, 0],
        X_test[:, 1],
        marker='^',
        c=mg.cm2(1),
        label='Test set',
        s=60
    )
    axes[0].legend(loc='upper left')
    axes[0].set_title('Original Data')

    # 利用 MinMaxScaler 缩放数据
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将正确缩放的数据可视化
    axes[1].scatter(
        X_train_scaled[:, 0],
        X_train_scaled[:, 1],
        c=mg.cm2(0),
        label='Training set',
        s=60
    )
    axes[1].scatter(
        X_test_scaled[:, 0],
        X_test_scaled[:, 1],
        marker='^',
        c=mg.cm2(1),
        label='Test set',
        s=60
    )
    axes[1].set_title('Scaled Data')

    # 单独对测试集进行缩放
    # 使得测试集的最小值为 0，最大值为 1
    # 千万不要这么做！
    test_scaler = MinMaxScaler()
    test_scaler.fit(X_test)
    X_test_scaled_badly = test_scaler.transform(X_test)

    # 将错误缩放的数据可视化
    axes[2].scatter(
        X_train_scaled[:, 0],
        X_train_scaled[:, 1],
        c=mg.cm2(0),
        label='Training set',
        s=60
    )
    axes[2].scatter(
        X_test_scaled_badly[:, 0],
        X_test_scaled_badly[:, 1],
        marker='^',
        c=mg.cm2(1),
        label='Test set',
        s=60
    )
    axes[2].set_title('Improperly Scaled Data')

    for ax in axes:
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')

    fig.savefig('day01.png', bbox_inches='tight', dpi=300)


def pre_cancer():
    """预处理 cancer"""

    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0
    )

    svm = SVC(C=100)
    svm.fit(X_train, y_train)
    print(f'Test set accuracy: {svm.score(X_test, y_test):.2f}')

    # 使用 0-1 缩放处理
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 在缩放后的训练数据上学习 SVM
    svm.fit(X_train_scaled, y_train)

    # 在缩放后的测试集上计算分数
    print(f'Scaled test set accuracy: {svm.score(X_test_scaled, y_test)}')

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 在缩放后的训练数据上学习 SVM
    svm.fit(X_train_scaled, y_train)

    # 在缩放后的测试集上计算分数
    print(f'Scaled test set accuracy: {svm.score(X_test_scaled, y_test)}')
