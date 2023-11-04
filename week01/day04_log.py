# 逻辑回归和线性 SVM

import mglearn as mg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def log_svm_forge():
    # 获取 forge 数据集
    X, y = mg.datasets.make_forge()

    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
        clf = model.fit(X, y)
        mg.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, 
                                ax=ax, alpha=.7)
        mg.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title('{}'.format(clf.__class__.__name__))
        ax.set_xlabel('Feature 0')
        ax.set_ylabel('Feature 1')
    axes[0].legend()
    fig.savefig('log.png', bbox_inches='tight', dpi=300) 
    
def log_cancer_l2():
    # 获取 cancer 数据集
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42
    )
    # 训练模型
    logreg1 = LogisticRegression().fit(X_train, y_train)
    print(f'Training set score: {logreg1.score(X_train, y_train):.3f}')
    print(f'Test set score: {logreg1.score(X_test, y_test):.3f}\n')
    # 训练模型
    logreg2 = LogisticRegression(C=100).fit(X_train, y_train)
    print(f'Training set score: {logreg2.score(X_train, y_train):.3f}')
    print(f'Test set score: {logreg2.score(X_test, y_test):.3f}\n')
    # 训练模型
    logreg3 = LogisticRegression(C=0.01).fit(X_train, y_train)
    print(f'Training set score: {logreg3.score(X_train, y_train):.3f}')
    print(f'Test set score: {logreg3.score(X_test, y_test):.3f}\n')
    # 创建画布
    fig, ax = plt.subplots()
    ax.plot(logreg1.coef_.T, 'o', label='C=1')
    ax.plot(logreg2.coef_.T, '^', label='C=100')
    ax.plot(logreg3.coef_.T, 'v', label='C=0.1')
    ax.set_xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    ax.hlines(0, 0, cancer.data.shape[1])
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('Coefficient magnitude')
    ax.legend()
    fig.savefig('log.png', bbox_inches='tight', dpi=300) 

def log_cancer_l1():    
    # 获取 cancer 数据集
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42
    )
    # 创建画布
    fig, ax = plt.subplots()

    for C, maker in zip([0.001, 1, 100], ['o', '^', 'v']):
        lr_l1 = LogisticRegression(C=C, penalty='l1', solver='liblinear').fit(X_train, y_train)
        print(f'Training accuracy of l1 logreg with C={C:.3f}: {lr_l1.score(X_train, y_train):.2f}')
        print(f'Test accuracy of l1 logreg with C={C:.3f}: {lr_l1.score(X_test, y_test):.2f}')
        ax.plot(lr_l1.coef_.T, maker, label=f'C={C:.3f}')
        
    ax.set_xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    ax.hlines(0, 0, cancer.data.shape[1])
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('Coefficient magnitude')
    ax.set_ylim(-5, 5)
    ax.legend()
    fig.savefig('log.png', bbox_inches='tight', dpi=300)
    
# 生成数据集
X, y = make_blobs()
# 训练线性 SVM 模型
svm = LinearSVC().fit(X, y)
print('Coefficent shape: ', svm.coef_.shape)
print('Intercept shape: ', svm.intercept_.shape)
