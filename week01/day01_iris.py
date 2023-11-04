# 鸢尾花分类，三分类监督学习
# setosa、versicolor 和 virginica 三个鸢尾花品种

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载 iris 数据集，并打印数据集的 key
iris_dataset = load_iris()
print(f'- Key of iris_dataset: \n{iris_dataset.keys()}')

def intro_iris(): 
    """展示 iris_dataset 中的各种 key"""
    
    # DESCR 是数据集的简要说明
    # print(iris_dataset['DESCR'[:193]] + '\n...')

    # target_names 的 value 是预测的花的品种
    # feature_names 的 value 是每一个特征的名字
    # 数据集的键值对除了可以通过 dict 方式，还可以直接通过 dataset.key 的方式
    print(f'- Target names: \n{iris_dataset.target_names}')
    print(f'- Featrue names: \n{iris_dataset.feature_names}')

    # 数据包含在 target 和 data 字段中，data 字段是一个 numpy 数组
    print(f'Shape of data: {iris_dataset.data.shape}')

    # target 是一个一维 numpy 数组，每一朵花对应一个数据
    # 品种被转化为 0 到 2 的整数，0 代表 setosa、1 代表 versicolor、2 代表 virginica
    print(f'Shape of target: {iris_dataset.target.shape}')
    print(f'Target: \n{iris_dataset.target}')
    
# 将 75% 的行数据和其标签作为训练集，剩下 25% 的行数据和其标签作为测试集
# 数据 x 由于是一个二维数组，所以用大写的 X 表示
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset.data, iris_dataset.target, random_state=0
)

# 使用 k 邻近算法构建模型
knn = KNeighborsClassifier(n_neighbors=1)
# fit 方法返回的是 knn 对象本身
knn.fit(X_train, y_train)

# 对数据进行预测，假设我们在野外发现了一朵 iris
# 花萼长 5 cm 宽 2.9 cm，花瓣长 1 cm 宽 0.2 cm
X_new = np.array([[5, 2.9, 1, 0.2]])

# 调用 knn 对其进行预测
prediction = knn.predict(X_new)
print(f'Prediction: {prediction}')
print(f'Prediction target name: {iris_dataset.target_names[prediction]}')

# 利用测试集，评估模型
y_pred = knn.predict(X_test)
print(f'Test set predictions: \n{y_pred}')
print(f'Test set score: {np.mean(y_pred == y_test):.2f}')