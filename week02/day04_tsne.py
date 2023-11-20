# t-SNE 流形学习

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

digits = load_digits()

# 构建一个 PCA 模型

pca = PCA(n_components=2)
pca.fit(digits.data)
# 将 digits 数据变换到前两个主成分的方向上
digits_pca = pca.transform(digits.data)
colors = [
    '#476a2a', '#785188',
    '#bd3430', '#4a2d4e',
    '#875525', '#a83683',
    '#4e655e', '#853541',
    '#3a3120', '#535d8e' 
]

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
ax.set_ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())

for i in range(len(digits.data)):
    # 将数据绘制成实际的文本
    ax.text(
        x=digits_pca[i, 0], 
        y=digits_pca[i, 1],
        s=str(digits.target[i]),
        color=colors[digits.target[i]],
        fontdict={
            'weight': 'bold',
            'size': 9
        }
    )
    
ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')

fig.savefig('day04_01.png', bbox_inches='tight', dpi=300)

# 使用 t-SNE
tsne = TSNE(random_state=42)
# 使用 fit_transform 而不是 fit
digits_tsne = tsne.fit_transform(digits.data)

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
ax.set_ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    # 将数据绘制成实际的文本
    ax.text(
        x=digits_tsne[i, 0], 
        y=digits_tsne[i, 1],
        s=str(digits.target[i]),
        color=colors[digits.target[i]],
        fontdict={
            'weight': 'bold',
            'size': 9
        }
    )
    
ax.set_xlabel('t-SNE feature 0')
ax.set_ylabel('t-SNE feature 1')

fig.savefig('day04_02.png', bbox_inches='tight', dpi=300)
