import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer,make_blobs, load_wine, load_diabetes,load_iris, make_classification
dataset = pd.read_csv('datasets/creditcard.csv')
cl1_data = dataset.loc[dataset.iloc[:, -1] == 0]
cl2_data = dataset.loc[dataset.iloc[:, -1] == 1]
dataset = pd.concat([cl1_data.iloc[:2000], cl2_data])
# dataset = pd.read_table('datasets/german.data-numeric',header=None, sep='\s+')

data=dataset.values
# data =  load_breast_cancer()
# feat = data['data']
# label = data['target']
# data =  np.load('./datasets/rotated_checkerboard2x2_train.npz')
feat =data[:,:-1]
label = data[:,-1]
# print(np.shape(feat))
# print(np.shape(label))
# lbl_names = np.unique(label)
#
# for i in range(len(lbl_names)):
# 	label[label==lbl_names[i]] = i
# label = label.astype(int)
# print(feat)

# feat =data['x']
# feat = StandardScaler().fit_transform(feat)
# label = data['y']
# label = np.ravel(label)
# feat, label = make_blobs(n_features=4, n_samples=1000, centers=3, random_state=0, cluster_std=2)
# feat, label = make_classification(n_features=4, n_samples=1000, n_classes=3, n_clusters_per_class=1)
# x=feat
# y=label
# np.savez('gaussian_clouds_blobs', x, y)
x=PCA(n_components=2).fit_transform(feat)
plt.scatter(x=x[:, 0], y=x[:, 1], c=label, cmap='Set1', s=5)
# plt.scatter(x=x[:, 0], y=x[:, 1])
plt.show()