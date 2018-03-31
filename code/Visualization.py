# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:37:03 2018

@author: Sun.T.x
"""

# The code uses t-SNE(t-distributed stochastic neighbor embedding) algorithm
# and PCA to implemente dimensionality reduction.

# To-do
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Part1: Main user vectors from o2m-RNN model.
# Source: ./results/objectUser/Profile2vec.txt

# Part2: Supplemental user vectors from seq2seq model.
# Source: ./results/objectUser/EncoderParameters.txt &
#           ./results/objectUser/DecoderParameters.txt

X_tsne = TSNE(learning_rate=100).fit_transform(userVecs)
X_pca = PCA().fit_transform(userVecs)
# userVecs: array, n_samples * n_features

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1])

