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
import os
import numpy as np
import re

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

os.chdir("G:/XDU/1Pri/Proj/code")
resPath = '../results/'

def readFile(file):
    text = file.read()
    s = re.sub(r"[\[\],]+", r" ", text)
    values = s.strip().split('  ')
    ret = np.zeros((1, 236544))
    for i, val in enumerate(values):
        ret[0][i] = float(val)
    return ret

userList = os.listdir(resPath)
n_users = len(userList)
mainVecs = np.zeros((n_users, 50))
encoderVecs = np.zeros((n_users, 236544))

userID = []
for i, user in enumerate(userList):
    mainfileName = resPath + user + '/profile2vecParameters.npy'
    encoderfileName = resPath + user + '/EncodeeParameters.txt'
    #decoderfileName = resPath + user + '/DecodeeParameters.txt'
    
    newVec = np.load(mainfileName).reshape(1,50)  # 列向量
    encoderFile = open(encoderfileName, 'r')
    #decoderFile = open(decoderfileName, 'r')
    encoderVec = readFile(encoderFile)
    #decoderVec = readFile(decoderFile)
    
    mainVecs[i] = newVec
    encoderVecs[i] = encoderVec
    userID.append(user)

# Part1: Main user vectors from o2m-RNN model.
# Source: ../results/objectUser/profile2vecParameters.npy

# Part2: Supplemental user vectors from seq2seq model.
# Source: ../results/objectUser/EncoderParameters.txt &
#           ../results/objectUser/DecodeeParameters.txt


#X_tsne = TSNE(learning_rate=100).fit_transform(mainVecs)
X_pca = PCA().fit_transform(mainVecs)
# userVecs: array, n_samples * n_features
plt.figure(figsize=(10,8), dpi=120)

fig1 = plt.subplot(111)
fig1.scatter(X_pca[:, 0], X_pca[:, 1])

#fig2 = plt.subplot(122)
#fig2.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i, user in enumerate(userList):
    fig1.text(X_pca[i, 0], X_pca[i, 1], user)
#    fig2.text(X_tsne[i, 0], X_tsne[i, 1], user)

