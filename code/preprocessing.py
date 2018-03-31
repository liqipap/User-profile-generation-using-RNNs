# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:55:28 2018

@author: Sun T.x.
"""

import jieba
import os
import sys
import re
import string
import pickle
import unicodedata
import numpy as np
from io import open


SOS_token = 0
EOS_token = 1

MAX_LENGTH = 15
embedding_dim = 50

# 设置当前工作路径
os.chdir("G:/XDU/1Pri/Proj")

raw_path = "data/raw/"
seg_path = "data/seg/"

objectUser = 'Luhan_Star'

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.strip())
    s = re.sub(r"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+",
               r" ", s)
    return s.strip()

def segText(rawPath, segPath):
    dir_list = os.listdir(rawPath)
    
    for _dir in dir_list:
        dir_path = rawPath + _dir + "/"
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            file_path = dir_path + file_name
            raw_file = open(file_path, 'r')
            raw_text = raw_file.read()
            
            seg_text = jieba.cut(raw_text)
            
            seg_dir = segPath + _dir + "/"
            if not os.path.exists(seg_dir):
                os.makedirs(seg_dir)
            seg_file = open(seg_dir + file_name, 'w')
            seg_file.write(" ".join(seg_text))
            
            raw_file.close()
            seg_file.close()
    print("Segmentation Done!\n")

def readLang(path, lang):
    print("Reading lines...\n")
    vocab = Lang(lang)
    dir_list = os.listdir(path)
    for _dir in dir_list:
        dir_path = path + _dir + "/"
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            file_path = dir_path + file_name
            file = open(file_path, 'r')
            for sent in file.readlines():
                sent = normalizeString(sent)
                vocab.addSentence(sent)
                #print(sent + '\n')
    print("Language loaded!\n")
    print("Word counts: %d\n" % vocab.n_words)
    return vocab

def filterSentence(sent):
    return len(sent.split(' ')) > MAX_LENGTH or len(sent.split(' ')) < 3

def getPairs(obj):
    path = seg_path + obj + '/'
    textFile = open(path + 'Text.txt', 'r')
    commentFile = open(path + 'Comment.txt', 'r')
    textLines = textFile.read().strip().split('\n')
    commentLines = commentFile.read().strip().split('\n')
    
    inputs = [normalizeString(l) for l in textLines]
    targets = [normalizeString(l) for l in commentLines]
    
    for i in range(len(inputs)-1, -1, -1):
        if filterSentence(inputs[i]) or filterSentence(targets[i]):
            del inputs[i]
            del targets[i]
    
    print("Get %s pairs.\n" % len(inputs))
    return inputs, targets

def load_my_vecs(path, vocab):
    # path为预训练的word2vec文件路径
    word_vecs = np.random.rand(vocab_size, embedding_dim)
    avg_vec = np.zeros((1, embedding_dim))
    indicator = np.zeros((1, vocab_size))
    words = []
    with open(path, encoding = 'utf-8') as f:
        count = 0
        lines = f.readlines()[1:] # 第一行不读
        for line in lines:
            values = line.split(" ")
            word = values[0]
            words.append(word)
            index = vocab.word2index.get(word, -1)
            if index != -1:        # 找到了对应的词嵌入
                count += 1
                indicator[0][index] = 1
                for i, val in enumerate(values):
                    if i == 0:
                        continue
                    if i > embedding_dim:
                        break
                    word_vecs[index][i-1] = float(val)
                avg_vec += word_vecs[index]
                
        avg_vec = avg_vec/count
        for i in range(vocab_size):
            if indicator[0][i] == 0:
                word_vecs[i] = avg_vec
                
    print("IOV count: %d\n" % count)
    print("OOV count: %d\n" % (vocab_size - count))
    return word_vecs, indicator, words

# Start
print("Data preparing...\n")
segText(raw_path, seg_path)
vocab = readLang(seg_path, 'Chinese')

vocab_size = vocab.n_words

wordEmbeddings, indicator, all_words = load_my_vecs('data/zhwiki/zhwiki/zhwiki_2017_03.sg_50d.word2vec',
                                                    vocab)

inputs, targets = getPairs(objectUser)
