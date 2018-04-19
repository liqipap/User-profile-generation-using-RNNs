"""
User profile generation using RNNs.
"""
import os
import torch
import time

SOS_token = 0
EOS_token = 1

embedding_dim = 50
hidden_size = 256
MAX_LENGTH = 25
teacher_forcing_ratio = 0.5

# 设置当前工作路径
os.chdir("G:/XDU/1Pri/Proj")

raw_path = "data/raw/"
seg_path = "data/seg/"
save_path = 'models/'
res_path = 'results/'

use_cuda = torch.cuda.is_available()

# 读入词汇表
runfile('code/preprocessing.py')
print("Data preparing...\n")
segText(raw_path, seg_path)
vocab = readLang(seg_path, 'Chinese')

vocab_size = vocab.n_words

# 加载词嵌入
wordEmbeddings, indicator, all_words = load_my_vecs('data/zhwiki/zhwiki/zhwiki_2017_03.sg_50d.word2vec',
                                                    vocab)

userList = os.listdir(seg_path)

# 加载RNN模型
runfile('code/o2mRNN.py')
runfile('code/train_o2m.py')

# Train profile-to-vector model
print("Training Profile-to-vector Model...")
print("Model initializing...")
model = o2mRNN(hidden_size, vocab_size, embedding_dim, wordEmbeddings, dropout_p=0.1)

if use_cuda:
    model = model.cuda()

for user in userList:
    print('Current object: %s' % user)
    objectUser = user
    texts = getTexts(objectUser)
    n_iters = len(texts)
    
    model.initUservec() #每次换用户时初始化用户向量
    
    trainIters(model, n_iters, print_every=200, learning_rate=0.01)
    save_o2m_model()
    save_o2m_paras()
    print("Model 1: User <%s> done!" % objectUser)
    print("-----------------------------------------\n")
    print("Taking a break...")
    time.sleep(60*10)
    print("-----------------------------------------\n")

runfile('code/seq2seq.py')
runfile('code/train_s2s.py')

# Train seq2seq model
print("Training Sequence-to-sequence Model...")
for user in userList:
    print('Current object: %s' % user)
    objectUser = user
    inputs, targets = getPairs(objectUser)
    # 每次训练需要重新初始化encoder-decoder
    encoder = EncoderRNN(vocab_size, embedding_dim, hidden_size, wordEmbeddings)
    decoder = DecoderRNN(hidden_size, vocab_size, embedding_dim, wordEmbeddings, dropout_p=0.1)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    n_iters = len(inputs)
    trainIters(encoder, decoder, n_iters, print_every=500)
    save_s2s_model()
    save_s2s_paras()
    
    del encoder
    del decoder
    print("Model 2: User <%s> done!" % objectUser)
    print("-----------------------------------------\n")
    print("Taking a break...")
    time.sleep(60*10)
    print("-----------------------------------------\n")








