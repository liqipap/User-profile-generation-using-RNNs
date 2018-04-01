
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import os
import time
import math
import random
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

use_cuda = torch.cuda.is_available()

vocab_size = vocab.n_words
embedding_dim = 50
hidden_size = 256
MAX_LENGTH = 15
teacher_forcing_ratio = 0.5

save_path = 'models/'
res_path = 'results/'

os.chdir("G:/XDU/1Pri/Proj")

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def train(target_variable, model, model_optimizer, criterion, max_length=MAX_LENGTH):
    hidden = model.initHidden()
    input = model.initUservec()
    model_optimizer.zero_grad()

    target_length = target_variable.size()[0]

    loss = 0
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            output, hidden = model(input, hidden, di)
            loss += criterion(output, target_variable[di])
            input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            output, hidden = model(input, hidden, di)
            topv, topi = output.data.topk(1)
            ni = topi[0][0]

            input = Variable(torch.LongTensor([[ni]]))
            input = input.cuda() if use_cuda else input

            loss += criterion(output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    model_optimizer.step()

    return loss.data[0] / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(model, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    print('Training model...\n')
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    training_targets = [variableFromSentence(vocab, random.choice(texts))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        target_variable = training_targets[iter - 1]

        loss = train(target_variable, model, model_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.xlabel('Iters (*100)')
    plt.ylabel('Loss')

def save_o2m_model():
    print("Saving model...\n")
    modelPath = save_path + objectUser + '-' + \
                str(datetime.now()).split('.')[0].split()[0] + '/'
    if os.path.exists(modelPath) == False:
    		os.mkdir(modelPath)
    torch.save(model.state_dict(), modelPath+'profile2vec.pkl')
    print("Model saved!\n")

def save_o2m_paras():
    # To-do
    print("Parameters saving...\n")
    paras = model.state_dict()['userVector'].cpu().numpy().reshape(-1)
        
    resultPath = res_path + objectUser + '/'
    if os.path.exists(resultPath) == False:
        os.mkdir(resultPath)
        
    np.save(resultPath + 'profile2vecParameters', paras)
#    f = open(resultPath + 'profile2vecParameters.txt','w')
#    f.writelines(str(paras))
#    f.close()

    print("Parameters saved!\n")

# Start training!
model = o2mRNN(hidden_size, vocab_size, embedding_dim, wordEmbeddings, dropout_p=0.1)

if use_cuda:
    model = model.cuda()

trainIters(model, 80000, print_every=5000)

save_o2m_model()
save_o2m_paras()
