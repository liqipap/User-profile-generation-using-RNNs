# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:30:39 2018

@author: Sun T.x.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import math
import random
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, 
          decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0][0]
    # encoder_outputs用于保存历史输出，可用作Attention
        
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

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

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    print('Training model...\n')
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_inputs = [variableFromSentence(vocab, random.choice(inputs))
                      for i in range(n_iters)]
    training_targets = [variableFromSentence(vocab, random.choice(targets))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        input_variable = training_inputs[iter - 1]
        target_variable = training_targets[iter - 1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
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

def save_s2s_model():
    print("Saving model...\n")
    modelPath = save_path + objectUser + '-' + \
                str(datetime.now()).split('.')[0].split()[0] + '/'
    if os.path.exists(modelPath) == False:
    		os.mkdir(modelPath)
    torch.save(encoder.state_dict(), modelPath+'encoder.pkl')
    torch.save(decoder.state_dict(), modelPath+'decoder.pkl')
    print("Model saved!\n")

def save_s2s_paras():
    print("Parameters saving...\n")
    encoderKeys = encoder.state_dict().keys()
    encoderParameters = []
    for key in encoderKeys:
        if key=='embedding.weight':
            continue
        encoderParameters.append(encoder.state_dict()[key].cpu().numpy())
        
    decoderKeys = decoder.state_dict().keys()
    decoderParameters = []
    for key in decoderKeys:
        if key=='embedding.weight':
            continue
        decoderParameters.append(decoder.state_dict()[key].cpu().numpy())
     
    resultPath = res_path + objectUser + '/'
    if os.path.exists(resultPath) == False:
        os.mkdir(resultPath)
        
    f = open(resultPath + 'EncodeeParameters.txt','w')
    EncoderVector = []
    for i in range(len(encoderParameters)):
        for j in range(len(encoderParameters[i])):
            if encoderParameters[i][j].size == 1:
                EncoderVector.append(encoderParameters[i][j])
            else:
                for k in range(len(encoderParameters[i][j])):
                    EncoderVector.append(encoderParameters[i][j][k])
    f.writelines(str(EncoderVector))
    f.close()
    
    f = open(resultPath + 'DecodeeParameters.txt','w')
    DecoderVector = []
    for i in range(len(decoderParameters)):
        for j in range(len(decoderParameters[i])):
            if decoderParameters[i][j].size == 1:
                DecoderVector.append(decoderParameters[i][j])
            else:
                for k in range(len(decoderParameters[i][j])):
                    DecoderVector.append(decoderParameters[i][j][k])
    f.writelines(str(DecoderVector))
    f.close()
    print("Parameters saved!\n")

