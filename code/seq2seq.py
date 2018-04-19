
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, wordVecs, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        wordVecs = np.array(wordVecs)
        self.embedding.weight.data.copy_(torch.from_numpy(wordVecs))
        
        self.gru = nn.GRU(embedding_dim, hidden_size, n_layers)

        
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)

        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        # Tip: GRU只需要初始化h_0一个参数，LSTM需要初始化h_0和c_0两个参数
        
        if use_cuda:
            return result.cuda()
        else:
            return result
        
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_dim, 
                 wordVecs, n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        wordVecs = np.array(wordVecs)
        self.embedding.weight.data.copy_(torch.from_numpy(wordVecs))
        
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.gru = nn.GRU(embedding_dim, hidden_size, n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
