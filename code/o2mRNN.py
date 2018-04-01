import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

use_cuda = torch.cuda.is_available()

class o2mRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_dim, 
                 wordVecs, n_layers=1, dropout_p=0.1):
        super(o2mRNN, self).__init__()
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
        
        self.userVector = self.initUservec()
    
    def forward(self, input, hidden, flag):
        if flag==0:
            output = self.dropout(self.userVector)
            output, hidden = self.gru(output, hidden)
            
            output = self.softmax(self.out(output[0]))
            return output, hidden
        else:
            embedded = self.embedding(input).view(1, 1, -1)
            embedded = self.dropout(embedded)
        
            output = embedded
            output, hidden = self.gru(output, hidden)
        
            output = self.softmax(self.out(output[0]))
            return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
    def initUservec(self):
        userVector = Parameter(torch.rand(1, 1, 50)-0.5, requires_grad=True)
        return userVector
#        if use_cuda:
#            return userVector.cuda()
#        else:
#            return userVector
        
