import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 N=8,
                 hidden_size=1000,
                 dropout=0.2,
                 LSTM=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        if LSTM:
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=N,
                               batch_first=True,
                               bidirectional=True)
        else:
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=N,
                              batch_first=True,
                              bidirectional=True)

    def forward(self, x, hidden):
        x, hidden_state = self.rnn(x, hidden)
        return x, hidden_state

    def first_hidden(self, batch_size, N):
        return Variable(
            torch.FloatTensor(N, batch_size, self.hidden_size).zero_()).cuda()


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 N=8,
                 hidden_size=1000,
                 dropout=0.2,
                 LSTM=True):
        super(Decoder, self).__init__()
        if LSTM:
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=N,
                               batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=N,
                              batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):
        x, _ = self.rnn(x, hidden)
        #output = x.view(1, x.size(2))
        #linear = torch.relu(self.linear(x))
        return x
