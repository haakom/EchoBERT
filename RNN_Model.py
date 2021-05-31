import torch.nn as nn
import torch
from RNN_utils import Encoder, Decoder
from Transformer_utils import PositionalEncoding


class RNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 N=8,
                 batch_size=8,
                 seq_length=10,
                 dropout=0.2,
                 LSTM=True,
                 disease=False):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.N = N
        self.disease = disease
        self.LSTM = LSTM
        self.encoder = Encoder(input_size=input_size,
                               N=N,
                               hidden_size=self.hidden_size,
                               dropout=dropout,
                               LSTM=LSTM)
        # Do positional encodings
        #self.pe = PositionalEncoding(d_model=input_size)

        if not self.disease:
            self.prob1 = nn.Linear(hidden_size * 2, 1)
            self.prob2 = nn.Linear(hidden_size * 2 * seq_length * 2, 1)
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)
        else:
            self.prob4 = nn.Linear(hidden_size * 2 * seq_length * 2, 1)
            #self.prob4 = nn.Conv1d(in_channels=seq_length, out_channels=1, kernel_size=3, stride=1, padding=1)
            self.dropout_4 = nn.Dropout(dropout)
            #self.pool = nn.AvgPool1d(kernel_size=d_model)

    def forward(self, src, dec, src_mask=None, trg_mask=None):
        dec = dec + 1
        src = torch.cat((src, dec), dim=1)
        # print(src.shape)
        #src = self.pe(src)
        out, hidden = self.encode(src)

        #encoder_output, hidden = self.encode(src)
        if self.disease:
            return self.disease_output(out)
        else:
            return self.output(out)

    def encode(self, src):
        if self.LSTM:
            hidden = (self.encoder.first_hidden(src.shape[0], self.N * 2),
                      self.encoder.first_hidden(src.shape[0], self.N * 2))
        else:
            hidden = self.encoder.first_hidden(src.shape[0], self.N)
        # print(hidden.shape)
        x, hidden = self.encoder(src, hidden)
        # print('Encoded')
        return x, hidden

    def output(self, rnn_out):
        probs = 0
        is_next = 0
        recon_vec = rnn_out.reshape(rnn_out.shape[0],
                                    rnn_out.shape[1] * rnn_out.shape[2])

        # Altered_vectors head
        altered_vecs = self.prob1(self.dropout_1(torch.relu(rnn_out)))

        # is_next head
        is_next = self.prob2(self.dropout_2(torch.relu(recon_vec)))

        return rnn_out, altered_vecs, is_next

    def disease_output(self, decoder_out):
        recon_vec = decoder_out.reshape(
            decoder_out.shape[0], decoder_out.shape[1] * decoder_out.shape[2])

        disease_head = self.prob4(self.dropout_4(torch.relu(recon_vec)))

        return disease_head
