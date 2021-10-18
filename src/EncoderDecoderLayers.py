import torch.nn as nn
from Transformer_utils import *


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(nhead, d_model, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(nhead, d_model, dropout)
        self.attn_2 = MultiHeadAttention(nhead, d_model, dropout)
        self.ff = FeedForward(d_model, dim_feedforward, dropout).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        # Self attention with decoder mask
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        # Self attention with encoder mask
        #print("Encoder : " +str(e_outputs.shape) + " Decoder: " +str(x2.shape), "Mask: " +str(src_mask.shape))
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 conv_in=False,
                 dilation=1,
                 seq_length=64,
                 d_ff=2048,
                 dropout=0.1,
                 input_d=512,
                 ladder=False):
        super().__init__()
        self.ladder = ladder
        self.conv_in = conv_in
        self.N = N
        self.upsample = nn.Linear(input_d, d_model, bias=False)
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(
            EncoderLayer(d_model=d_model,
                         nhead=heads,
                         dim_feedforward=d_ff,
                         dropout=dropout), N)
        self.dropout_1 = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        #print("Encoding")
        x = src.float()
        if self.conv_in:
            src = torch.relu(self.Conv1(src))
            src = self.dropout_1(src)
            src = self.Conv2(src)
            src = torch.tanh(src)
        # Linearly project our input to the correct size
        x = self.upsample(x)
        # Positional encoding
        x = self.pe(x)
        if self.ladder:
            outs = []
            for i in range(self.N):
                x = self.layers[i](x, mask)
                outs.append(x)
            self.norm(outs[-1])
            return outs
        else:
            for i in range(self.N):
                x = self.layers[i](x, mask)
            return self.norm(x)


class Decoder(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 conv_in=False,
                 dilation=1,
                 seq_length=64,
                 d_ff=2048,
                 dropout=0.1,
                 input_d=512,
                 ladder=False):
        super().__init__()
        self.ladder = ladder
        self.conv_in = conv_in
        self.N = N
        self.upsample = nn.Linear(input_d, d_model, bias=False)
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(
            DecoderLayer(d_model=d_model,
                         nhead=heads,
                         dim_feedforward=d_ff,
                         dropout=dropout), N)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(0.1)

    def forward(self, trg, encoder_outputs, src_mask, trg_mask):
        #print("Decoding")
        x = trg.float()
        if self.conv_in:
            trg = torch.relu(self.Conv1(trg))
            trg = self.dropout_1(trg)
            trg = self.Conv2(trg)
            trg = torch.tanh(trg)
        # Linearly project our input to the correct size
        x = self.upsample(x)
        # Positional encoding
        x = self.pe(x)
        if self.ladder:
            for i in range(self.N):
                # Use the inverse corresponding layer from the encoder to get finer features
                x = self.layers[i](x, encoder_outputs[-1 - i], src_mask,
                                   trg_mask)
            return self.norm(x)
        else:
            for i in range(self.N):
                x = self.layers[i](x, encoder_outputs, src_mask, trg_mask)
            return self.norm(x)
