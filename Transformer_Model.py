import torch.nn as nn
import torch
import torch.functional as F
from Transformer_utils import PositionalEncoding, get_clones, gelu
from EncoderDecoderLayers import EncoderLayer, DecoderLayer, Encoder, Decoder


class Custom_Transformer(nn.Module):
    def __init__(self,
                 input_d=232,
                 d_model=512,
                 nheads=8,
                 nlayers=6,
                 seq_length=128,
                 dropout=0.1,
                 d_ff=2048,
                 classifier_dropout=0.1,
                 disease=False):
        super(Custom_Transformer, self).__init__()

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nheads,
                                          dim_feedforward=d_ff,
                                          num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers,
                                          dropout=dropout)

        self.upsample = nn.Linear(input_d, d_model, bias=False)

        self.pe = PositionalEncoding(d_model=d_model)
        self.disease = disease
        if not self.disease:
            self.prob1 = nn.Conv1d(in_channels=seq_length,
                                   out_channels=512,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            self.prob11 = nn.Linear(d_model, 1)
            self.prob2 = nn.Linear(d_model * seq_length, 1)
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_11 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)
        else:
            self.prob4 = nn.Linear(d_model * seq_length, 1)
            #self.prob4 = nn.Conv1d(in_channels=seq_length, out_channels=1, kernel_size=3, stride=1, padding=1)
            self.dropout_4 = nn.Dropout(classifier_dropout)
            #self.pool = nn.AvgPool1d(kernel_size=d_model)

    def forward(self, src, trg, src_mask, trg_mask):
        src = self.upsample(src)
        trg = self.upsample(trg)

        src = self.pe(src)
        src = src.permute(1, 0, 2)

        trg = self.pe(trg)
        trg = trg.permute(1, 0, 2)

        out = self.transformer(src=src,
                               tgt=trg,
                               src_mask=src_mask,
                               tgt_mask=trg_mask)
        if self.disease:
            return self.disease_output(out)
        else:
            return self.output(out)

    def output(self, transformer_out):
        probs = 0
        is_next = 0
        transformer_out = transformer_out.permute(1, 0, 2)
        recon_vec = transformer_out.reshape(
            transformer_out.shape[0],
            transformer_out.shape[1] * transformer_out.shape[2])

        # Altered_vectors head
        altered_vecs = self.prob1(self.dropout_1(torch.relu(transformer_out)))
        altered_vecs = self.prob11(self.dropout_11(torch.relu(altered_vecs)))

        # is_next head
        is_next = self.prob2(self.dropout_2(torch.relu(recon_vec)))

        return transformer_out, altered_vecs, is_next

    def disease_output(self, transformer_out):
        transformer_out = transformer_out.permute(1, 0, 2)
        recon_vec = transformer_out.reshape(
            transformer_out.shape[0],
            transformer_out.shape[1] * transformer_out.shape[2])
        #transformer = torch.relu(transformer_out.permute(1, 0, 2))
        disease_head = self.prob4(self.dropout_4(torch.relu(recon_vec)))

        # print(transformer_out.shape)
        #disease_head = self.prob4(self.dropout_4(torch.relu(transformer_out)))
        # print(disease_head.shape)
        #disease_head = self.pool(disease_head).view(disease_head.shape[0], 1)

        # print(disease_head.shape)

        return disease_head


class BERT(nn.Module):
    def __init__(self,
                 input_d=232,
                 d_model=512,
                 nheads=8,
                 nlayers=6,
                 seq_length=128,
                 dropout=0.1,
                 d_ff=2048,
                 classifier_dropout=0.1,
                 disease=False):
        super(BERT, self).__init__()
        # Build BERT encoder
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model,
                                             nhead=nheads,
                                             dim_feedforward=d_ff,
                                             dropout=dropout,
                                             activation='gelu'),
            num_layers=nlayers,
            norm=torch.nn.LayerNorm(d_model))

        # set disease or not
        self.disease = disease

        # Upsample input
        self.upsample = nn.Linear(input_d, d_model, bias=False)

        # Do positional encodings
        self.pe = PositionalEncoding(d_model=d_model)

        if not self.disease:
            self.prob1 = nn.Linear(d_model, 1)
            self.prob2 = nn.Linear(d_model * seq_length * 2, 1)
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)
        else:
            self.prob4 = nn.Linear(d_model * seq_length * 2, 1)
            #self.prob4 = nn.Conv1d(in_channels=seq_length, out_channels=1, kernel_size=3, stride=1, padding=1)
            self.dropout_4 = nn.Dropout(classifier_dropout)
            #self.pool = nn.AvgPool1d(kernel_size=d_model)

    def forward(self, src, dec, src_mask, trg_mask):
        dec = dec + 1
        src = torch.cat((src, dec), dim=1)
        # print(src.shape)
        src = self.upsample(src)
        src = self.pe(src)
        src = src.permute(1, 0, 2)

        encoded = self.encoder(src)

        if self.disease:
            return self.disease_output(encoded)
        else:
            return self.output(encoded)

    def output(self, transformer_out):
        probs = 0
        is_next = 0
        transformer_out = transformer_out.permute(1, 0, 2)
        recon_vec = transformer_out.reshape(
            transformer_out.shape[0],
            transformer_out.shape[1] * transformer_out.shape[2])

        # Altered_vectors head
        altered_vecs = self.prob1(self.dropout_1(torch.relu(transformer_out)))

        # is_next head
        is_next = self.prob2(self.dropout_2(torch.relu(recon_vec)))

        return transformer_out, altered_vecs, is_next

    def disease_output(self, transformer_out):
        transformer_out = transformer_out.permute(1, 0, 2)
        recon_vec = transformer_out.reshape(
            transformer_out.shape[0],
            transformer_out.shape[1] * transformer_out.shape[2])
        #transformer = torch.relu(transformer_out.permute(1, 0, 2))
        disease_head = self.prob4(self.dropout_4(torch.relu(recon_vec)))

        # print(transformer_out.shape)
        #disease_head = self.prob4(self.dropout_4(torch.relu(transformer_out)))
        # print(disease_head.shape)
        #disease_head = self.pool(disease_head).view(disease_head.shape[0], 1)

        # print(disease_head.shape)

        return disease_head


class Linear_transformer(nn.Module):
    def __init__(self,
                 d_model,
                 N,
                 heads,
                 seq_length,
                 d_ff,
                 dropout,
                 input_d,
                 find_next=False,
                 n_rec_vectors=200):
        super(Linear_transformer, self).__init__()
        self.find_next = find_next
        self.input_d = input_d
        self.encoder = Encoder(d_model=d_model,
                               N=N,
                               heads=heads,
                               seq_length=seq_length,
                               d_ff=d_ff,
                               dropout=dropout,
                               input_d=input_d)
        self.decoder = Decoder(d_model=d_model,
                               N=N,
                               heads=heads,
                               seq_length=seq_length,
                               d_ff=d_ff,
                               dropout=dropout,
                               input_d=input_d)

        self.prob1 = nn.Linear(d_model, 1)
        self.prob2 = nn.Linear(d_model * seq_length, 512)
        self.prob3 = nn.Linear(512, 1)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, src, trg, src_mask, trg_mask):
        decoder_output = self.decode(trg, self.encode(src, src_mask), src_mask,
                                     trg_mask)
        return self.output(torch.relu(decoder_output))

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, trg, encoder_outputs, src_mask, trg_mask, print=False):
        return self.decoder(trg, encoder_outputs, src_mask, trg_mask)

    def output(self, transformer_out):
        probs = 0
        is_next = 0

        recon_vec = decoder_out.view(
            transformer_out.shape[0],
            transformer_out.shape[1] * transformer_out.shape[2])

        # Altered_vectors head
        altered_vecs = self.prob1(self.dropout_1(transformer_out))

        # is_next head
        is_next = torch.relu(self.prob2(self.dropout_2(recon_vec)))
        is_next = self.prob3(self.dropout_3(is_next))

        return transformer_out, altered_vecs, is_next
