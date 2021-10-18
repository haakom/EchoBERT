import numpy as np
import random
from data_utils import Data_utils
from torch.utils.data import Dataset


class SyntheticEchoDataset(Dataset):
    def __init__(self,
                 dataset=None,
                 seq_length=None,
                 normalize=False,
                 mean=None,
                 std=None):

        self.seq_lenght = seq_length
        self.normalize = normalize
        self.data_utils = Data_utils()
        self.mean = mean
        self.std = std

        self.encoder, self.decoder = self.split_encoder_decoder(
            dataset, seq_length)

        if normalize:
            # Normalize data
            self.encoder, self.decoder, self.targets = self.data_utils.normalize_dataset(
                self.encoder, self.decoder, self.mean, self.std)

    def split_encoder_decoder(self, data, seq_length):
        encoder = data[:, :seq_length, :]
        decoder = data[:, seq_length:, :]
        return encoder, decoder

    def __len__(self):
        return self.encoder.shape[0]

    def __getitem__(self, idx):
        sample = {
            'encoder': self.encoder[idx],
            'decoder': self.decoder[idx],
            'target':
            self.targets[random.choice(np.arange(self.targets.shape[0]))]
        }

        return sample
