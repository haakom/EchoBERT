import copy

import h5py
import numpy as np


class Data_utils():
    def normalize_single_vector(self, vector, mean, std, max_value):
        # TODO Maybe we can just use the vector's mean and std, as the mass of fish is the same for each vector?
        #vector = vector/max_value
        out = (vector - mean) / std
        #out = torch.clamp(out, -1, 1)
        return out

    def normalize(self, dataset, mean, std):
        dataset = (dataset - mean) / std
        # Clip dataset, so that we get a proper distribution
        #dataset = np.clip(dataset, -1, 1)
        return dataset

    def cut_dataset(self, start_at, cut_at, dataset, cut_ends):
        """
        Cuts the dataset to fit our model vector size
        """

        if cut_ends:
            dataset = dataset[:, start_at:cut_at]
        return dataset

    def apply_log(self, source):
        """
        Applies a log function to bring the dataset values down to a range 0-1
        :param source: The dataset to work on
        :return: The log version of the dataset
        """
        # Bring values down to range between 0 and 1
        source = np.log(source)
        source[np.isneginf(source)] = 0

        max_value = np.max(source)
        source = source / np.max(max_value)
        return source

    def merge_sequences(self, data):
        data = np.reshape(data,
                          newshape=(data.shape[0] * data.shape[1],
                                    data.shape[2]))
        return data

    def shuffle_data(self, dataset):
        np.random.seed(40)
        idx = np.random.permutation(dataset.shape[0])
        dataset = dataset[idx]
        return dataset

    def shuffle_disease_data(self, dataset, targets):
        #seq_lenght = dataset.shape[1]
        #dim = dataset.shape[2]
        # Reshape data for resampling
        #dataset = np.reshape(dataset, newshape=(dataset.shape[0], seq_lenght*dim))
        # Initialize oversampler
        #ros = RandomOverSampler(random_state=0)
        # Resample data through oversampling
        #dataset, targets = ros.fit_resample(dataset, np.ravel(targets))
        # Reshape data back to batch, seq, dim again
        #dataset = np.reshape(dataset, newshape=(dataset.shape[0], seq_lenght, dim))
        #targets = np.reshape(targets, newshape=(targets.shape[0], 1))
        # Shuffle data
        np.random.seed(40)
        idx = np.random.permutation(dataset.shape[0])
        new_dataset = dataset[idx]
        new_targets = targets[idx]
        return new_dataset, new_targets

    def oversample_data(self, dataset, targets):
        seq_lenght = dataset.shape[1]
        dim = dataset.shape[2]
        # Reshape data for resampling
        dataset = np.reshape(dataset,
                             newshape=(dataset.shape[0], seq_lenght * dim))
        # Initialize oversampler
        ros = RandomOverSampler(random_state=0)
        # Resample data through oversampling
        dataset, targets = ros.fit_resample(dataset, np.ravel(targets))
        # Reshape data back to batch, seq, dim again
        dataset = np.reshape(dataset,
                             newshape=(dataset.shape[0], seq_lenght, dim))
        targets = np.reshape(targets, newshape=(targets.shape[0], 1))
        return dataset, targets

    def split_train_test(self, data, split_at=0.8):
        split_point = int(data.shape[0] * split_at)
        train_data = data[:split_point]
        test_data = data[split_point:]
        return train_data, test_data

    def generate_seq_data(self, path, seq_length):
        """
        Divide dataset into sequences of length seq_length*2
        :param dataset: the dataset
        :param seq_length: lenght of each sequence
        :return: sequence divided dataset
        """
        f = h5py.File(path + '.hdf5', 'r')
        dataset = f['mydataset'][()]

        dataset = self.divide_in_sequences(dataset, seq_length * 2)
        dataset = self.shuffle_data(dataset)
        train_data, test_data = self.split_train_test(dataset)
        #train_data = self.merge_sequences(train_data)
        #test_data = self.merge_sequences(test_data)
        train_data = self.apply_log(train_data)
        test_data = self.apply_log(test_data)
        mean = np.mean(train_data)
        std = np.std(train_data)
        max_val = np.max(train_data)

        return train_data, test_data, mean, std, max_val

    def load_disease_data(self,
                          data_path,
                          seq_length,
                          train=True,
                          targets_name="_targets"):
        print("Loading data: " + targets_name)
        f = h5py.File(data_path + '_data.hdf5', 'r')
        dataset = f['mydataset'][()]
        f = h5py.File(data_path + targets_name + '.hdf5', 'r')
        targets = f['mydataset'][()]

        dataset = self.divide_in_sequences(dataset, seq_length * 2)
        dataset = self.apply_log(dataset)
        targets = self.divide_in_sequences(targets, seq_length * 2)
        tmp_targets = np.zeros((targets.shape[0], 1))
        for batch in range(targets.shape[0]):
            tmp_targets[batch] = np.round(np.mean(targets[batch]))
        targets = tmp_targets
        if train:
            dataset, targets = self.shuffle_disease_data(dataset, targets)

            mean = np.mean(dataset)
            std = np.std(dataset)
            max_val = np.max(dataset)

            return dataset, targets, mean, std, max_val
        else:
            return dataset, targets

    def split_train_val(self, dataset, seq_length):
        dataset = self.divide_in_sequences(dataset, seq_length * 2)
        train_data, val_data = self.split_train_test(dataset)
        train_data = self.merge_sequences(train_data)
        val_data = self.merge_sequences(val_data)
        return train_data, val_data

    def divide_in_sequences(self, dataset, seq_length):
        """
        Divide dataset into sequences of length seq_length
        :param dataset: the dataset
        :param seq_length: lenght of each sequence
        :return: sequence divided dataset
        """
        num_possible_sqeuences = dataset.shape[0] // seq_length
        num_vectors_to_use = num_possible_sqeuences * seq_length
        dataset = dataset[0:num_vectors_to_use]

        return dataset.reshape(num_possible_sqeuences, seq_length,
                               dataset.shape[1])

    def normalize_dataset(self,
                          source,
                          decoder,
                          mean,
                          std,
                          return_targets=True):
        """
        Normalize our data for the optimal network performance.
        :param source: encoder data
        :param targets: MSE calculation data
        :return: normalized src and decoder data as well as clipped targets data to make it fit the sigmoid activation
        better.
        """
        if return_targets:
            targets = copy.deepcopy(decoder)
            targets = self.normalize(targets, mean, std)

        source = self.normalize(source, mean, std)
        decoder = self.normalize(decoder, mean, std)

        if return_targets:
            return source, decoder, targets
        else:
            return source, decoder

    def generate_next_sequence_data(self, source, decoder, targets):
        """
        Generate a dataset consisting of sequences where the goal is to predict the next sequence.
        The source and targets data are dividied into sequences consisting of src having the first sequence and decoder and targets   having the following one. They den iterate forward.
        :param source: src data for the encoder
        :param targets: target data for MSE calculation
        :param decoder: dec data for the decoder
        :return: sequence split datasets.
        """
        source = np.concatenate((source[0::2, :, :], source[1::2, :, :]),
                                axis=0)

        decoder = np.concatenate((decoder[1::2, :, :], decoder[0::2, :, :]),
                                 axis=0)

        targets = np.concatenate((targets[1::2, :, :], targets[0::2, :, :]),
                                 axis=0)

        return source, decoder, targets

    def divide_into_train_and_val(self,
                                  source,
                                  targets,
                                  decoder,
                                  split_percentage=0.9):
        """
        Split dataset into training and dataset with the split percentage given.
        :param source: source data
        :param targets: target data
        :param decoder: decoder data
        :param split_percentage: percentage to split on
        :return: training and validation data
        """
        assert source.shape[0] == decoder.shape[0]
        split_point = int(source.shape[0] * split_percentage)

        train_source = source[0:split_point]
        train_decoder = decoder[0:split_point]
        train_targets = targets[0:split_point]

        val_source = source[split_point:]
        val_decoder = decoder[split_point:]
        val_targets = targets[split_point:]

        return train_source, train_decoder, train_targets, val_source, val_decoder, val_targets

    def test_dataset(self, dataloader):
        input_d = 192

        batch = next(iter(dataloader))
        # Get source and target data
        src = batch['encoder']
        dec = batch['decoder']
        wrng_seq = batch['target']

        input_d = src.shape[2]

        self.plot_data(src, dec, input_d)

    def plot_data(self, src_, dec_, input_d):
        src_plot = np.zeros(shape=(input_d, 1))
        dec_plot = np.zeros(shape=(input_d, 1))
        wrng_plot = np.zeros(shape=(input_d, 1))

        for seq in range(src_.shape[0]):
            src = src_[seq]
            dec = dec_[seq]

            src = src
            dec = dec

            src_echo = np.rot90(src, 1)
            dec_echo = np.rot90(dec, 1)

            src_plot = np.concatenate((src_plot, src_echo), axis=1)
            dec_plot = np.concatenate((dec_plot, dec_echo), axis=1)

        self.do_plot(src_plot, dec_plot)

    def do_plot(self, src_plot, dec_plot):
        import matplotlib.pyplot as plt
        import matplotlib
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        matplotlib.rcParams['figure.figsize'] = [30, 30]
        matplotlib.rcParams.update({'font.size': 14})

        plot_length = 1000
        start = 0
        src_show = src_plot[:, start:start + plot_length]
        dec_show = dec_plot[:, start:start + plot_length]
        #wrng_show = wrng_plot[:, start:start+plot_length]

        if True:
            fig = plt.figure(figsize=(30, 30))
            fig.suptitle("Src, Dec and Wrng comparison:", fontsize=26)
            ax1 = fig.add_subplot(311)
            im1 = ax1.imshow(src_show, cmap="gnuplot2")
            ax1.set_title("Src:", fontsize=24)

            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')

            ax2 = fig.add_subplot(312)
            im2 = ax2.imshow(dec_show, cmap="gnuplot2")
            ax2.set_title("Dec:", fontsize=24)

            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax, orientation='vertical')
            plt.show()
