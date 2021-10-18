from sklearn.metrics import matthews_corrcoef
from collections import OrderedDict
import pytorch_lightning as pl
from test_tube import Experiment
from Transformer_Model import BERT
from RNN_Model import RNN
from DiseaseEchoDataset import DiseaseEchoDataset
from Data_loading import SyntheticEchoDataset
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from Train_functions import calculate_disease_loss, calculate_synthetic_loss, randomize_next_sequence, create_masked_LM_vectors
from data_utils import Data_utils
from argparse import ArgumentParser
from test_tube.hpc import HyperOptArgumentParser
import glob

<<<<<<< HEAD
# Class containing EchoBERT
=======
>>>>>>> origin/master
class EchoTransformer(pl.LightningModule):
    def __init__(self, params):
        super(EchoTransformer, self).__init__()
        self.data_utils = Data_utils()
        self.params = params
        self.load_data_into_memory()
        self.model = self.get_model()

    def get_model(self):
        # Are we training a BERT model or RNN
        if self.params.model_type == 'BERT':
            model = BERT(input_d=self.params.input_d,
                         d_model=self.params.d_model,
                         dropout=self.params.dropout,
                         nlayers=self.params.n_layers,
                         nheads=self.params.n_heads,
                         seq_length=self.params.seq_length,
                         d_ff=self.params.d_ff,
                         classifier_dropout=self.params.classifier_dropout,
                         disease=self.params.disease_model)
        else:
            model = RNN(input_size=self.params.input_d,
                        output_size=self.params.input_d,
                        hidden_size=self.params.d_model,
                        N=self.params.n_layers,
                        batch_size=self.params.batch_size,
                        seq_length=self.params.seq_length,
                        dropout=self.params.dropout,
                        LSTM=True,
                        disease=self.params.disease_model)
        # Are we loading a model?
        if self.params.load_model:
            print('Loading lightning model')
            checkpoint_path = self.params.lightning_model_path + \
                self.params.cage_nr+'/version_0/checkpoints/'  # synthetic.ckpt'
            checkpoint_path = sorted(glob.glob(checkpoint_path + '*'))[0]
            print(checkpoint_path)
            device = torch.device('cuda')
            model.load_state_dict(torch.load(checkpoint_path,
                                             map_location=device),
                                  strict=False)
        return model

    def forward(self, batch, batch_idx, forward_type):
        src = batch['encoder']
        dec = batch['decoder']
        dec_mask = None
        src_mask = None

        # If we are training a disease model
        if self.params.disease_model:
            targets = batch['target']
            preds = self.model(src.float(), dec.float(), src_mask, dec_mask)
            loss, f1 = calculate_disease_loss(preds=preds, targets=targets)

            tensorboard_logs = {
                forward_type + 'loss': loss,
                forward_type + 'f1_score': f1
            }
        # If we are training a model on synthetic targets
        else:
            wrng_seq = batch['target']
            # Randomize the next sequence for is_next_sequence prediction
            dec, wrng_seq, is_next = randomize_next_sequence(dec, wrng_seq)

            # Set random x% of data to zero vector
            src, dec, targets = create_masked_LM_vectors(
                self.params.mask_rate, src, dec, wrng_seq)

            transformer_out, probs, is_next_pred = self.model(
                src.float(), dec.float(), src_mask, dec_mask)

            loss, next_seq, altered_loss, is_next_acc, altered_acc = calculate_synthetic_loss(
                probs=probs,
                is_next_pred=is_next_pred,
                targets=targets,
                is_next=is_next)
            tensorboard_logs = {
                forward_type + 'loss': loss,
                forward_type + 'is_next_loss': next_seq,
                forward_type + 'is_next_acc': is_next_acc,
                forward_type + 'altered_loss': altered_loss,
                forward_type + 'altered_acc': altered_acc
            }
            # These arent used, but must be assigned
            targets = 0.0
            preds = 0.0
        return loss, tensorboard_logs, targets, preds

    def training_step(self, batch, batch_nb):
        loss, tensorboard_logs, targets, preds = self.forward(
            batch, batch_nb, 'train_')
        output = OrderedDict({'loss': loss, 'log': tensorboard_logs})
        return output

    def validation_step(self, batch, batch_nb):
        # Run through model
        loss_val, tensorboard_logs, targets_val, preds_val = self.forward(
            batch, batch_nb, 'val_')

        # Create an ordered dict
        output = OrderedDict({
            'val_loss': loss_val,
            'log': tensorboard_logs,
            'val_preds': preds_val,
            'val_targets': targets_val
        })
        return output

    def validation_end(self, outputs):
        val_loss_mean = 0
        f1_score_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
        if self.params.disease_model:
            # Concat all validation preds and targets
            for i, output in enumerate(outputs):
                f1_score_mean += output['log']['val_f1_score']
                if i == 0:
                    all_preds = output['val_preds'].cpu().detach()
                    all_targets = output['val_targets'].cpu().detach()
                else:
                    all_preds = torch.cat(
                        (all_preds, output['val_preds'].cpu().detach()), 0)
                    all_targets = torch.cat(
                        (all_targets, output['val_targets'].cpu().detach()), 0)
            # Apply sigmoid to predictions
            all_preds = torch.round(torch.sigmoid(all_preds))
            # Change zeros to negative ones
            all_preds[all_preds == 0] = -1
            all_targets[all_targets == 0] = -1
            # Calculate MCC score
            mcc_score = matthews_corrcoef(all_targets.int(), all_preds.int())
            f1_score_mean /= len(outputs)
            val_loss_mean /= len(outputs)

            tqdm_dict = {
                'val_loss': val_loss_mean,
                'val_f1_score': f1_score_mean
            }
            result = OrderedDict({
                'progress_bar': tqdm_dict,
                'log': {
                    'val_loss': val_loss_mean,
                    'val_f1_score': f1_score_mean,
                    'val_mcc_score': mcc_score
                }
            })

        else:
            val_loss_mean /= len(outputs)

            tqdm_dict = {'val_loss': val_loss_mean}
            result = OrderedDict({
                'progress_bar': tqdm_dict,
                'log': {
                    'val_loss': val_loss_mean,
                }
            })
        return result

    def test_step(self, batch, batch_nb):
        # Run through model
        loss_test, tensorboard_logs, targets_test, preds_test = self.forward(
            batch, batch_nb, 'test_')

        # Create an ordered dict
        output = OrderedDict({
            'test_loss': loss_test,
            'log': tensorboard_logs,
            'test_preds': preds_test,
            'test_targets': targets_test
        })
        return output

    def test_end(self, outputs):
        test_loss_mean = 0
        f1_score_mean = 0
        for output in outputs:
            test_loss_mean += output['test_loss']
        if self.params.disease_model:
            # Concat all validation preds and targets
            for i, output in enumerate(outputs):
                f1_score_mean += output['log']['test_f1_score']
                if i == 0:
                    all_preds = output['test_preds'].cpu().detach()
                    all_targets = output['test_targets'].cpu().detach()
                else:
                    all_preds = torch.cat(
                        (all_preds, output['test_preds'].cpu().detach()), 0)
                    all_targets = torch.cat(
                        (all_targets, output['test_targets'].cpu().detach()),
                        0)
            # Apply sigmoid to predictions
            all_preds = torch.round(torch.sigmoid(all_preds))
            # Change zeros to negative ones
            all_preds[all_preds == 0] = -1
            all_targets[all_targets == 0] = -1
            # Calculate MCC score
            mcc_score = matthews_corrcoef(all_targets.int(), all_preds.int())
            f1_score_mean /= len(outputs)
            test_loss_mean /= len(outputs)

            tqdm_dict = {
                'test_loss': test_loss_mean,
                'test_f1_score': f1_score_mean
            }
            result = OrderedDict({
                'progress_bar': tqdm_dict,
                'log': {
                    'test_loss': test_loss_mean,
                    'test_f1_score': f1_score_mean,
                    'test_mcc_score': mcc_score
                }
            })

        else:
            test_loss_mean /= len(outputs)

            tqdm_dict = {'test_loss': test_loss_mean}
            result = OrderedDict({
                'progress_bar': tqdm_dict,
                'log': {
                    'test_loss': test_loss_mean,
                }
            })

        return result

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(),
                          lr=self.params.learning_rate,
                          weight_decay=1e-2,
                          amsgrad=True)
        onecycle = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.params.learning_rate,
            epochs=self.params.num_epochs,
            steps_per_epoch=self.num_samples)  # Find suitable learning rate
        return [optimizer], [onecycle]

    def load_data_into_memory(self):
        path_to_data = self.params.path_to_data+'/'+self.params.data_set + \
            '/not_'+self.params.cage_nr+'_all_dates_train'
        if self.params.disease_model:
            print('Loading disease data')
            # Load data into memory
            train_data, train_targets, self.mean, self.std, max_value = self.data_utils.load_disease_data(
                data_path=path_to_data,
                seq_length=self.params.seq_length,
                targets_name=self.params.targets_data)

            # Split into train and validate
            self.train_data, self.val_data = self.data_utils.split_train_test(
                train_data)
            self.train_targets, self.val_targets = self.data_utils.split_train_test(
                train_targets)

            # Weighted sampling for training
            trainratio = np.bincount(self.train_targets.astype(int).squeeze())
            classcount = trainratio.tolist()
            train_weights = 1. / torch.tensor(classcount, dtype=torch.float)
            train_sampleweights = train_weights[self.train_targets.squeeze()]
            self.train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=train_sampleweights,
                num_samples=self.train_targets.shape[0])
        else:
            print('Loading synthetic data')
            # Load data into memory
            train_val_data, self.test_data, self.mean, self.std, max_value = self.data_utils.generate_seq_data(
                path=path_to_data + '_data', seq_length=self.params.seq_length)

            # Split into train and validation data
            self.train_data, self.val_data = self.data_utils.split_train_test(
                train_val_data)

        self.num_samples = max(
            len(self.train_data) // self.params.batch_size, 1)

    @pl.data_loader
    def train_dataloader(self):
        if self.params.disease_model:
            print('Creating disease train dataloader')
            train_dataset = DiseaseEchoDataset(
                dataset=self.train_data,
                targets=self.train_targets,
                seq_length=self.params.seq_length,
                normalize=True,
                mean=self.mean,
                std=self.std)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=self.params.batch_size,
                                          shuffle=False,
                                          num_workers=self.params.num_workers,
                                          pin_memory=True,
                                          sampler=self.train_sampler)
        else:
            print('Creating synthetic train dataloader')
            train_dataset = SyntheticEchoDataset(
                dataset=self.train_data,
                seq_length=self.params.seq_length,
                normalize=True,
                mean=self.mean,
                std=self.std)

            train_dataloader = DataLoader(train_dataset,
                                          batch_size=self.params.batch_size,
                                          shuffle=True,
                                          num_workers=self.params.num_workers,
                                          pin_memory=True)
        return train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        if self.params.disease_model:
            print('Creating disease val dataloader')
            val_dataset = DiseaseEchoDataset(dataset=self.val_data,
                                             targets=self.val_targets,
                                             seq_length=self.params.seq_length,
                                             normalize=True,
                                             mean=self.mean,
                                             std=self.std)

            val_dataloader = DataLoader(val_dataset,
                                        batch_size=self.params.batch_size,
                                        shuffle=True,
                                        num_workers=self.params.num_workers,
                                        pin_memory=False)
        else:
            print('Creating synthetic val dataloader')
            val_dataset = SyntheticEchoDataset(
                dataset=self.val_data,
                seq_length=self.params.seq_length,
                normalize=True,
                mean=self.mean,
                std=self.std)

            val_dataloader = DataLoader(val_dataset,
                                        batch_size=self.params.batch_size,
                                        shuffle=True,
                                        num_workers=self.params.num_workers,
                                        pin_memory=False)
        return val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        if self.params.disease_model:
            print('Creating disease test dataloader')
            test_data, test_targets = self.data_utils.load_disease_data(
                data_path=self.params.path_to_data + 'Original_Datasets/' +
                self.params.cage_nr + '_whole_cage_all_dates',
                seq_length=self.params.seq_length,
                train=False)

            test_dataset = DiseaseEchoDataset(
                dataset=test_data,
                targets=test_targets,
                seq_length=self.params.seq_length,
                normalize=True,
                mean=self.mean,
                std=self.std)

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.params.test_batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=False)
        else:
            print('Creating synthetic test dataloader')
            test_dataset = SyntheticEchoDataset(
                dataset=self.test_data,
                seq_length=self.params.seq_length,
                normalize=True,
                mean=self.mean,
                std=self.std)

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.params.test_batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=False)
        return test_dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = HyperOptArgumentParser(strategy=parent_parser.strategy,
                                              parents=[parent_parser],
                                              add_help=False)
        # Learning rate
        model_parser.add_argument('--learning_rate',
                                  type=float,
                                  default=1e-3,
                                  help='learning rate, default is 1e-3')
        model_parser.add_argument(
            '--use_cycle_lr',
            type=int,
            default=1,
            help='whether to use a cycle learning rate scheduler, default is 1'
        )
        model_parser.add_argument(
            '--load_model',
            type=int,
            default=0,
            help='Whether to load model weights or not, default is False')

        # Model specifications
        model_parser.add_argument(
            '--model_type',
            type=str,
            default='BERT',
            help=
            'which model type to train, default is BERT, other option is LSTM')
        model_parser.add_argument(
            '--input_d',
            type=int,
            default=192,
            help='The input dimensions of the model, default is 192')
        model_parser.add_argument(
            '--d_model',
            type=int,
            default=256,
            help='The internal dimension of the model, default is 256')
        model_parser.add_argument(
            '--n_layers',
            type=int,
            default=4,
            help='The number of layers for the model, default is 4')
        model_parser.add_argument(
            '--n_heads',
            type=int,
            default=16,
            help='The number of heads in the BERT model, default is 16')
        model_parser.add_argument(
            '--d_ff',
            type=int,
            default=2048,
            help=
            'The dimension of the feed forward layers of the BERT model, default is 2048'
        )
        model_parser.add_argument(
            '--dropout',
            type=float,
            default=0.1,
            help='The percentage of units for dropout, default is 0.1')
        model_parser.add_argument(
            '--classifier_dropout',
            type=float,
            default=0.1,
            help=
            'The percentage of units for dropout in the classification layer, default is 0.1'
        )

        # Data loading
        model_parser.add_argument(
            '--num_workers',
            type=int,
            default=4,
            help='The number of workers used in the dataloaders, default is 4')
        model_parser.add_argument(
            '--mask_rate',
            type=float,
            default=0.5,
            help=
            'Percentage of examples masked in synthetic task, default is 0.5')
        return model_parser
