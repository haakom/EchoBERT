import pytorch_lightning as pl
from EchoTransformer_Lightning import EchoTransformer as EchoTransformer_Lightning
import torch
import argparse
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube.hpc import SlurmCluster, HyperOptArgumentParser
from test_tube import Experiment
import glob
import os
from argparse import Namespace

# hyperparameters is a test-tube hyper params object
# see https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/

from pytorch_lightning.logging import TensorBoardLogger


def train_fx(hparams, cluster):
    print(hparams)
    # Build path to save model
    if hparams.disease_model:
        save_model_path = hparams.save_model_dir + '/disease'
    else:
        save_model_path = hparams.save_model_dir + '/synthetic'
    # Set seeds
    SEED = hparams.seed
    # torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Set up callback for early stopping
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=6,
                                        mode='min')

    # Set up a logger to store model information and tensorboard
    logger = TensorBoardLogger(save_dir=save_model_path,
                               version=str(cluster.hpc_exp_number),
                               name=hparams.cage_nr)
    # Set up a place to store models
    checkpoint_callback = ModelCheckpoint(
        filepath=save_model_path + '/' + hparams.cage_nr + '/version_' +
        str(cluster.hpc_exp_number) + '/checkpoints',
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix='')

    # init model
    model = EchoTransformer_Lightning(params=hparams)

    # set up trainer
    if hparams.cuda == 1:
        # Single GPU training
        trainer = pl.Trainer(gpus=1,
                             logger=logger,
                             checkpoint_callback=checkpoint_callback,
                             max_nb_epochs=hparams.num_epochs,
                             val_check_interval=0.5,
                             early_stop_callback=early_stop_callback,
                             min_nb_epochs=hparams.min_nb_epochs)
    else:
        # CPU training
        trainer = pl.Trainer(experiment=exp,
                             max_nb_epochs=hparams.num_epochs,
                             default_save_path=hparams.save_model_dir,
                             log_save_interval=10,
                             val_percent_check=0.1,
                             val_check_interval=0.5,
                             early_stop_callback=early_stop_callback,
                             min_nb_epochs=10)

    # fit model
    trainer.fit(model)

    # evaluate model
    trainer.test(model)


def optimize_on_cluster(hyperparams):

    # init cluster
    cluster = SlurmCluster(hyperparam_optimizer=hyperparams,
                           log_path=hyperparams.slurm_log_path)

    # let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
    cluster.notify_job_status(email='your@email.here',
                              on_done=True,
                              on_fail=False)

    # set the job options. In this instance, we'll run 20 different models
    # each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1

    # set slurm partition and account
    cluster.add_slurm_cmd(cmd='partition',
                          value='your/partition',
                          comment='what partition to use')
    # cluster.add_slurm_cmd(cmd='partition', value='EPICALL',
    #                       comment='what partition to use')
    cluster.add_slurm_cmd(cmd='account',
                          value='your/account',
                          comment='what account to use')

    # we'll request 10GB of memory per node
    cluster.memory_mb_per_node = 200000

    # set a walltime
    cluster.job_time = '06:00:00'

    # 1 minute before walltime is up, SlurmCluster will launch a continuation job and kill this job.
    # you must provide your own loading and saving function which the cluster object will call
    cluster.minutes_to_checkpoint_before_walltime = 1

    # Set up environment
    cluster.add_command('module purge')
    cluster.load_modules([
        'icc/2018.1.163-GCC-6.4.0-2.28', 'OpenMPI/2.1.2', 'goolfc/2017b',
        'CUDA/10.1.105'
    ])
    cluster.add_command(
        'MPIRUNFILE=/share/apps/software/Compiler/intel/2018.1.163-GCC-6.4.0-2.28/OpenMPI/2.1.2/bin/mpirun'
    )
    cluster.add_command('PYTHON=$HOME/your/conda/environment')

    if not hyperparams.disease_model:
        job_name = 'synthetic'
        job_disp_name = 'syn'
    else:
        job_name = 'disease'
        job_disp_name = 'dis'
    cluster.optimize_parallel_cluster_gpu(train_fx,
                                          nb_trials=30,
                                          job_name=job_name,
                                          job_display_name=job_disp_name)


if __name__ == '__main__':
    # Cages set as hyperparametrs
    hyperparser = HyperOptArgumentParser(strategy='grid_search',
                                         add_help=False)

    # Logging
    hyperparser.add_argument('--slurm_log_path',
                             type=str,
                             default='your/log/path/here',
                             help='where slurm will save scripts to')
    hyperparser.add_argument('--save_model_dir',
                             type=str,
                             default='/your/save/model/path/here')
    # Data
    hyperparser.add_argument(
        '--path_to_data',
        type=str,
        required=True,
        help='path to training dataset, the path should point to a folder ')
    hyperparser.add_argument('--data_set',
                             type=str,
                             default='X_val',
                             help='The type of data to load, default is X_val')
    hyperparser.add_argument('--batch_size',
                             type=int,
                             default=45,
                             help='batch size for training, default is 45')
    hyperparser.add_argument('--test_batch_size',
                             type=int,
                             default=256,
                             help='batch size for testing, default is 256')
    hyperparser.add_argument(
        '--seq_length',
        type=int,
        default=256,
        help=
        'The length of the sequences used for the model. The actual sequence length is 2*seq_length. Default is 256'
    )
    hyperparser.add_argument(
        '--targets_data',
        type=str,
        default='_targets',
        help=
        'Which targets to use for disease classification, default is _targets (15th of april)'
    )
    # Experiment args
    hyperparser.add_argument(
        '--checkpoint_model_dir',
        type=str,
        required=False,
        help=
        'path to folder where checkpoints of trained models will be saved, should not be the same as the model save dir!!!!'
    )
    hyperparser.add_argument('--cuda',
                             type=int,
                             default=1,
                             help='set it to 1 for running on GPU, 0 for CPU')
    hyperparser.add_argument('--seed',
                             type=int,
                             default=42,
                             help='random seed for training')
    hyperparser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help=
        'number of steps after which the training loss is logged, default is 100'
    )
    hyperparser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=100,
        help=
        'number of batches after which a checkpoint of the trained model will be created'
    )
    hyperparser.add_argument(
        '--distributed',
        type=int,
        default=0,
        help=
        'if we are running the code on a distributed system i.e SLURM, default False'
    )
    hyperparser.add_argument(
        '--lightning_model_path',
        type=str,
        default='/your/model/path/here/',
        help='path to the pytorch lightning model to load')
    hyperparser.add_argument(
        '--disease_model',
        type=int,
        default=0,
        help='are we training a disease model? default is False (0)')
    hyperparser.add_argument('--num_epochs',
                             type=int,
                             default=30,
                             help='number of training epochs, default is 30')
    hyperparser.add_argument(
        '--min_nb_epochs',
        type=int,
        default=10,
        help=
        'the minimum number of epochs to run before early stopping is allowed to stop the run, default is 10'
    )

    # Get model specific args
    hyperparser = EchoTransformer_Lightning.add_model_specific_args(
        hyperparser)

    # Experiment
    hyperparser.opt_list(
        '--cage_nr',
        type=str,
        default='15.1',
        options=['15.1', '15.2', '15.3', '15.4', '15.5', '15.6'],
        tunable=True)
    hyperparser.opt_list('--run_num',
                         type=int,
                         default=0,
                         options=[0, 1, 2, 3, 4],
                         tunable=True)
    # Parse
    hyperparams = hyperparser.parse_args()
    optimize_on_cluster(hyperparams)
