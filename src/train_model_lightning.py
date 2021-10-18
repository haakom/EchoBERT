import pytorch_lightning as pl
from EchoTransformer_Lightning import EchoTransformer as EchoTransformer_Lightning
import torch
import argparse
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



def main(hparams):
    # Set seeds
    SEED = hparams.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Set up callback for early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min')

    # checkpoint_callback = ModelCheckpoint(
    #     filepath=hparams.checkpoint_model_dir,
    #     save_best_only=True,
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min',
    #     prefix=''
    # )

    # init model
    model = EchoTransformer_Lightning(hparams)

    # set up trainer
    if hparams.cuda == 1:
        # Single GPU training
        trainer = pl.Trainer(
            gpus=hparams.nb_gpus, use_amp=True, max_nb_epochs=hparams.num_epochs, default_save_path=hparams.save_model_dir, val_check_interval=0.5, early_stop_callback=early_stop_callback, min_nb_epochs=10)
    else:
        # CPU training
        trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, max_nb_epochs=hparams.num_epochs, default_save_path=hparams.save_model_dir,
                             val_percent_check=0.1, val_check_interval=0.5, early_stop_callback=early_stop_callback, min_nb_epochs=10)

    # fit model
    trainer.fit(model)

    # evaluate model
    trainer.test(model)


if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(
        description="parser for Echo data disease detection")
    main_arg_parser.add_argument("--save_model_dir", type=str, default='your/path/here',
                                 help="path to folder where trained model will be saved.")
    main_arg_parser.add_argument("--checkpoint_model_dir", type=str, required=False,
                                 help="path to folder where checkpoints of trained models will be saved, should not be the same as the model save dir!!!!")
    main_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    main_arg_parser.add_argument("--seed", type=int, default=42,
                                 help="random seed for training")
    main_arg_parser.add_argument("--log_interval", type=int, default=100,
                                 help="number of images after which the training loss is logged, default is 500")
    main_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")
    main_arg_parser.add_argument("--checkpoint_interval", type=int, default=1000,
                                 help="number of batches after which a checkpoint of the trained model will be created")
    main_arg_parser.add_argument("--distributed", type=int, default=0,
                                 help="if we are running the code on a distributed system i.e SLURM, default False")
    main_arg_parser.add_argument("--load_lightning_model", type=int, default=0,
                                 help="whether to load pytorch lightning model or custom model, default False (custom)")
    main_arg_parser.add_argument("--lightning_model_path", type=str, default='your/model/path/here'
                                 required=False, help="path to the pytorch lightning model to load")
    main_arg_parser.add_argument(
        "--nb_gpus", type=int, default=1, help="number of gpus to run on, default is 1")
    main_arg_parser.add_argument(
        "--nb_gpu_nodes", type=int, default=1, help="number of nodes with gpus to use, default 1")
    # Model specific args
    args = main_arg_parser.parse_args()

    main(args)
