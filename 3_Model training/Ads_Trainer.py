import os
from datetime import datetime
import argparse
from argparse import ArgumentParser, Namespace
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,StochasticWeightAveraging, RichProgressBar, DeviceStatsMonitor
from Ads_Dataloader import *
from Ads_Model import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def main(hparams):
    
    seed_everything(hparams.seed)
    data_module = AdsDataModule(hparams = hparams)
    model = AdsLSTM(hparams = hparams)
    
    
    tb_logger = TensorBoardLogger(
                             save_dir="./experiments",                                
                             name = hparams.exp_name,                                
                             log_graph=False,
                             )


    ckpt_path = os.path.join(
            "experiments/{}".format(hparams.exp_name),
            'version_'+str(tb_logger.version),
            "checkpoints",
        )
        
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        monitor="val_loss"
    )


    trainer = Trainer(
        logger=tb_logger, 
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        detect_anomaly = True,
        #gpus=1,
        max_epochs=50,
        deterministic=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        accumulate_grad_batches=16
     )
    
    trainer.fit(model,data_module)
    trainer.validate(model,data_module)
    trainer.test(model,data_module)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Model Training",
        add_help=True,
    )
    
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--exp_name",
        default="exp1_{}".format(datetime.now().strftime("%d-%m-%Y--%H-%M-%S")),
        type=str,
        help="Experiment Name"
        )

    
    Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    AdsDataModule.add_data_specific_args(parser)
    AdsLSTM.add_model_specific_args(parser)

    hparams = parser.parse_args()

    main(hparams)


