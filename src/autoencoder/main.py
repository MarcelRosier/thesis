import os

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from autoencoder.dataset import TumorT1CDataset
from autoencoder import networks
from autoencoder.modules import Autoencoder, GenerateCallback
from constants import AE_CHECKPOINT_PATH_SERVER, AE_CHECKPOINT_PATH_LOCAL, IS_LOCAL

CHECKPOINT_PATH = AE_CHECKPOINT_PATH_LOCAL if IS_LOCAL else AE_CHECKPOINT_PATH_SERVER

matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
sns.set()
pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


# Hyper parameters
BASE_CHANNELS = 16
MAX_EPOCHS = 20
LATENT_DIM = 2048
nets = networks.get_k3_m8_net()


def run(cuda_id=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device(
        f"cuda:{cuda_id}") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    train_dataset = TumorT1CDataset(subset=(35000, 36000))
    val_dataset = TumorT1CDataset(subset=(2000, 2100))
    test_dataset = TumorT1CDataset(subset=(3000, 3100))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=8,
                              shuffle=False,
                              num_workers=4)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=8,
                             shuffle=False,
                             num_workers=4)
    print("Starting training")
    model, result = train_tumort1c(device=device, train_loader=train_loader,
                                   val_loader=val_loader, test_loader=test_loader)
    print("Finished Training")
    print(result)


def train_tumort1c(device, train_loader, val_loader, test_loader):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"tumor_autoencoder"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=MAX_EPOCHS,  # 500,
                         log_every_n_steps=10,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    # GenerateCallback(
                                    #     get_train_images(8), every_n_epochs=1),
                                    LearningRateMonitor("epoch")])
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    model = Autoencoder(nets=nets)
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(
        model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(
        model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result
