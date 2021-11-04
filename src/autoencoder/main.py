from datetime import datetime
import os

import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
from constants import AE_CHECKPOINT_PATH, ENV
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from autoencoder import networks
from autoencoder.dataset import TumorT1CDataset
from autoencoder.modules import Autoencoder

CHECKPOINT_PATH = AE_CHECKPOINT_PATH[ENV]

matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
sns.set()
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


# Hyper parameters
BASE_CHANNELS = 16
MAX_EPOCHS = 4
LATENT_DIM = 2048
MIN_DIM = 4
BATCH_SIZE = 8

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
writer = SummaryWriter(log_dir=CHECKPOINT_PATH + f"/{timestamp}")


nets = networks.get_basic_net(c_hid=BASE_CHANNELS, latent_dim=LATENT_DIM)


def run(cuda_id=0):
    train_dataset = TumorT1CDataset(subset=(35000, 36000))
    val_dataset = TumorT1CDataset(subset=(2000, 2100))
    test_dataset = TumorT1CDataset(subset=(3000, 3100))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4)

    model, result = train_tumort1c(cuda_id=cuda_id, train_loader=train_loader,
                                   val_loader=val_loader, test_loader=test_loader)
    # save model?
    print(result)


def train_tumort1c(cuda_id, train_loader, val_loader, test_loader):
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device(
        f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    print("Device:", device)
    # Setup
    model = Autoencoder(nets=nets, min_dim=MIN_DIM)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    print("Starting training")
    for epoch in range(MAX_EPOCHS):
        loss = 0
        for batch_features, _ in train_loader:
            # load it to the active device
            batch_features = batch_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions = x_hat
            outputs = model(batch_features)

            # compute training reconstruction loss (MSELoss = MeanSquaredErrorLoss)
            # compare x_hat with x
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            # perform backpropagation of errors
            train_loss.backward()

            # perform parameter update based on current gradients
            # optimize weights
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        writer.add_scalar(f"{criterion} Loss/train", loss, epoch + 1)
        writer.flush()
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, MAX_EPOCHS, loss))
        # TODO: save checkpoints

    writer.close()
    print("Finished Training")
    return model, "result?"
