import os
from datetime import datetime
from pathlib import Path

import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
import utils
from constants import AE_CHECKPOINT_PATH, AE_MODEL_SAVE_PATH, ENV
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from autoencoder import networks
from autoencoder.datasets import TumorT1CDataset
from autoencoder.losses import CustomDiceLoss
from autoencoder.modules import Autoencoder

CHECKPOINT_PATH = AE_CHECKPOINT_PATH[ENV]
MODEL_SAVE_PATH = AE_MODEL_SAVE_PATH[ENV]

matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
sns.set()
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


# Hyper parameters
BASE_CHANNELS = 32
MAX_EPOCHS = 120
LATENT_DIM = 1024
MIN_DIM = 16
BATCH_SIZE = 2
TRAIN_SIZE = 4500
VAL_SIZE = 200
LEARNING_RATE = 1e-5
CHECKPOINT_FREQUENCY = 60
VAE = False
BETA = 100  # KL beta weighting. increase for disentangled VAE


timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
run_name = f"BC_{BASE_CHANNELS}_LD_{LATENT_DIM}_MD_{MIN_DIM}_BS_{BATCH_SIZE}_TS_{TRAIN_SIZE}_LR_{LEARNING_RATE}_ME_{MAX_EPOCHS}_VAE_{VAE}_BETA_{BETA}_{datetime.timestamp(datetime.now())}"
run_name = run_name.split('.')[0]
writer = SummaryWriter(log_dir=CHECKPOINT_PATH + f"/{run_name}")


nets = networks.get_basic_net_16_16_16(
    c_hid=BASE_CHANNELS,  latent_dim=LATENT_DIM)


def run(cuda_id=0):
    # print params
    utils.pretty_print_params(BASE_CHANNELS=BASE_CHANNELS, MAX_EPOCHS=MAX_EPOCHS, LATENT_DIM=LATENT_DIM, MIN_DIM=MIN_DIM, BATCH_SIZE=BATCH_SIZE,
                              TRAIN_SIZE=TRAIN_SIZE, VAL_SIZE=VAL_SIZE, LEARNING_RATE=LEARNING_RATE, CHECKPOINT_FREQUENCY=CHECKPOINT_FREQUENCY, VAE=VAE, BETA=BETA)

    # datasets
    train_dataset = TumorT1CDataset(subset=(35000, 35000 + TRAIN_SIZE))
    val_dataset = TumorT1CDataset(subset=(2000, 2000 + VAL_SIZE))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4)

    # train
    if VAE:
        model = train_VAE_tumort1c(
            cuda_id=cuda_id, train_loader=train_loader, val_loader=val_loader)
    else:
        model = train_tumort1c(
            cuda_id=cuda_id, train_loader=train_loader, val_loader=val_loader)

    # add graph to tensorboard
    model.to(torch.device("cpu"))
    dataiter = iter(train_loader)
    tumor, _ = dataiter.next()
    writer.add_graph(model, input_to_model=tumor)


def train_tumort1c(cuda_id, train_loader, val_loader):
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device(
        f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print gpu info
    utils.pretty_print_gpu_info(device=device)

    # Model setup
    model = Autoencoder(nets=nets, min_dim=MIN_DIM)
    model.to(device)  # move to gpu
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CustomDiceLoss(
        smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=False)

    # training loop
    print("Starting training")
    for epoch in range(MAX_EPOCHS):
        loss = 0
        # set to training mode
        model.train()
        for batch_features, _ in train_loader:
            # load it to the active device
            batch_features = batch_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions = x_hat
            outputs = model(batch_features)

            # compute loss
            train_loss, intersection_tensor, _ = criterion(
                outputs, batch_features)

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

        # compute validation_loss
        val_loss = 0
        with torch.no_grad():
            model.eval()  # set to eval mode
            for batch, _ in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                cur_loss, _, _ = criterion(outputs, batch)
                val_loss += cur_loss.item()
        val_loss = val_loss / len(val_loader)

        # prints
        print("epoch : {}/{}, train_loss = {:.6f}".format(epoch + 1, MAX_EPOCHS, loss))
        print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1, MAX_EPOCHS, val_loss))

        # add scalars to tensorboard
        writer.add_scalar(f"{criterion} /train", loss, epoch + 1)
        writer.add_scalar(f"{criterion} /validation", val_loss, epoch + 1)

        writer.flush()
        # save checkpoints with frequency CHECKPOINT_FREQUENCY
        if (epoch + 1) % CHECKPOINT_FREQUENCY == 0 and epoch + 1 < MAX_EPOCHS:
            save_checkpoint(epoch=epoch + 1, model=model,
                            loss=loss, optimizer=optimizer)

    writer.close()
    print("Finished Training")
    save_checkpoint(epoch="final", model=model,
                    loss=loss, optimizer=optimizer)
    return model


def train_VAE_tumort1c(cuda_id, train_loader, val_loader):
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device(
        f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print gpu info
    utils.pretty_print_gpu_info(device=device)

    # Model setup
    # TODO model = (nets=nets, min_dim=MIN_DIM)
    model.to(device)  # move to gpu
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # TODO criterion =

    # training loop
    print("Starting training")
    for epoch in range(MAX_EPOCHS):
        loss = 0
        # set to training mode
        model.train()
        for batch_features, _ in train_loader:
            # load it to the active device
            batch_features = batch_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions = x_hat
            outputs = model(batch_features)

            # compute loss
            # TODO
            train_loss, intersection_tensor, _ = criterion(
                outputs, batch_features)

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

        # compute validation_loss
        val_loss = 0
        with torch.no_grad():
            model.eval()  # set to eval mode
            for batch, _ in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                # TODO
                cur_loss, _, _ = criterion(outputs, batch)
                val_loss += cur_loss.item()
        val_loss = val_loss / len(val_loader)

        # prints
        print("epoch : {}/{}, train_loss = {:.6f}".format(epoch + 1, MAX_EPOCHS, loss))
        print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1, MAX_EPOCHS, val_loss))

        # add scalars to tensorboard
        writer.add_scalar(f"{criterion} /train", loss, epoch + 1)
        writer.add_scalar(f"{criterion} /validation", val_loss, epoch + 1)

        writer.flush()
        # save checkpoints with frequency CHECKPOINT_FREQUENCY
        if (epoch + 1) % CHECKPOINT_FREQUENCY == 0 and epoch + 1 < MAX_EPOCHS:
            save_checkpoint(epoch=epoch + 1, model=model,
                            loss=loss, optimizer=optimizer)

    writer.close()
    print("Finished Training")
    save_checkpoint(epoch="final", model=model,
                    loss=loss, optimizer=optimizer)
    return model


def save_checkpoint(epoch, model, loss, optimizer):
    # save model, ensure models folder exists
    save_folder = os.path.join(MODEL_SAVE_PATH, run_name)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': MAX_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(save_folder, run_name + f"_ep_{epoch}.pt"))
