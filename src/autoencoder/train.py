import os
from datetime import datetime
from pathlib import Path

import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
import utils
from monai.losses.dice import DiceLoss
from constants import AE_CHECKPOINT_PATH, AE_MODEL_SAVE_PATH, ENV
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from autoencoder import networks
from autoencoder.datasets import TumorDataset
from autoencoder.losses import CustomDiceLoss
from autoencoder.modules import Autoencoder, VarAutoencoder

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
BASE_CHANNELS = 24
MAX_EPOCHS = 50
LATENT_DIM = 8
MIN_DIM = 16
BATCH_SIZE = 2
TRAIN_SIZE = 1500
VAL_SIZE = 150
LEARNING_RATE = 3e-5
CHECKPOINT_FREQUENCY = 50
VAE = True
BETA = 0.001  # KL beta weighting. increase for disentangled VAE
T1C = True


timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
run_name = f"{'ref_no_kld_VAE_'if VAE else ''}{'T1C'if T1C else 'FLAIR'}_BC_{BASE_CHANNELS}_LD_{LATENT_DIM}_MD_{MIN_DIM}_BS_{BATCH_SIZE}_TS_{TRAIN_SIZE}_LR_{LEARNING_RATE}_ME_{MAX_EPOCHS}_BETA_{BETA}_{datetime.timestamp(datetime.now())}"

# remove trailing time details after dot
run_name = ''.join(run_name.split('.')[:-1])
writer = SummaryWriter(log_dir=CHECKPOINT_PATH + f"/{run_name}")


nets = networks.get_basic_net_16_16_16(
    c_hid=BASE_CHANNELS,  latent_dim=LATENT_DIM)


def run(cuda_id=0):
    # print params
    utils.pretty_print_params(BASE_CHANNELS=BASE_CHANNELS, MAX_EPOCHS=MAX_EPOCHS, LATENT_DIM=LATENT_DIM, MIN_DIM=MIN_DIM, BATCH_SIZE=BATCH_SIZE,
                              TRAIN_SIZE=TRAIN_SIZE, VAL_SIZE=VAL_SIZE, LEARNING_RATE=LEARNING_RATE, CHECKPOINT_FREQUENCY=CHECKPOINT_FREQUENCY, VAE=VAE, BETA=BETA, T1C=T1C)

    # datasets
    train_dataset = TumorDataset(subset=(35000, 35000 + TRAIN_SIZE), t1c=T1C)
    val_dataset = TumorDataset(subset=(2000, 2000 + VAL_SIZE), t1c=T1C)

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
    criterion = DiceLoss(
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

        # compute validation_loss
        val_loss = 0
        with torch.no_grad():
            model.eval()  # set to eval mode
            for batch, _ in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                cur_loss = criterion(outputs, batch)
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


###
# VAE section
###


def vae_loss_function(criterion, recon_x, x, mu, log_var, beta):
    """
    For our loss we'll want to use a combination of a reconstruction loss (here, Dice) and KLD.
    By increasing the importance of the KLD loss with beta,
    we encourage the network to disentangle the latent generative factors.
    (KLD = Kullback-Leibler-Divergenz, a statistical distance:
    measures difference between two probability distributions)
    """
    reconstruction_loss = criterion(recon_x, x)
    kld = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # print(f"{reconstruction_loss=} | {kld=}")
    return reconstruction_loss + kld, kld  # + kld


def train_VAE_tumort1c(cuda_id, train_loader, val_loader):
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device(
        f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print gpu info
    utils.pretty_print_gpu_info(device=device)

    # Model setup
    nets = networks.get_basic_net_16_16_16_without_last_linear(
        c_hid=BASE_CHANNELS,  latent_dim=LATENT_DIM)
    model = VarAutoencoder(nets=nets, min_dim=MIN_DIM,
                           base_channels=BASE_CHANNELS, training=False,
                           latent_dim=LATENT_DIM)
    model.to(device)  # move to gpu
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=False)
    # criterion = torch.nn.BCELoss(reduction='sum')

    # training loop
    print("Starting training")
    for epoch in range(MAX_EPOCHS):
        loss = 0
        kld = 0
        # set to training mode
        model.train()
        for batch_features, _ in train_loader:
            # load it to the active device
            batch_features = batch_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions = x_hat and latent parameters mu, logvar
            outputs, mu, logvar = model(batch_features)

            # compute loss
            train_loss, cur_kld = vae_loss_function(
                criterion=criterion, recon_x=outputs, x=batch_features, mu=mu, log_var=logvar, beta=BETA)

            train_loss.backward()
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss
            kld += cur_kld

        # compute the epoch training loss
        loss = loss / len(train_loader)
        kld = kld / len(train_loader)

        # compute validation_loss
        val_loss = 0
        val_kld_loss = 0
        with torch.no_grad():
            model.eval()  # set to eval mode
            for batch, _ in val_loader:
                batch = batch.to(device)
                outputs, mu, logvar = model(batch_features)
                cur_loss, cur_kld = vae_loss_function(
                    criterion=criterion, recon_x=outputs, x=batch_features, mu=mu, log_var=logvar, beta=BETA)
                val_loss += cur_loss.item()
                val_kld_loss += cur_kld
        val_loss = val_loss / len(val_loader)
        val_kld_loss = cur_kld / len(val_loader)

        # prints
        print("epoch : {}/{}, train_loss = {:.6f}".format(epoch + 1, MAX_EPOCHS, loss))
        print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1, MAX_EPOCHS, val_loss))
        print("epoch : {}/{}, kld = {:.6f}".format(epoch + 1, MAX_EPOCHS, kld))
        print("epoch : {}/{}, val_kld = {:.6f}".format(epoch +
              1, MAX_EPOCHS, val_kld_loss))

        # add scalars to tensorboard
        writer.add_scalar(f"vae_loss_function /train", loss, epoch + 1)
        writer.add_scalar(
            f"vae_loss_function /validation", val_loss, epoch + 1)
        writer.add_scalar(
            f"vae_loss_function /kld", kld, epoch + 1)
        writer.add_scalar(
            f"vae_loss_function /val_kld", val_kld_loss, epoch + 1)

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
