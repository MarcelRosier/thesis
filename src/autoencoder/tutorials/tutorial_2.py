# Standard libraries
import json
import math
# from IPython.display import set_matplotlib_formats
import os
import urllib.request
from urllib.error import HTTPError

import matplotlib
# Imports for plotting
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from matplotlib.colors import to_rgb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.notebook import tqdm

from tutorial_2_modules import Autoencoder

# %matplotlib inline
# set_matplotlib_formats('svg', 'pdf')  # For export
matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
sns.set()

# Progress bar

# PyTorch
# Torchvision
# PyTorch Lightning
# Tensorboard extension (for visualization purposes later)
# %load_ext tensorboard

# env
IS_LOCAL = False

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH_LOCAL = "/home/marcel/Projects/uni/thesis/src/autoencoder/dataset"
CHECKPOINT_PATH_LOCAL = "/home/marcel/Projects/uni/thesis/src/autoencoder/checkpoints"
DATASET_PATH_SERVER = "/home/rosierm/thesis/src/autoencoder/dataset"
CHECKPOINT_PATH_SERVER = "/home/rosierm/thesis/src/autoencoder/checkpoints"

DATASET_PATH = DATASET_PATH_LOCAL if IS_LOCAL else DATASET_PATH_SERVER
CHECKPOINT_PATH = CHECKPOINT_PATH_LOCAL if IS_LOCAL else CHECKPOINT_PATH_SERVER

pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial9/"
# Files to download
pretrained_files = ["cifar10_64.ckpt", "cifar10_128.ckpt",
                    "cifar10_256.ckpt", "cifar10_384.ckpt"]
# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)

# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Loading the training dataset. We need to split it into a training and validation part
train_dataset = CIFAR10(root=DATASET_PATH, train=True,
                        transform=transform, download=True)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(
    train_dataset, [45000, 5000])

# Loading the test set
test_set = CIFAR10(root=DATASET_PATH, train=False,
                   transform=transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(
    train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = data.DataLoader(
    val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(
    test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)


def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack(
        [img1, img2], dim=0), nrow=2, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

# example for issues of MSELoss
# for i in range(2):
#     # Load example image
#     img, _ = train_dataset[i]
#     img_mean = img.mean(dim=[1, 2], keepdims=True)

#     # Shift image by one pixel
#     SHIFT = 1
#     img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
#     img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
#     img_shifted[:, :1, :] = img_mean
#     img_shifted[:, :, :1] = img_mean
#     compare_imgs(img, img_shifted, "Shifted -")

#     # Set half of the image to zero
#     img_masked = img.clone()
#     img_masked[:, :img_masked.shape[1]//2, :] = img_mean
#     compare_imgs(img, img_masked, "Masked -")


class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(
                imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image(
                "Reconstructions", grid, global_step=trainer.global_step)


def train_cifar(latent_dim):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"custom_cifar10_{latent_dim}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=250,  # 500,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    GenerateCallback(
                                        get_train_images(8), every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(
        model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


model_dict = {}
latent_dim_list = [64, 128, 256, 384]
for latent_dim in latent_dim_list:
    model_ld, result_ld = train_cifar(latent_dim)
    model_dict[latent_dim] = {"model": model_ld, "result": result_ld}

latent_dims = sorted([k for k in model_dict])
val_scores = [model_dict[k]["result"]["val"][0]["test_loss"]
              for k in latent_dims]

# fig = plt.figure(figsize=(6, 4))
# plt.plot(latent_dims, val_scores, '--', color="#000", marker="*",
#          markeredgecolor="#000", markerfacecolor="y", markersize=16)
# plt.xscale("log")
# plt.xticks(latent_dims, labels=latent_dims)
# plt.title("Reconstruction error over latent dimensionality", fontsize=14)
# plt.xlabel("Latent dimensionality")
# plt.ylabel("Reconstruction error")
# plt.minorticks_off()
# plt.ylim(0, 100)
# plt.show()


def visualize_reconstructions(model, input_imgs, latent_dim):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(
        imgs, nrow=4, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7, 4.5))
    plt.title(f"Reconstructed from {model.hparams.latent_dim} latents")
    plt.imshow(grid)
    plt.axis('off')
    # plt.show() if IS_LOCAL else
    plt.savefig(f"fig_{latent_dim}.png")


def get_avg_reconstruction_loss_for_model(model, train_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in train_loader:
            loss = model._get_reconstruction_loss(batch)
            total_loss += loss
        avg = total_loss / len(train_loader)
        print(avg)
        return avg


def plot_loss_vs_latent_dims(dim_list, loss_list):
    x_data = dim_list
    y_data = loss_list
    objects = np.arange(len(x_data))
    plt.bar(objects, y_data, align='center', alpha=0.5)
    plt.xticks(objects, x_data)
    plt.xlabel('Latent Dimensions')
    plt.ylabel('Average reconstruction loss')
    # plt.title('')

    plt.show()


# reconstruction_loss_list = [get_avg_reconstruction_loss_for_model(
#     model_dict[dim]["model"], train_loader=train_loader) for dim in latent_dim_list]

# plot_loss_vs_latent_dims(dim_list=latent_dim_list,
#                          loss_list=reconstruction_loss_list)

input_imgs = get_train_images(8)
# for latent_dim in model_dict:
#     visualize_reconstructions(model_dict[latent_dim]["model"], input_imgs)
for latent_dim in latent_dim_list:
    visualize_reconstructions(
        model_dict[latent_dim]["model"], input_imgs, latent_dim=latent_dim)
