# PyTorch
# PyTorch Lightning
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
# Torchvision
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import sys


class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net_1 = nn.Sequential(
            nn.Conv3d(num_input_channels, c_hid, kernel_size=3,
                      padding=1, stride=2),  # 128^3 => 64^3
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(c_hid, 2*c_hid, kernel_size=3,
                      padding=1, stride=2),  # 64^3 => 32^3
            act_fn(),
            nn.Conv3d(2 * c_hid, 3 * c_hid, kernel_size=3,
                      padding=1, stride=2),  # 32^3 => 16^3
            act_fn(),
            # keeps the size, purpose ??
            nn.Conv3d(3 * c_hid, 3 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(3 * c_hid, 4*c_hid, kernel_size=3,
                      padding=1, stride=2),  # 16^3 => 8^3
            act_fn(),
            nn.Conv3d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(4*c_hid, 4*c_hid, kernel_size=3,
                      padding=1, stride=2),  # 8^3 => 4^3
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(4*64*c_hid, latent_dim)  # 2 * 4^3 * c_hid
        )

        self.net_2 = nn.Sequential(
            nn.Conv3d(num_input_channels, c_hid, kernel_size=3,
                      padding=1, stride=2),  # 128^3 => 64^3
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(c_hid, 2*c_hid, kernel_size=3,
                      padding=1, stride=2),  # 64^3 => 32^3
            act_fn(),
            nn.Conv3d(2 * c_hid, 2 * c_hid, kernel_size=3,
                      padding=1, stride=2),  # 32^3 => 16^3
            act_fn(),
            # keeps the size, purpose ??
            nn.Conv3d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(2 * c_hid, 2*c_hid, kernel_size=3,
                      padding=1, stride=2),  # 16^3 => 8^3
            act_fn(),
            nn.Conv3d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(2*c_hid, 2*c_hid, kernel_size=3,
                      padding=1, stride=2),  # 8^3 => 4^3
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2*64*c_hid, latent_dim)  # 2 * 4^3 * c_hid
        )

    def forward(self, x):
        return self.net_2(x)


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*64*c_hid),
            act_fn()
        )
        self.net_1 = nn.Sequential(
            nn.ConvTranspose3d(4*c_hid, 4*c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 4^3 => 8^3
            act_fn(),
            nn.Conv3d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose3d(4*c_hid, 3 * c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 8^3 => 16^3
            act_fn(),
            nn.Conv3d(3 * c_hid, 3 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose3d(3 * c_hid, 2 * c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 16^3 => 32^3
            act_fn(),
            nn.Conv3d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose3d(2*c_hid, c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 32^3 => 64^3
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose3d(c_hid, num_input_channels, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 64^3 => 128^3
            nn.Sigmoid()  # The input images is scaled between 0 and 1, hence the output has to be bounded as well
        )
        self.net_2 = nn.Sequential(
            nn.ConvTranspose3d(2*c_hid, 2*c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 4^3 => 8^3
            act_fn(),
            nn.Conv3d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose3d(2*c_hid, 2 * c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 8^3 => 16^3
            act_fn(),
            nn.Conv3d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose3d(2 * c_hid, c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 16^3 => 32^3
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose3d(c_hid, c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 32^3 => 64^3
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose3d(c_hid, num_input_channels, kernel_size=3,
                               output_padding=1, padding=1, stride=2),  # 64^3 => 128^3
            nn.Sigmoid()  # The input images is scaled between 0 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        # reverse function for nn.Flatter()
        x = x.reshape(x.shape[0], -1, 4, 4, 4)
        x = self.net_2(x)
        return x


class Autoencoder(pl.LightningModule):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 1,
                 width: int = 128,
                 height: int = 128,
                 depth: int = 128):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(
            num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(
            num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(
            10, num_input_channels, width, height, depth)

    def forward(self, x):
        """
        The forward function takes in an tumor batch and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of tumors, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3, 4])  # TODO is this correct?
        loss = loss.mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=20,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss)


class GenerateCallback(pl.Callback):

    def __init__(self, input_tumors, every_n_epochs=1):
        super().__init__()
        self.input_tumors = input_tumors  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_tumors = self.input_tumors.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_tumors = pl_module(input_tumors)
                pl_module.train()
            # Plot and add to tensorboard
            # writer.add_scalar('training loss', 1, trainer.current_epoch)
            #  running_loss /100, epoch * n_total_steps + i)
