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
from autoencoder import networks


class Encoder(nn.Module):

    def __init__(self, net):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self,
                 linear: object,
                 net: object):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.linear = linear
        self.net = net

    def forward(self, x):
        x = self.linear(x)
        # reverse function for nn.Flatter()
        x = x.reshape(x.shape[0], -1, 8, 8, 8)  # 4, 4, 4)  # TODO generalize?
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):

    def __init__(self,
                 nets: object,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 num_input_channels: int = 1,
                 width: int = 128,
                 height: int = 128,
                 depth: int = 128):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        encoder_net, linear_net, decoder_net = nets
        # Creating encoder and decoder
        self.encoder = encoder_class(net=encoder_net)
        self.decoder = decoder_class(linear=linear_net, net=decoder_net)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(
            10, num_input_channels, width, height, depth)

    def forward(self, x):
        """
        The forward function takes in an tumor batch and returns the reconstructed image
        """
        z = self.encoder(x)
        # print(z.shape)
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
