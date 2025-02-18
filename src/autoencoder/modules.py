from torch.autograd import Function
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import threshold
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import VarAutoEncoder


class Encoder(nn.Module):

    def __init__(self, net):
        """
        Inputs:
            - net: the neural net
        """
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self,
                 linear: object,
                 net: object,
                 min_dim: int = 4,
                 ):
        """
        Inputs:
            - linear : linear layer
            - net : the rest of the net ->  conv layers
            - min_dim - the dimension of the the smallest 3d layer, needed to reshape form the 1d vector
        """
        super().__init__()
        self.linear = linear
        self.net = net
        self.min_dim = min_dim

    def forward(self, x):
        x = self.linear(x)
        # reverse function for nn.Flatter()
        x = x.reshape(x.shape[0], -1, self.min_dim,
                      self.min_dim, self.min_dim)
        x = self.net(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self,
                 nets: object,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 min_dim: int = 4,
                 only_encode=False):
        super().__init__()
        encoder_net, linear_net, decoder_net = nets
        # Creating encoder and decoder
        self.encoder = encoder_class(net=encoder_net)
        self.decoder = decoder_class(
            linear=linear_net, net=decoder_net, min_dim=min_dim)
        self.only_encode = only_encode

    def forward(self, x):
        """
        The forward function takes in an tumor batch and returns the reconstructed volume
        """
        z = self.encoder(x)
        if self.only_encode:
            return z
        x_hat = self.decoder(z)
        return x_hat

# HASH section
# Bi-half layer


gamma = 1            # parameter γ


class hash(Function):
    @staticmethod
    def forward(ctx, U):
        # if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device(f"cuda:0")  # ("cpu")  #
        # print(device)
        # import os
        # print(os.environ['CUDA_VISIBLE_DEVICES'])
        # print(torch.cuda.get_device_name())
        # Yunqiang for half and half (optimal transport)
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat(
            (torch.ones([int(N/2), D], device=device), -torch.ones([N - int(N/2), D], device=device)))  # .to(torch.cuda.get_device_name())
        # print(index.is_cuda)
        # print(B_creat.is_cuda)
        B = torch.zeros(U.shape, device=device).scatter_(
            0, index, B_creat)

        ctx.save_for_backward(U, B)

        return B

    @ staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B)/(B.numel())

        grad = g + gamma*add_g

        return grad


def hash_layer(input):
    return hash.apply(input)


class HashAutoencoder(nn.Module):

    def __init__(self,
                 nets: object,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 min_dim: int = 4,
                 only_encode=False):
        super().__init__()
        encoder_net, linear_net, decoder_net = nets
        # Creating encoder and decoder
        self.encoder = encoder_class(net=encoder_net)
        self.decoder = decoder_class(
            linear=linear_net, net=decoder_net, min_dim=min_dim)
        self.only_encode = only_encode
        self.printlayer = PrintLayer()

    def forward(self, x):
        e = self.encoder(x)
        if self.only_encode:
            return torch.sign(e)
        b = hash_layer(e)
        x_hat = self.decoder(b)
        return x_hat

# /mnt/Drive3/ivan_marcel/final_encs/encoded_HASH_FLAIR_1024_1500/syn_50k


class VarAutoencoder(nn.Module):

    def __init__(self,
                 nets: object,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 min_dim: int = 4,
                 base_channels: int = 24,
                 training: bool = False,
                 latent_dim: int = 2048,
                 only_encode=False):
        super().__init__()
        encoder_net, linear_net, decoder_net = nets
        # Creating encoder and decoder
        linear_size = 3 * 16*16*16 * base_channels
        self.encoder = encoder_class(net=encoder_net)
        self.decoder = decoder_class(
            linear=linear_net, net=decoder_net, min_dim=min_dim)
        self.only_encode = only_encode
        self.training = training
        self.latent_dim = latent_dim
        self.mu = nn.Linear(linear_size, self.latent_dim)
        self.logvar = nn.Linear(linear_size, self.latent_dim)

    def encode_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def decode_forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)

        if not self.only_encode:  # self.training:  # multiply random noise with std only during training
            std = torch.randn_like(std).mul(std)

        return std.add_(mu)

    def forward(self, x):
        """
        The forward function takes in an tumor batch and returns the reconstructed volume
        """
        mu, logvar = self.encode_forward(x)
        z = self.reparameterize(mu, logvar)
        if self.only_encode:
            return z
        x_hat = self.decode_forward(z)
        return x_hat, mu, logvar


######################
# Helper Modules
######################

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print("unique latent values:", torch.unique(x, return_counts=True))
        return x


class STEThreshold(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, input, threshold=0.5):
        output = (input > threshold).type(input.dtype)
        return output

    @ staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
