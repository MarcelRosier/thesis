import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


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
                 min_dim: int = 4):
        super().__init__()
        encoder_net, linear_net, decoder_net = nets
        # Creating encoder and decoder
        self.encoder = encoder_class(net=encoder_net)
        self.decoder = decoder_class(
            linear=linear_net, net=decoder_net, min_dim=min_dim)

    def forward(self, x):
        """
        The forward function takes in an tumor batch and returns the reconstructed volume
        """
        z = self.encoder(x)
        # print(z.shape)
        x_hat = self.decoder(z)
        return x_hat
