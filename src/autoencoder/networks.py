# PyTorch
# PyTorch Lightning
import torch.nn as nn

# basic network


def get_basic_net(num_input_channels=1, c_hid=16, act_fn=nn.GELU, latent_dim=2048):
    """So far the best scores, yet no improvement after first epoch? underfit?"""
    encoder_basic = nn.Sequential(
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
    linear_basic = nn.Sequential(
        nn.Linear(latent_dim, 4*64*c_hid),
        act_fn()
    )
    decoder_basic = nn.Sequential(
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
    return encoder_basic, linear_basic, decoder_basic


def get_k3_m8_net(latent_dim=4096, num_input_channels=1, c_hid=12, act_fn=nn.GELU):
    """TODO"""
    encoder = nn.Sequential(
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
        nn.Conv3d(3 * c_hid, 3 * c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.Conv3d(3 * c_hid, 4*c_hid, kernel_size=3,
                  padding=1, stride=2),  # 16^3 => 8^3
        act_fn(),
        nn.Conv3d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.Flatten(),  # cube to single feature vector
        nn.Linear(4*8*8*8*c_hid, latent_dim)  #
    )
    linear = nn.Sequential(
        nn.Linear(latent_dim, 4*8*8*8*c_hid),
        act_fn()
    )
    decoder = nn.Sequential(
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
        nn.ConvTranspose3d(c_hid, num_input_channels, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 64^3 => 128^3
        nn.Sigmoid()  # The input images is scaled between 0 and 1, hence the output has to be bounded as well
    )
    return encoder, linear, decoder


# 2nd network
def get_2nd_net(num_input_channels, c_hid, act_fn, latent_dim):
    encoder_net_2 = nn.Sequential(
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
    linear_net_2 = nn.Sequential(
        nn.Linear(latent_dim, 4*64*c_hid),
        act_fn()
    )
    decoder_net_2 = nn.Sequential(
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

    return encoder_net_2, linear_net_2, decoder_net_2


# 3rd network
def get_3rd_net(num_input_channels, c_hid, act_fn, latent_dim):
    encoder_net_3 = nn.Sequential(
        nn.Conv3d(num_input_channels, c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.Conv3d(c_hid, c_hid, kernel_size=3,
                  padding=1, stride=2),  # 128^3 => 64^3
        act_fn(),
        nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.Conv3d(c_hid, 2*c_hid, kernel_size=3,
                  padding=1, stride=2),  # 64^3 => 32^3
        act_fn(),
        nn.Conv3d(2 * c_hid, 3 * c_hid, kernel_size=3,
                  padding=1, stride=2),  # 32^3 => 16^3
        act_fn(),
        nn.Conv3d(3 * c_hid, 3 * c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.Conv3d(3 * c_hid, 3*c_hid, kernel_size=3,
                  padding=1, stride=2),  # 16^3 => 8^3
        act_fn(),
        nn.Conv3d(3*c_hid, 3*c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.Conv3d(3 * c_hid, 3*c_hid, kernel_size=3,
                  padding=1, stride=2),  # 8^3 => 4^3
        act_fn(),
        nn.Flatten(),  # Image grid to single feature vector
        nn.Linear(3*4*4*4*c_hid, latent_dim)  # 2 * 8^3 * c_hid
    )
    linear_net_3 = nn.Sequential(
        nn.Linear(latent_dim, 4*64*c_hid),
        act_fn()
    )
    decoder_net_3 = nn.Sequential(
        nn.ConvTranspose3d(3*c_hid, 3 * c_hid, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 4^3 => 8^3
        act_fn(),
        nn.Conv3d(3*c_hid, 3*c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.ConvTranspose3d(3*c_hid, 3 * c_hid, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 8^3 => 16^3
        act_fn(),
        nn.Conv3d(3 * c_hid, 3 * c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.ConvTranspose3d(3 * c_hid, 2 * c_hid, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 16^3 => 32^3
        act_fn(),
        nn.ConvTranspose3d(2 * c_hid, c_hid, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 32^3 => 64^3
        act_fn(),
        nn.Conv3d(c_hid, c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.ConvTranspose3d(c_hid, c_hid, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 64^3 => 128^3
        act_fn(),
        nn.Conv3d(c_hid, num_input_channels, kernel_size=3, padding=1),
        act_fn(),
        nn.Sigmoid()  # The input images is scaled between 0 and 1, hence the output has to be bounded as well
    )
    return encoder_net_3, linear_net_3, decoder_net_3
# 3rd network


def get_kernel_5_net(latent_dim, c_hid=8, act_fn=nn.ReLU, num_input_channels=1):
    """Produced bad results with no improvement during training"""
    encoder = nn.Sequential(
        nn.Conv3d(num_input_channels, c_hid,
                  kernel_size=5, padding=2),  # => reamins 128, ftr extraction
        act_fn(),
        nn.Conv3d(c_hid, 2 * c_hid,
                  kernel_size=5, stride=2, padding=2),  # => 64
        act_fn(),
        nn.Conv3d(2 * c_hid, 3 * c_hid,
                  kernel_size=5, stride=2, padding=2),  # => 32
        act_fn(),
        nn.Conv3d(3 * c_hid, 3 * c_hid,
                  kernel_size=3, stride=2, padding=1),  # => 16
        act_fn(),
        nn.Conv3d(3 * c_hid, 4 * c_hid,
                  kernel_size=3, stride=2, padding=1),  # => 8
        act_fn(),
        nn.Conv3d(4 * c_hid, 5 * c_hid,
                  kernel_size=3, stride=2, padding=1),  # => 4
        act_fn(),
        nn.Flatten(),  # Image grid to single feature vector
        nn.Linear(5*4*4*4*c_hid, latent_dim)  # 5 * c_hid * 4^3 (2560)
    )
    linear = nn.Sequential(
        nn.Linear(latent_dim, 5*64*c_hid),
        act_fn()
    )
    decoder = nn.Sequential(
        nn.ConvTranspose3d(5*c_hid, 4 * c_hid, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 4^3 => 8^3
        act_fn(),
        nn.Conv3d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.ConvTranspose3d(4*c_hid, 3 * c_hid, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 8^3 => 16^3
        act_fn(),
        nn.Conv3d(3 * c_hid, 3 * c_hid, kernel_size=3, padding=1),
        act_fn(),
        nn.ConvTranspose3d(3 * c_hid, 3 * c_hid, kernel_size=3,
                           output_padding=1, padding=1, stride=2),  # 16^3 => 32^3
        act_fn(),
        nn.ConvTranspose3d(3 * c_hid, 2 * c_hid, kernel_size=5,
                           output_padding=1, padding=2, stride=2),  # 32^3 => 64^3
        act_fn(),
        nn.Conv3d(2 * c_hid, 2 * c_hid, kernel_size=5, padding=2),
        act_fn(),
        nn.ConvTranspose3d(2 * c_hid, c_hid, kernel_size=5,
                           output_padding=1, padding=2, stride=2),  # 64^3 => 128^3
        act_fn(),
        nn.Conv3d(c_hid, num_input_channels, kernel_size=5, padding=2),
        act_fn(),
        nn.Sigmoid()  # The input images is scaled between 0 and 1, hence the output has to be bounded as well
    )
    return encoder, linear, decoder
