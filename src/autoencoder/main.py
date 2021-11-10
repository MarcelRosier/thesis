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
from monai.losses.dice import DiceLoss, GeneralizedDiceLoss
from monai.losses.tversky import TverskyLoss

from autoencoder import networks
from autoencoder.dataset import TumorT1CDataset
from autoencoder.modules import Autoencoder, STEThreshold
from autoencoder.losses import CustomGeneralizedDiceLoss

CHECKPOINT_PATH = AE_CHECKPOINT_PATH[ENV]

matplotlib.rcParams['lines.linewidth'] = 2.0
sns.reset_orig()
sns.set()
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


# Hyper parameters
BASE_CHANNELS = 24
MAX_EPOCHS = 4
LATENT_DIM = 4096
MIN_DIM = 16
BATCH_SIZE = 2
TRAIN_SIZE = 10

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
writer = SummaryWriter(log_dir=CHECKPOINT_PATH + f"/{timestamp}")


nets = networks.get_basic_net_16_16_16(
    c_hid=BASE_CHANNELS,  latent_dim=LATENT_DIM)


def run(cuda_id=0):
    train_dataset = TumorT1CDataset(subset=(35000, 35000 + TRAIN_SIZE))
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
    model.to(torch.device("cpu"))
    # test output
    dataiter = iter(train_loader)
    tumor, _ = dataiter.next()
    print(f"pre nonzero= {torch.count_nonzero(tumor)}")
    output = model(tumor)
    print(f"post nonzero= {torch.count_nonzero(output)}")
    print(torch.unique(output))
    print(torch.min(output))
    print(torch.max(output))
    Z = torch.zeros_like(output)
    rounded = torch.where(output > 1e-10, output, Z)
    print(torch.unique(rounded))
    print(f"post nonzero(rounded)= {torch.count_nonzero(rounded)}")

    writer.add_graph(model, input_to_model=tumor)
    # save model?
    print(result)


def train_tumort1c(cuda_id, train_loader, val_loader, test_loader):
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device(
        f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"CUDA_VISIBLE_DEVICES = [{os.environ['CUDA_VISIBLE_DEVICES']}]")
    print("Device:", device)
    # Setup
    model = Autoencoder(nets=nets, min_dim=MIN_DIM)
    model.to(device)  # move to gpu
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()
    # criterion = DiceLoss(smooth_nr=0, smooth_dr=1e-5,
    #                                 squared_pred=True, to_onehot_y=False, sigmoid=False)
    criterion = CustomGeneralizedDiceLoss(
        smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=False)

    print("Starting training")
    for epoch in range(MAX_EPOCHS):
        loss = 0
        max_xhat = 0
        total_inter = 0
        total_den = 0
        for batch_features, _ in train_loader:
            # load it to the active device
            batch_features = batch_features.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions = x_hat
            outputs = model(batch_features)
            with torch.no_grad():
                cur_max = torch.max(outputs).item()
                if cur_max > max_xhat:
                    max_xhat = cur_max

                # compute training reconstruction loss (MSELoss = MeanSquaredErrorLoss)
                # compare x_hat with x
            # train_loss = criterion(outputs, batch_features)
            outputs =
            train_loss, intersection_tensor, den_tensor = criterion(
                outputs, batch_features)
            # TODO remove later
            total_inter += intersection_tensor.mean(dim=[0]).item()
            total_den += den_tensor.mean(dim=[0]).item()
            # TODO ende

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

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, MAX_EPOCHS, loss))
        print("epoch : {}/{}, max_xhat = {:.6f}".format(epoch + 1, MAX_EPOCHS, max_xhat))
        # TODO adapt/ generalize when changing batch/ training size
        intersection = total_inter / (TRAIN_SIZE / BATCH_SIZE)
        denominator = total_den / (TRAIN_SIZE / BATCH_SIZE)
        print(f'{intersection=}')
        print(f'{denominator=}')

        # add scalars to tensorboardF
        writer.add_scalar(f"{criterion} Loss/train", loss, epoch + 1)
        writer.add_scalar(f"{criterion} max_xhat", max_xhat, epoch + 1)
        writer.add_scalar(f"{criterion} intersection", intersection, epoch + 1)
        writer.add_scalar(f"{criterion} denominator", denominator, epoch + 1)

        writer.flush()
        # TODO: save checkpoints

    writer.close()
    print("Finished Training")
    return model, "END"
