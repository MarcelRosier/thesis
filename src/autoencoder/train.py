import os
from datetime import datetime
from pathlib import Path

import matplotlib
import utils
import seaborn as sns
import torch
import torch.nn as nn
from constants import AE_CHECKPOINT_PATH, AE_MODEL_SAVE_PATH, ENV
# from monai.losses.dice import DiceLoss, GeneralizedDiceLoss
# from monai.losses.tversky import TverskyLoss
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from autoencoder import networks
from autoencoder.dataset import TumorT1CDataset
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
BASE_CHANNELS = 24
MAX_EPOCHS = 120
LATENT_DIM = 4096
MIN_DIM = 16
BATCH_SIZE = 2
TRAIN_SIZE = 1500
VAL_SIZE = 150
LEARNING_RATE = 1e-5
CHECKPOINT_FREQUENCY = 30

# print params
print(f"INFO:\n{BASE_CHANNELS=}\n{MAX_EPOCHS=}\n{LATENT_DIM=}\n{MIN_DIM=}\n{BATCH_SIZE=}\n{TRAIN_SIZE=}\n{VAL_SIZE=}\n{LEARNING_RATE=}\n{CHECKPOINT_FREQUENCY=}")
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
run_name = f"BC_{BASE_CHANNELS}_LD_{LATENT_DIM}_MD_{MIN_DIM}_BS_{BATCH_SIZE}_TS_{TRAIN_SIZE}_LR_{LEARNING_RATE}_ME_{MAX_EPOCHS}_{datetime.timestamp(datetime.now())}"
run_name = run_name.split('.')[0]
writer = SummaryWriter(log_dir=CHECKPOINT_PATH + f"/{run_name}")


nets = networks.get_basic_net_16_16_16(
    c_hid=BASE_CHANNELS,  latent_dim=LATENT_DIM)


def run(cuda_id=0):
    train_dataset = TumorT1CDataset(subset=(35000, 35000 + TRAIN_SIZE))
    val_dataset = TumorT1CDataset(subset=(2000, 2000 + VAL_SIZE))
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
    # Z = torch.zeros_like(output)
    # print(torch.unique(rounded))
    # print(f"post nonzero(rounded)= {torch.count_nonzero(rounded)}")

    writer.add_graph(model, input_to_model=tumor)
    print(result)


def train_tumort1c(cuda_id, train_loader, val_loader, test_loader):
    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device(
        f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"CUDA_VISIBLE_DEVICES = [{os.environ['CUDA_VISIBLE_DEVICES']}]")
    # gpu info
    print("Device:", device)
    print('Active CUDA Device: GPU', torch.cuda.get_device_name())

    #
    # Setup
    model = Autoencoder(nets=nets, min_dim=MIN_DIM)
    model.to(device)  # move to gpu
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CustomDiceLoss(
        smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=False)

    print("Starting training")
    for epoch in range(MAX_EPOCHS):
        loss = 0
        max_xhat = 0
        total_inter = 0
        total_den = 0
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
            with torch.no_grad():
                cur_max = torch.max(outputs).item()
                if cur_max > max_xhat:
                    max_xhat = cur_max

            # compute loss
            train_loss, intersection_tensor, den_tensor = criterion(
                outputs, batch_features)
            total_inter += intersection_tensor.mean(dim=[0]).item()
            total_den += den_tensor.mean(dim=[0]).item()

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

        # TODO adapt/ generalize when changing batch/ training size
        intersection = total_inter / (TRAIN_SIZE / BATCH_SIZE)
        denominator = total_den / (TRAIN_SIZE / BATCH_SIZE)

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
        print("epoch : {}/{}, max_xhat = {:.6f}".format(epoch + 1, MAX_EPOCHS, max_xhat))
        print(f'{intersection=}')
        print(f'{denominator=}')

        # add scalars to tensorboard
        writer.add_scalar(f"{criterion} /train", loss, epoch + 1)
        writer.add_scalar(f"{criterion} /validation", val_loss, epoch + 1)
        writer.add_scalar(f"{criterion} max_xhat", max_xhat, epoch + 1)
        writer.add_scalar(f"{criterion} intersection", intersection, epoch + 1)
        writer.add_scalar(f"{criterion} denominator", denominator, epoch + 1)

        writer.flush()
        if (epoch + 1) % CHECKPOINT_FREQUENCY == 0 and epoch + 1 < MAX_EPOCHS:
            save_checkpoint(epoch=epoch + 1, model=model,
                            loss=loss, optimizer=optimizer)

    writer.close()
    print("Finished Training")
    save_checkpoint(epoch="final", model=model,
                    loss=loss, optimizer=optimizer)
    return model, "END"


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
