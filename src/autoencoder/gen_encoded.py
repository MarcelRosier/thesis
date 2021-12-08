from datetime import datetime
import os

import matplotlib
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import utils
from constants import (AE_CHECKPOINT_PATH, AE_MODEL_SAVE_PATH, ENV,
                       TEST_SET_RANGES)
from progress.bar import Bar
# from monai.losses.dice import DiceLoss, GeneralizedDiceLoss
# from monai.losses.tversky import TverskyLoss
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from autoencoder import networks
from autoencoder.datasets import TumorT1CDataset
from autoencoder.losses import CustomDiceLoss
from autoencoder.modules import Autoencoder

CHECKPOINT_PATH = AE_CHECKPOINT_PATH[ENV]
MODEL_SAVE_PATH = AE_MODEL_SAVE_PATH[ENV]
TEST_SET_RANGES = TEST_SET_RANGES[ENV]

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
LATENT_DIM = 2048
MIN_DIM = 16
BATCH_SIZE = 1
TRAIN_SIZE = 1500
VAL_SIZE = 150
LEARNING_RATE = 1e-5
CHECKPOINT_FREQUENCY = 30
TEST_SET_SIZE = "200"
TEST_START = TEST_SET_RANGES[TEST_SET_SIZE]['START']
TEST_SIZE = TEST_SET_RANGES[TEST_SET_SIZE]['END'] - TEST_START
SYNTHETIC = True


def run(cuda_id=0):
    # print params
    utils.pretty_print_params(BASE_CHANNELS=BASE_CHANNELS, MAX_EPOCHS=MAX_EPOCHS, LATENT_DIM=LATENT_DIM, MIN_DIM=MIN_DIM, BATCH_SIZE=BATCH_SIZE,
                              TRAIN_SIZE=TRAIN_SIZE, VAL_SIZE=VAL_SIZE, LEARNING_RATE=LEARNING_RATE, CHECKPOINT_FREQUENCY=CHECKPOINT_FREQUENCY, TEST_SIZE=TEST_SIZE)
    nets = networks.get_basic_net_16_16_16(
        c_hid=BASE_CHANNELS,  latent_dim=LATENT_DIM)

    if SYNTHETIC:
        test_dataset = TumorT1CDataset(
            subset=(TEST_START, TEST_START + TEST_SIZE))
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=4)
    else:
        test_dataset = TumorT1CDataset(syntethic=False)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1)

    # print gpu info
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    device = torch.device(
        f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    utils.pretty_print_gpu_info(device)

    # Load model
    checkpoint_path = ""
    if LATENT_DIM == 4096 and TRAIN_SIZE == 1500:
        checkpoint_path = "/mnt/Drive3/ivan_marcel/models/BC_24_LD_4096_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_1636735907/BC_24_LD_4096_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_1636735907_ep_final.pt"
    if LATENT_DIM == 4096 and TRAIN_SIZE == 3000:
        checkpoint_path = "/mnt/Drive3/ivan_marcel/models/BC_24_LD_4096_MD_16_BS_2_TS_3000_LR_1e-05_ME_100_1638535363/BC_24_LD_4096_MD_16_BS_2_TS_3000_LR_1e-05_ME_100_1638535363_ep_final.pt"
    elif LATENT_DIM == 2048:
        checkpoint_path = "/mnt/Drive3/ivan_marcel/models/BC_24_LD_2048_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_1638352499/BC_24_LD_2048_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_1638352499_ep_final.pt"

    print(f"Loading: {checkpoint_path=}")
    #! only_encode=True
    model = Autoencoder(nets=nets, min_dim=MIN_DIM, only_encode=True)
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    # model.to(device)  # move to gpu
    model.eval()

    # generate encoded dataset

    bar = Bar('Processing', max=TEST_SIZE)
    print(f"Starting @{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for tumor, internal_id_tensor in test_loader:
        folder_id = test_dataset.tumor_ids[internal_id_tensor.item()]
        encoded = model(tumor)
        # save
        np_encoded = encoded.cpu().detach().numpy()
        if SYNTHETIC:
            save_path = f"/mnt/Drive3/ivan_marcel/encoded_{LATENT_DIM}_{TRAIN_SIZE}/syn_{TEST_SET_SIZE}/{folder_id}.npy"
        else:
            save_path = f"/mnt/Drive3/ivan_marcel/encoded_{LATENT_DIM}_{TRAIN_SIZE}/real/{folder_id}.npy"

        with open(save_path, 'wb') as file:
            np.save(file=file, arr=np_encoded)
        bar.next()
    bar.finish()
