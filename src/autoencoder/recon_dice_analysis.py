
import json
import os
from datetime import datetime
from progress.bar import Bar

import numpy as np
import torch
import utils
from monai.losses.dice import DiceLoss
from torch.utils.data import DataLoader

from autoencoder import networks
from autoencoder.datasets import TumorDataset
from autoencoder.modules import Autoencoder, VarAutoencoder


def compute_recon_dice_scores(is_t1c, cuda_id):
    SYNTHETIC = False
    VAE = False
    if SYNTHETIC:
        test_dataset = TumorDataset(
            subset=(0, 50000), t1c=is_t1c)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=4)
    else:
        test_dataset = TumorDataset(syntethic=False, t1c=is_t1c)
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
    if VAE:
        nets = networks.get_basic_net_16_16_16_without_last_linear(
            c_hid=24,  latent_dim=8)
        model = VarAutoencoder(nets=nets, min_dim=16,
                               base_channels=24, training=False,
                               latent_dim=8, only_encode=False)
    else:
        nets = networks.get_basic_net_16_16_16(
            c_hid=24,  latent_dim=1024)
        model = Autoencoder(nets=nets, min_dim=16, only_encode=False)
        if is_t1c:
            checkpoint_path = "/mnt/Drive3/ivan_marcel/models/final/final_T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_600_BETA_0001_1642258438/T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_600_BETA_0001_1642258438_ep_300.pt"
        else:
            checkpoint_path = "/mnt/Drive3/ivan_marcel/models/final/final_FLAIR_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_300_BETA_0001_1642493260/FLAIR_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_300_BETA_0001_1642493260_ep_final.pt"

    print(f"Loading: {checkpoint_path=}")

    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    # model.to(device)  # move to gpu
    model.eval()

    # generate encoded dataset
    data = {}
    bar = Bar('Processing', max=len(test_loader))
    print(f"Starting @{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    i = 0
    criterion = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=False)
    for tumor, internal_id_tensor in test_loader:
        folder_id = test_dataset.tumor_ids[internal_id_tensor.item()]
        encoded = model(tumor)
        # save
        np_encoded = encoded.cpu().detach()
        # dice_score = utils.calc_dice_coef(
        #     tumor.cpu().detach().numpy(), np_encoded)
        dice_loss = criterion(tumor.cpu().detach(), np_encoded)
        data[folder_id] = 1 - dice_loss.item()

        bar.next()
    bar.finish()
    print(data)
    with open(f'/home/ivan_marcel/thesis/src/autoencoder/data/recon_analysis/ae_TS_1500/{"syn" if SYNTHETIC else "real"}/monai_scores_{"t1c" if is_t1c else "flair"}.json', 'w') as file:
        json.dump(data, file)


def analyze():
    with open('/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/recon_analysis/ae_TS_1500/syn/scores_flair.json') as file:
        flair_data: dict = json.load(file)
    with open('/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/recon_analysis/ae_TS_1500/syn/scores_t1c.json') as file:
        t1c_data = json.load(file)
    avg_flair = sum(flair_data.values()) / len(flair_data.values())
    print(avg_flair)
