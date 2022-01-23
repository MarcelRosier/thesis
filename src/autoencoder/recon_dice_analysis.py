
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
    VAE = True
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

        if is_t1c:
            checkpoint_path = "/mnt/Drive3/ivan_marcel/models/final/final_VAE_T1C_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642602180/VAE_T1C_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642602180_ep_final.pt"
        else:
            checkpoint_path = "/mnt/Drive3/ivan_marcel/models/final/final_VAE_FLAIR_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642709006/ref_no_kld_VAE_FLAIR_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642709006_ep_final.pt"
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
    # bar = Bar('Processing', max=len(test_loader))
    print(f"Starting @{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    i = 0
    criterion = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=False)
    for tumor, internal_id_tensor in test_loader:
        folder_id = test_dataset.tumor_ids[internal_id_tensor.item()]
        encoded = model(tumor)[0]
        # print(encoded[0].shape)
        # print(encoded[1].shape)
        # save
        np_encoded = encoded.cpu().detach()
        # dice_score = utils.calc_dice_coef(
        #     tumor.cpu().detach().numpy(), np_encoded)
        dice_loss = criterion(tumor.cpu().detach(), np_encoded)
        data[folder_id] = 1 - dice_loss.item()
        i += 1
        print(i)
    #     bar.next()
    # bar.finish()
    print(data)
    with open(f'/home/ivan_marcel/thesis/src/autoencoder/data/recon_analysis/vae_TS_1500/{"syn" if SYNTHETIC else "real"}/monai_scores_{"t1c" if is_t1c else "flair"}.json', 'w') as file:
        json.dump(data, file)


def analyze():
    with open('/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/recon_analysis/ae_TS_1500/syn/scores_flair.json') as file:
        flair_data: dict = json.load(file)
    with open('/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/recon_analysis/ae_TS_1500/syn/scores_t1c.json') as file:
        t1c_data = json.load(file)
    avg_flair = sum(flair_data.values()) / len(flair_data.values())
    print(avg_flair)


def compare_custom_monai_ranking():
    monai_base_path = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/monai_dice/50000/dim_128/dice/tgm019_preop.json"
    custom_base_path = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice/tgm019_preop.json"

    monai_best = utils.find_n_best_score_ids(
        monai_base_path, utils.DSValueType.T1C, max, n_best=500)
    custom_best = utils.find_n_best_score_ids(
        custom_base_path, utils.DSValueType.T1C, max, n_best=500)
    # print(monai_best)
    # print(custom_best)
    # print(monai_best == custom_best)


def compare_monai_vs_custom_loss():

    def load_syn(tid):
        # load tumor data
        tumor = np.load(
            f"/mnt/Drive3/ivan/samples_extended/Dataset/{tid}/Data_0001.npz")['data']

        # crop 129^3 to 128^3 if needed
        if tumor.shape != (128, 128, 128):
            tumor = np.delete(np.delete(
                np.delete(tumor, 128, 0), 128, 1), 128, 2)
        # normalize
        max_val = tumor.max()
        if max_val != 0:
            tumor *= 1.0/max_val

        # threshold
        tumor_02 = np.copy(tumor)
        tumor_02[tumor_02 < 0.2] = 0
        tumor_02[tumor_02 >= 0.2] = 1
        tumor_06 = np.copy(tumor)
        tumor_06[tumor_06 < 0.6] = 0
        tumor_06[tumor_06 >= 0.6] = 1
        return tumor_06, tumor_02

    from utils import load_real_tumor
    from monai.losses.dice import DiceLoss
    import torch
    criterion = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=False)

    real_tumor_001 = "/mnt/Drive3/ivan_marcel/real_tumors/tgm001_preop"
    real_tumor_017 = "/mnt/Drive3/ivan_marcel/real_tumors/tgm017_preop"

    syn_t1c_39725, syn_flair_39725 = load_syn(tid=39725)
    syn_t1c_32000, syn_flair_32000 = load_syn(tid=32000)

    (t1c_001, flair_001) = load_real_tumor(real_tumor_001)
    (t1c_017, flair_017) = load_real_tumor(real_tumor_017)

    c_t_1 = utils.calc_dice_coef(syn_t1c_39725, t1c_001)
    c_f_1 = utils.calc_dice_coef(syn_flair_39725, flair_001)
    c_t_17 = utils.calc_dice_coef(syn_t1c_32000, t1c_017)
    c_f_17 = utils.calc_dice_coef(syn_flair_32000, flair_017)

    # monai
    syn_t1c_39725 = torch.from_numpy(syn_t1c_39725)
    syn_t1c_39725.unsqueeze_(0)
    syn_t1c_39725.unsqueeze_(0)
    syn_flair_39725 = torch.from_numpy(syn_flair_39725)
    syn_flair_39725.unsqueeze_(0)
    syn_flair_39725.unsqueeze_(0)
    syn_t1c_32000 = torch.from_numpy(syn_t1c_32000)
    syn_t1c_32000.unsqueeze_(0)
    syn_t1c_32000.unsqueeze_(0)
    syn_flair_32000 = torch.from_numpy(syn_flair_32000)
    syn_flair_32000.unsqueeze_(0)
    syn_flair_32000.unsqueeze_(0)
    t1c_001 = torch.from_numpy(t1c_001)
    t1c_001.unsqueeze_(0)
    t1c_001.unsqueeze_(0)
    flair_001 = torch.from_numpy(flair_001)
    flair_001.unsqueeze_(0)
    flair_001.unsqueeze_(0)
    t1c_017 = torch.from_numpy(t1c_017)
    t1c_017.unsqueeze_(0)
    t1c_017.unsqueeze_(0)
    flair_017 = torch.from_numpy(flair_017)
    flair_017.unsqueeze_(0)
    flair_017.unsqueeze_(0)

    m_t_1 = 1 - criterion(syn_t1c_39725, t1c_001).item()
    m_f_1 = 1 - criterion(syn_flair_39725, flair_001).item()
    m_t_17 = 1 - criterion(syn_t1c_32000, t1c_017).item()
    m_f_17 = 1 - criterion(syn_flair_32000, flair_017).item()

    print(f"{c_t_1=}")
    print(f"{m_t_1=}")
    print(f"{c_f_1=}")
    print(f"{m_f_1=}")
    print(f"{c_t_17=}")
    print(f"{m_t_17=}")
    print(f"{c_f_17=}")
    print(f"{m_f_17=}")

    rec_3000 = np.load(
        '/home/ivan_marcel/thesis/media/reconstructed_tumors/3000_reconstructed_2048.npy')[0][0]
    input_t1c, _ = load_syn(tid=3000)
    print(input_t1c.shape)
    print(np.unique(input_t1c))
    print(rec_3000.shape)
    print(np.unique(rec_3000))
    d_c = utils.calc_dice_coef(input_t1c, rec_3000)

    input_t1c = torch.from_numpy(input_t1c)
    input_t1c.unsqueeze_(0)
    input_t1c.unsqueeze_(0)
    rec_3000 = torch.from_numpy(rec_3000)
    rec_3000.unsqueeze_(0)
    rec_3000.unsqueeze_(0)
    d_m = 1 - criterion(input_t1c, rec_3000).item()
    print(f"{d_c=}")
    print(f"{d_m=}")
