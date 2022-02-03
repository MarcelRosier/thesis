from datetime import datetime
from autoencoder.modules import Autoencoder, VarAutoencoder
from autoencoder import networks
import json
import multiprocessing
import os
from functools import partial

import numpy as np
import utils
from constants import REAL_TUMOR_BASE_PATH, ENV
import torch
REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]


def run_query_for_encoded_data(real_tumor_id, use_stored_real_data=True, is_ae=True, is_1024=False):
    """Run two queries for t1c and flair for 50k dataset and return best combined match"""
    print(real_tumor_id)
    if is_ae:
        base_path_flair = "/mnt/Drive3/ivan_marcel/final_encs/encoded_FLAIR_1024_1500"
        base_path_t1c = "/mnt/Drive3/ivan_marcel/final_encs/encoded_T1C_1024_1500"
    else:
        base_path_flair = f"/mnt/Drive3/ivan_marcel/final_encs/encoded_VAE_FLAIR_{'1024' if is_1024 else '8'}_1500"
        base_path_t1c = f"/mnt/Drive3/ivan_marcel/final_encs/encoded_VAE_T1C_{'1024' if is_1024 else '8'}_1500"
    # load real_tumor
    if use_stored_real_data:
        real_tumor_t1c = utils.normalize(
            np.load(f"{base_path_t1c}/real/{real_tumor_id}.npy"))
        real_tumor_flair = utils.normalize(
            np.load(f"{base_path_flair}/real/{real_tumor_id}.npy"))
    else:
        if not is_ae:
            nets = networks.get_basic_net_16_16_16_without_last_linear(
                c_hid=24,  latent_dim=8)
            model_t1c = VarAutoencoder(nets=nets, min_dim=16,
                                       base_channels=24, training=False,
                                       latent_dim=8, only_encode=True)
            model_flair = VarAutoencoder(nets=nets, min_dim=16,
                                         base_channels=24, training=False,
                                         latent_dim=8, only_encode=True)
            checkpoint_path_t1c = "/mnt/Drive3/ivan_marcel/models/final/final_VAE_T1C_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642602180/VAE_T1C_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642602180_ep_final.pt"
            checkpoint_path_flair = "/mnt/Drive3/ivan_marcel/models/final/final_VAE_FLAIR_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642709006/ref_no_kld_VAE_FLAIR_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642709006_ep_final.pt"
        else:
            nets = networks.get_basic_net_16_16_16(
                c_hid=24,  latent_dim=1024)
            model_t1c = Autoencoder(nets=nets, min_dim=16, only_encode=True)
            model_flair = Autoencoder(nets=nets, min_dim=16, only_encode=True)
            checkpoint_path_t1c = "/mnt/Drive3/ivan_marcel/models/final/final_T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_600_BETA_0001_1642258438/T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_600_BETA_0001_1642258438_ep_300.pt"
            checkpoint_path_flair = "/mnt/Drive3/ivan_marcel/models/final/final_FLAIR_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_300_BETA_0001_1642493260/FLAIR_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_300_BETA_0001_1642493260_ep_final.pt"

        t1c, flair = utils.load_real_tumor(
            base_path=REAL_TUMOR_BASE_PATH + "/" + real_tumor_id)

        cp_t1c = torch.load(checkpoint_path_t1c)
        model_state_dict_t1c = cp_t1c['model_state_dict']
        model_t1c.load_state_dict(model_state_dict_t1c)
        model_t1c.eval()
        t1c = torch.from_numpy(t1c)
        t1c = t1c.float()
        t1c.unsqueeze_(0)
        t1c.unsqueeze_(0)
        real_tumor_t1c = model_t1c(t1c).detach().numpy()
        real_tumor_t1c = utils.normalize(real_tumor_t1c)

        cp_flair = torch.load(checkpoint_path_flair)
        model_state_dict_flair = cp_flair['model_state_dict']
        model_flair.load_state_dict(model_state_dict_flair)
        model_flair.eval()
        flair = torch.from_numpy(flair)
        flair = flair.float()
        flair.unsqueeze_(0)
        flair.unsqueeze_(0)
        real_tumor_flair = model_flair(flair).detach().numpy()
        real_tumor_flair = utils.normalize(real_tumor_flair)
    # print(real_tumor_t1c)
    # print(real_tumor_flair)
    # ref_real_tumor_t1c = utils.normalize(
    #     np.load(f"{base_path_t1c}/real/{real_tumor_id}.npy"))
    # ref_real_tumor_flair = utils.normalize(
    #     np.load(f"{base_path_flair}/real/{real_tumor_id}.npy"))
    # print(ref_real_tumor_t1c)
    # print(ref_real_tumor_flair)
    # print((ref_real_tumor_t1c == real_tumor_t1c).all())
    # print((ref_real_tumor_flair == real_tumor_flair).all())
    # return None
    folders = utils.get_sorted_syn_tumor_list()

    folders = folders[:50000]

    results = {}

    # func = partial(calc_score_for_pair, real_tumor_t1c,
    #                real_tumor_flair, base_path_t1c, base_path_flair)

    for folder in folders:
        syn_t1c = utils.normalize(
            np.load(f"{base_path_t1c}/syn_50k/{folder}.npy"))
        syn_flair = utils.normalize(
            np.load(f"{base_path_flair}/syn_50k/{folder}.npy"))
        flair_score = utils.calc_l2_norm(real_tumor_flair, syn_flair)
        t1c_score = utils.calc_l2_norm(real_tumor_t1c, syn_t1c)
        results[folder] = {
            'flair': str(flair_score),
            't1c': str(t1c_score),
            'combined': str(flair_score + t1c_score)
        }

    # with multiprocessing.Pool(32) as pool:
    #     results = pool.map_async(func, folders)
    #     single_scores = results.get()
    #     results = {k: v for d in single_scores for k, v in d.items()}
    # best_key = min(results.keys(), key=lambda k: results[k]['combined'])

    # best_score = {
    #     'best_score': results[best_key],
    #     'partner': best_key
    # }
    data_path = "/home/ivan_marcel/thesis/src/autoencoder/data"
    # filename_dump = f"{data_path}/final_50k_enc_sim/{'ae' if is_ae else 'vae'}/{real_tumor_id}.json"
    filename_dump = f"{data_path}/final_50k_enc_sim/{'ae' if is_ae else 'vae'}/1024/{real_tumor_id}.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)

    # return best_score


def calc_score_for_pair(rt, rf, base_path_t1c, base_path_flair, syn_id):
    syn_t1c = utils.normalize(
        np.load(f"{base_path_t1c}/syn_50k/{syn_id}.npy"))
    syn_flair = utils.normalize(
        np.load(f"{base_path_flair}/syn_50k/{syn_id}.npy"))
    flair_score = utils.calc_l2_norm(rf, syn_flair)
    t1c_score = utils.calc_l2_norm(rt, syn_t1c)
    results = {}
    results[syn_id] = {
        'flair': str(flair_score),
        't1c': str(t1c_score),
        'combined': str(flair_score + t1c_score)
    }
    return results


def run(processes: int):
    is_ae = False
    real_tumors = os.listdir(REAL_TUMOR_BASE_PATH)
    real_tumors.sort(key=lambda name: int(name[3:6]))
    func = partial(run_query_for_encoded_data,
                   use_stored_real_data=True, is_ae=is_ae, is_1024=True)
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map_async(func, real_tumors)
        t = results.get()
    # total_start = datetime.now()
    # runtimes = []
    # real_tumors = [real_tumors[15]]
    # for tumor_id in real_tumors:
    #     start = datetime.now()
    #     t = func(tumor_id)
    #     end = datetime.now()
    #     runtimes.append(end-start)
    #     print(t)
    #     print(end-start)
    # total_end = datetime.now()
    # print(f"total_duration={(total_end-total_start)}")
    # print(f"first: {runtimes[0]}")
    # print(f"avg: {sum(runtimes) / len(runtimes)}")
