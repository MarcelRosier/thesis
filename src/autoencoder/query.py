import json
import multiprocessing
import os
from functools import partial

import numpy as np
import utils
from constants import REAL_TUMOR_BASE_PATH, ENV

REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]


def run_query_for_encoded_data(real_tumor_id, is_ae=True):
    """Run two queries for t1c and flair for 50k dataset and return best combined match"""
    if is_ae:
        base_path_flair = "/mnt/Drive3/ivan_marcel/final_encs/encoded_FLAIR_1024_1500"
        base_path_t1c = "/mnt/Drive3/ivan_marcel/final_encs/encoded_T1C_1024_1500"
    else:
        base_path_flair = ""
        base_path_t1c = ""
    # load real_tumor
    real_tumor_t1c = utils.normalize(
        np.load(f"{base_path_t1c}/real/{real_tumor_id}.npy"))
    real_tumor_flair = utils.normalize(
        np.load(f"{base_path_flair}/real/{real_tumor_id}.npy"))

    folders = utils.get_sorted_syn_tumor_list()

    folders = folders[:50000]

    results = {}

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
    data_path = "/home/ivan_marcel/thesis/src/autoencoder/data"
    filename_dump = f"{data_path}/final_50k_enc_sim/{real_tumor_id}.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)


def run(processes: int):
    real_tumors = os.listdir(REAL_TUMOR_BASE_PATH)
    real_tumors.sort(key=lambda name: int(name[3:6]))
    func = partial(run_query_for_encoded_data, is_ae=True)
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map_async(func, real_tumors)
        t = results.get()
