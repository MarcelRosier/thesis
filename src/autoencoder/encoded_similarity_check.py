import json
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import utils
from constants import (ENCODED_4096_BASE_PATH, ENV, REAL_TUMOR_BASE_PATH,
                       SYN_TUMOR_BASE_PATH)
from numpy.lib.arraysetops import intersect1d

SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]
REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]
ENCODED_4096_BASE_PATH = ENCODED_4096_BASE_PATH[ENV]

data_path = "/home/ivan_marcel/thesis/src/autoencoder/data"


def calc_groundtruth(real_tumor, syn_subset=None):
    # load real_tumor
    real_tumor_path = os.path.join(REAL_TUMOR_BASE_PATH, real_tumor)
    real_tumor_t1c, _ = utils.load_real_tumor(base_path=real_tumor_path)

    # load syn tumors
    folders = utils.get_sorted_syn_tumor_list()
    if syn_subset is not None:
        folders = folders[syn_subset[0]: syn_subset[1]]

    print(f"Running with {len(folders)} folders")
    results = {}
    for folder in folders:
        syn_tumor_t1c = utils.load_single_tumor(tumor_id=folder, threshold=0.6)
        distance = utils.calc_l2_norm(
            syn_data=syn_tumor_t1c, real_data=real_tumor_t1c)
        results[folder] = {
            't1c': distance,
        }
    # save data
    now_date = datetime.now().strftime("%Y-%m-%d")
    now_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename_dump = f"{data_path}/{now_date}/{now_datetime}_dump_l2.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)

    # print best
    best_key = min(results.keys(), key=lambda k: results[k]['t1c'])
    print(best_key)


def calc_encoded_similarities(real_tumor: str, syn_subset: Tuple = None):
    # load real tumor
    real_tumor_t1c = np.load(os.path.join(
        ENCODED_4096_BASE_PATH, f"real/{real_tumor}.npy"))[0]
    real_tumor_t1c = utils.normalize(real_tumor_t1c)
    print(real_tumor_t1c)

    files = os.listdir(os.path.join(ENCODED_4096_BASE_PATH, "syn"))
    files.sort(key=lambda f: int(f.split('.')[0]))
    results = {}
    for file in files:
        syn_tumor = np.load(os.path.join(
            ENCODED_4096_BASE_PATH, f"syn/{file}"))
        syn_tumor = utils.normalize(syn_tumor)
        distance = utils.calc_l2_norm(
            syn_data=syn_tumor, real_data=real_tumor_t1c)

        key = file.split('.')[0]
        results[key] = {
            't1c': str(distance),
        }

    # save data
    now_date = datetime.now().strftime("%Y-%m-%d")
    now_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename_dump = f"{data_path}/{now_date}/{now_datetime}_dump_l2_encoded.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)

    # print best
    best_key = min(results.keys(), key=lambda k: results[k]['t1c'])
    print(best_key)


def run(real_tumor, syn_subset=(3000, 3200)):
    # calc_groundtruth(real_tumor=real_tumor, syn_subset=syn_subset)
    # find 15 best from groundtruth
    gt_path = "/home/ivan_marcel/thesis/src/data/2021-11-24/tgm_001_vs_3000_3199_l2/2021-11-24 14:25:47_parallel_datadump_l2.json"
    gt_best = utils.find_n_best_score_ids(
        path=gt_path,
        n_best=15,
        value_type=utils.DSValueType.T1C,
        order_func=min
    )
    print("groundtruth l2 top 15: ", gt_best)
    # calc_encoded_similarities(real_tumor=real_tumor)
    encoded_path = "/home/ivan_marcel/thesis/src/autoencoder/data/2021-11-24/2021-11-24 16:01:24_dump_l2_encoded.json"
    encoded_best = utils.find_n_best_score_ids(
        path=encoded_path,
        n_best=15,
        value_type=utils.DSValueType.T1C,
        order_func=min
    )
    print("encoded l2 top 15: ", encoded_best)

    # intersection
    print("intersection: ", intersect1d(gt_best, encoded_best))
