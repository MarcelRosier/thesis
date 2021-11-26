from functools import partial
import json
import multiprocessing
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
TEST_START = 4000
TEST_SIZE = 2000


def calc_groundtruth(real_tumor, syn_subset=(TEST_START, TEST_START + TEST_SIZE)):
    # load real_tumor
    real_tumor_path = os.path.join(REAL_TUMOR_BASE_PATH, real_tumor)
    real_tumor_t1c, _ = utils.load_real_tumor(base_path=real_tumor_path)
    # unencoded real tumor is already only 0 and 1
    assert len(np.unique(real_tumor_t1c) <= 2)
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
    filename_dump = f"{data_path}/encoded_l2_sim/testset_size_{syn_subset[1]-syn_subset[0]}/{real_tumor}_gt.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)


def calc_encoded_similarities(real_tumor: str, syn_subset: Tuple = (TEST_START, TEST_START + TEST_SIZE)):
    # load real tumor
    real_tumor_t1c = np.load(os.path.join(
        ENCODED_4096_BASE_PATH, f"real/{real_tumor}.npy"))[0]
    real_tumor_t1c = utils.normalize(real_tumor_t1c)
    print(real_tumor_t1c)

    files = os.listdir(os.path.join(ENCODED_4096_BASE_PATH, "syn_200"))
    files.sort(key=lambda f: int(f.split('.')[0]))
    results = {}
    for file in files:
        syn_tumor = np.load(os.path.join(
            ENCODED_4096_BASE_PATH, f"syn_200/{file}"))
        syn_tumor = utils.normalize(syn_tumor)
        distance = utils.calc_l2_norm(
            syn_data=syn_tumor, real_data=real_tumor_t1c)

        key = file.split('.')[0]
        results[key] = {
            't1c': str(distance),
        }

    # save data
    # now_date = datetime.now().strftime("%Y-%m-%d")
    # now_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename_dump = f"{data_path}/encoded_l2_sim/testset_size_200/{real_tumor}_encoded.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)


def run(real_tumor):
    # calc_groundtruth(real_tumor=real_tumor, syn_subset=syn_subset)
    # find 15 best from groundtruth
    # gt_path = "/home/ivan_marcel/thesis/src/data/2021-11-24/tgm_001_vs_3000_3199_l2/2021-11-24 14:25:47_parallel_datadump_l2.json"
    # gt_best = utils.find_n_best_score_ids(
    #     path=gt_path,
    #     n_best=15,
    #     value_type=utils.DSValueType.T1C,
    #     order_func=min
    # )
    # print("groundtruth l2 top 15: ", gt_best)
    real_tumors = os.listdir(REAL_TUMOR_BASE_PATH)
    real_tumors.sort(key=lambda name: int(name[3:6]))

    func = partial(calc_groundtruth)
    print(func)
    with multiprocessing.Pool(8) as pool:
        results = pool.map_async(func, real_tumors)
        t = results.get()

    # for i, tumor in enumerate(real_tumors):
    #     print(f"{i + 1}/71")
    #     calc_groundtruth(real_tumor=tumor)

    # encoded_path = "/home/ivan_marcel/thesis/src/autoencoder/data/2021-11-24/2021-11-24 16:01:24_dump_l2_encoded.json"
    # encoded_best = utils.find_n_best_score_ids(
    #     path=encoded_path,
    #     n_best=15,
    #     value_type=utils.DSValueType.T1C,
    #     order_func=min
    # )
    # print("encoded l2 top 15: ", encoded_best)

    # # intersection
    # print("intersection: ", intersect1d(gt_best, encoded_best))
