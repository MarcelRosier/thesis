import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import utils
from constants import (ENCODED_4096_BASE_PATH, ENV, REAL_TUMOR_BASE_PATH,
                       SYN_TUMOR_BASE_PATH)
from numpy.lib.arraysetops import intersect1d
from rbo import rbo

SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]
REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]
ENCODED_4096_BASE_PATH = ENCODED_4096_BASE_PATH[ENV]

data_path = "/home/ivan_marcel/thesis/src/autoencoder/data"
TEST_START = 4000
TEST_SIZE = 2000


def calc_groundtruth(real_tumor, test_set_size: str, syn_subset: Tuple = (TEST_START, TEST_START + TEST_SIZE)):
    """
    Calculate the L2 similarity between the input tumor and all (not encoded!) synthetic tumors and store the result in:\n
    autoencoder/data/encoded_l2_sim/testset_size_{test_set_size}/{real_tumor}_gt.json"
    @real_tumor - string specifying the real tumor
    @test_set_size - string specifying the test set size
    """

    # load real_tumor
    real_tumor_path = os.path.join(REAL_TUMOR_BASE_PATH, real_tumor)
    real_tumor_t1c, _ = utils.load_real_tumor(base_path=real_tumor_path)

    # unencoded real tumor is already only 0 and 1
    # added check to ensure thresholding is really not needed
    assert len(np.unique(real_tumor_t1c) <= 2)

    # load syn tumors, and extract wanted subset if specified
    folders = utils.get_sorted_syn_tumor_list()
    if syn_subset is not None:
        folders = folders[syn_subset[0]: syn_subset[1]]

    # run calc
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
    filename_dump = f"{data_path}/encoded_l2_sim/testset_size_{test_set_size}/{real_tumor}_gt.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)


def calc_encoded_similarities(real_tumor: str, test_set_size: str):
    """
    Calculate the L2 similarity between the input tumor and all encoded synthetic tumors and store the result in:\n
    autoencoder/data/encoded_l2_sim/testset_size_{test_set_size}/{real_tumor}_encoded.json"
    @real_tumor - string specifying the real tumor
    @test_set_size - string specifying the test set size
    """
    # load real tumor
    real_tumor_t1c = np.load(os.path.join(
        ENCODED_4096_BASE_PATH, f"real/{real_tumor}.npy"))[0]
    real_tumor_t1c = utils.normalize(real_tumor_t1c)

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
    filename_dump = f"{data_path}/encoded_l2_sim/testset_size_{test_set_size}/{real_tumor}_encoded.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)


def run_top_15_comp(folder_path: str = "/home/ivan_marcel/thesis/src/autoencoder/data/encoded_l2_sim/testset_size_2000") -> pd.DataFrame:
    """
    Generate a DataFrame=[tumor, gt_top, encoded_top, intersection] containing the top 15 ranks and intersection of those for each real tumor
    @folder_path - path to the folder containing the json data (gt and encoded) for each real tumor
    @return - the resulting DataFrame
    """
    # load real tumors and reduce name to id
    real_tumors = os.listdir(REAL_TUMOR_BASE_PATH)
    real_tumors.sort(key=lambda name: int(name[3:6]))

    df = pd.DataFrame(columns=["tumor", "gt_top",
                      "encoded_top", "intersection"])

    # loop over tumors and add the tumor, the top_15 ranks of the gt and encoded and the intersection the dataframe @df
    for tumor in real_tumors:
        # get gt top
        gt_best = utils.find_n_best_score_ids(
            path=f"{folder_path}/{tumor}_gt.json",
            n_best=15,
            value_type=utils.DSValueType.T1C,
            order_func=min
        )
        # get encoded top
        encoded_best = utils.find_n_best_score_ids(
            path=f"{folder_path}/{tumor}_encoded.json",
            n_best=15,
            value_type=utils.DSValueType.T1C,
            order_func=min
        )

        # intersection
        df = df.append({
            "tumor": tumor,
            "gt_top": gt_best,
            "encoded_top": encoded_best,
            "intersection": intersect1d(gt_best, encoded_best)
        }, ignore_index=True)

    return df


def calc_similarity_of_top_lists(csv_path: str, top_n: int = 15, dataset_size="200", save=False) -> Dict[str, float]:
    """
    Calculate the RBO ext similiraties of the groundtruth and encoded ranking
    @csv_path - path to the csv file containing [tumor,gt_top,encoded_top,intersection]
    @top_n - top_n ranks that should be considered (<=15)
    @dataset_size - size of used dataset, used for storing
    @save - if True the json dict will be saved to: /home/ivan_marcel/thesis/src/autoencoder/data/rbo_comp/rbo_comp_{dataset_size}_top_{top_n}.json
    @return -  dict: tumor_ids -> rbo.ext score
    """
    # load and prepare data
    df = pd.read_csv(csv_path)
    tumor_ids = df['tumor'].to_list()
    gt_lists = df['gt_top'].to_list()
    encoded_lists = df['encoded_top'].to_list()
    # split
    gt_lists = [gt_list[1:-1].split(',') for gt_list in gt_lists]
    # trim
    gt_lists = [[e.replace(' ', '')[1:-1] for e in gt_list]
                for gt_list in gt_lists]
    # split
    encoded_lists = [enc_list[1:-1].split(',') for enc_list in encoded_lists]
    # trim
    encoded_lists = [[e.replace(' ', '')[1:-1] for e in enc_list]
                     for enc_list in encoded_lists]

    # reduce the wanted top_n size
    if top_n < 15:
        gt_lists = [gt[:top_n] for gt in gt_lists]
        encoded_lists = [enc[:top_n] for enc in encoded_lists]

    # calc similarities
    sims = {}
    for i, (gt, enc) in enumerate(zip(gt_lists, encoded_lists)):
        print(tumor_ids[i])
        ext_sim = rbo(gt, enc, p=.9).ext
        print(f"{ext_sim=}")
        sims[tumor_ids[i]] = ext_sim

    if save:
        save_path = f"/home/ivan_marcel/thesis/src/autoencoder/data/rbo_comp/rbo_comp_{dataset_size}_top_{top_n}.json"
        if os.path.exists(save_path):
            print("File exists. Aborting")
            return sims
        with open(save_path, "w") as file:
            json.dump(sims, file)
    return sims


def run(real_tumor):
    pass
    # sims = calc_similarity_of_top_lists(
    #     csv_path="/home/ivan_marcel/thesis/src/autoencoder/data/gt_enc_comp_200.csv", top_n=1, dataset_size="200", save=False)
    """Example usages"""
    # df = run_top_15_comp()
    # df.to_csv("/home/ivan_marcel/thesis/src/autoencoder/data/gt_enc_comp_2k.csv",
    #           encoding='utf-8', index=False)
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
    # real_tumors = os.listdir(REAL_TUMOR_BASE_PATH)
    # real_tumors.sort(key=lambda name: int(name[3:6]))

    # func = partial(calc_groundtruth)
    # print(func)
    # with multiprocessing.Pool(8) as pool:
    #     results = pool.map_async(func, real_tumors)
    #     t = results.get()

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
