from datetime import datetime
import json
import os
from typing import Dict, List, Tuple

import multiprocessing
from functools import partial
import numpy as np
from numpy.lib.type_check import real
import pandas as pd
import utils
from utils import SimilarityMeasureType
from constants import (ENCODED_BASE_PATH, ENV, REAL_TUMOR_BASE_PATH,
                       SYN_TUMOR_BASE_PATH, TEST_SET_RANGES)
from numpy.lib.arraysetops import intersect1d
from rbo import rbo
from progress.bar import Bar

SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]
REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]
ENCODED_BASE_PATH = ENCODED_BASE_PATH[ENV]
TEST_SET_RANGES = TEST_SET_RANGES[ENV]

data_path = "/home/ivan_marcel/thesis/src/autoencoder/data"
TEST_START = 4000
TEST_SIZE = 2000


def calc_groundtruth(real_tumor: str, test_set_size: str, metric=SimilarityMeasureType.L2):
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

    folders = folders[TEST_SET_RANGES[test_set_size]
                      ['START']: TEST_SET_RANGES[test_set_size]['END']]

    # run calc
    print(
        f"Running with {len(folders)} folders, RANGE={TEST_SET_RANGES[test_set_size]['START']}-{TEST_SET_RANGES[test_set_size]['END']}")
    results = {}
    for folder in folders:
        syn_tumor_t1c = utils.load_single_tumor(tumor_id=folder, threshold=0.6)
        norm = utils.calc_l2_norm if metric == SimilarityMeasureType.L2 else utils.calc_dice_coef
        distance = norm(
            syn_data=syn_tumor_t1c, real_data=real_tumor_t1c)
        results[folder] = {
            't1c': distance,
        }

    # save data
    filename_dump = None
    if metric == SimilarityMeasureType.L2:
        filename_dump = f"{data_path}/encoded_l2_sim/testset_size_{test_set_size}/{real_tumor}_gt.json"
    else:
        filename_dump = f"{data_path}/groundtruth_dice_sim/testset_size_{test_set_size}/{real_tumor}_gt.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)


def calc_encoded_similarities(real_tumor: str, test_set_size: str, latent_dim: int, train_size: int):
    """
    Calculate the L2 similarity between the input tumor and all encoded synthetic tumors and store the result in:\n
    autoencoder/data/encoded_l2_sim/testset_size_{test_set_size}/{real_tumor}_encoded.json"
    @real_tumor - string specifying the real tumor
    @test_set_size - string specifying the test set size
    """
    # load real tumor
    encoded_folder_path = f"{ENCODED_BASE_PATH}_{latent_dim}_{train_size}"
    real_tumor_t1c = np.load(os.path.join(
        encoded_folder_path, f"real/{real_tumor}.npy"))[0]
    real_tumor_t1c = utils.normalize(real_tumor_t1c)

    files = os.listdir(os.path.join(
        encoded_folder_path, f"syn_{test_set_size}"))
    files.sort(key=lambda f: int(f.split('.')[0]))
    results = {}
    for file in files:
        syn_tumor = np.load(os.path.join(
            encoded_folder_path, f"syn_{test_set_size}/{file}"))
        syn_tumor = utils.normalize(syn_tumor)
        distance = utils.calc_l2_norm(
            syn_data=syn_tumor, real_data=real_tumor_t1c)

        key = file.split('.')[0]
        results[key] = {
            't1c': str(distance),
        }

    # save data
    filename_dump = f"{data_path}/encoded_l2_sim/testset_size_{test_set_size}/enc_{latent_dim}_{train_size}/{real_tumor}_encoded.json"
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(results, file)


def run_top_15_comp(enc: str, test_set_size: str, gt_metric: SimilarityMeasureType = SimilarityMeasureType.L2, save: bool = False, folder_path: str = "/home/ivan_marcel/thesis/src/autoencoder/data/encoded_l2_sim/testset_size_2k", ) -> pd.DataFrame:
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

    is_l2 = gt_metric == SimilarityMeasureType.L2
    # loop over tumors and add the tumor, the top_15 ranks of the gt and encoded and the intersection the dataframe @df
    for tumor in real_tumors:
        # get gt top
        gt_best_path = f"{data_path}/{'encoded_l2_sim' if is_l2 else 'groundtruth_dice_sim'}/{test_set_size}{'/gt' if is_l2 else ''}/{tumor}_gt.json"
        gt_best_order_func = min if is_l2 else max
        gt_best = utils.find_n_best_score_ids(
            path=gt_best_path,
            n_best=15,
            value_type=utils.DSValueType.T1C,
            # for l2 the minimal distance is the best, for dice the max score
            order_func=gt_best_order_func
        )
        # get encoded top
        encoded_best = utils.find_n_best_score_ids(
            path=f"{data_path}/encoded_l2_sim/{test_set_size}/{enc}/{tumor}_encoded.json",
            n_best=15,
            value_type=utils.DSValueType.T1C,
            # encoded ranking always uses l2 -> min
            order_func=min
        )

        # intersection
        df = df.append({
            "tumor": tumor,
            "gt_top": gt_best,
            "encoded_top": encoded_best,
            "intersection": intersect1d(gt_best, encoded_best)
        }, ignore_index=True)

    if save:
        df.to_csv(f"/home/ivan_marcel/thesis/src/autoencoder/data/{'l2' if  is_l2 else 'dice'}_gt_{enc}_comp_{test_set_size}.csv",
                  encoding='utf-8', index=False)
    return df


def load_top_15_lists(csv_path) -> Tuple[List, List]:
    """
    Load the 2 top 15 lists for groundtruth and 4096 encoded
    @return (tumor_ids, gt_top, enc4096_top)
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
    return (tumor_ids, gt_lists, encoded_lists)


def calc_similarity_of_top_lists(csv_path: str, top_n: int = 15, dataset_size: str = "200", save: bool = False) -> Dict[str, float]:
    """
    Calculate the RBO ext similiraties of the groundtruth and encoded ranking
    @csv_path - path to the csv file containing [tumor,gt_top,encoded_top,intersection]
    @top_n - top_n ranks that should be considered (<=15)
    @dataset_size - size of used dataset, used for storing
    @save - if True the json dict will be saved to: /home/ivan_marcel/thesis/src/autoencoder/data/rbo_comp/rbo_comp_{dataset_size}_top_{top_n}.json
    @return -  dict: tumor_ids -> rbo.ext score
    """

    tumor_ids, gt_lists, encoded_lists = load_top_15_lists(csv_path=csv_path)
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


def calc_best_match_pairs(test_set_size: str, enc: str,  save=False) -> Dict:
    """
    Get the best match for all encoded tumors and find the same tumor in the ranked list of the unencoded comparison
    @test_set_size - testset size string [200, 2k, 20k] 
    @return {
        'tumor': {'encoded_best_match': syn_tumor_id, 'unencoded_rank': rank}
    }
    """
    tumor_ids, gt_lists, encoded_lists = load_top_15_lists(
        csv_path=f"/home/ivan_marcel/thesis/media/{enc}/gt_{enc}_comp_{test_set_size}.csv")
    folder_path = f"/home/ivan_marcel/thesis/src/autoencoder/data/encoded_l2_sim/testset_size_{test_set_size}"
    res = {}
    for real_tumor, gt_top, enc_top in zip(tumor_ids, gt_lists, encoded_lists):
        best_enc_match = enc_top[0]
        try:
            index_in_gt = gt_top.index(best_enc_match)
        except ValueError:
            gt_best_extended = utils.find_n_best_score_ids(
                path=f"{folder_path}/gt/{real_tumor}_gt.json",
                n_best=200 if test_set_size == "200" else 1024,
                value_type=utils.DSValueType.T1C,
                order_func=min
            )
            try:
                index_in_gt = gt_best_extended.index(best_enc_match)
            except ValueError:
                index_in_gt = -1

        res[real_tumor] = {
            'encoded_best_match': best_enc_match,
            'unencoded_rank': index_in_gt,
        }

    if save:
        path = f"/home/ivan_marcel/thesis/src/autoencoder/data/encoded_l2_sim/{enc}_gt_match_pairs/testset_size_{test_set_size}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            json.dump(res, file)
    return res


def run_calc_encoded_sim_for_all_tumors(processes: int = 1, test_set_size: str = "200", latent_dim: int = 4096, train_size: int = 1500):
    """
    Run the encoded similarity calculation for all real tumors, comparing each tumor with all syn tumors in the dataset specified by test_set_size\n
    All results will be stored in the encoded l2 sim folder\n
    testset_size_{size}\n
    |--- tgmXXX_preop_encoded.json
    @threads - number of threads
    @test_set_size - name (typically size) of the encoded syn dataset
    """
    real_tumors = os.listdir(REAL_TUMOR_BASE_PATH)
    real_tumors.sort(key=lambda name: int(name[3:6]))

    func = partial(calc_encoded_similarities, test_set_size=test_set_size,
                   latent_dim=latent_dim, train_size=train_size)
    print(func)
    with multiprocessing.Pool(processes) as pool:
        results = pool.map_async(func, real_tumors)
        t = results.get()


def run_calc_groundtruth_sim_for_all_tumors(processes: int = 1, test_set_size: str = "200", metric: SimilarityMeasureType = SimilarityMeasureType.L2):
    """
    Run the encoded similarity calculation for all real tumors, comparing each tumor with all syn tumors in the dataset specified by test_set_size\n
    All results will be stored in the encoded l2 sim folder\n
    testset_size_{size}\n
    |--- tgmXXX_preop_encoded.json
    @threads - number of threads 
    @test_set_size - name (typically size) of the encoded syn dataset
    """
    real_tumors = os.listdir(REAL_TUMOR_BASE_PATH)
    real_tumors.sort(key=lambda name: int(name[3:6]))
    # func = partial(calc_groundtruth,
    #                test_set_size=test_set_size, metric=metric)
    # print(func)
    # print(multiprocessing.cpu_count())
    # with multiprocessing.Pool(processes=processes) as pool:
    #     results = pool.map_async(func, real_tumors)
    #     t = results.get()

    bar = Bar('Processing', max=len(real_tumors))
    for real_tumor in real_tumors:
        print(
            f"Starting calc for {real_tumor} @{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        calc_groundtruth(real_tumor=real_tumor,
                         test_set_size=test_set_size, metric=metric)
        bar.next()
    bar.finish()


def run(real_tumor):
    # run_calc_encoded_sim_for_all_tumors(
    #     test_set_size="2k", latent_dim=2048, train_size=1500)
    # calc_best_match_pairs(test_set_size="2k", enc="enc_2048_1500", save=True)
    # run_calc_groundtruth_sim_for_all_tumors(
    #     processes=1, test_set_size="2k", metric=SimilarityMeasureType.DICE)
    # sims = calc_similarity_of_top_lists(
    #     csv_path="/home/ivan_marcel/thesis/src/autoencoder/data/gt_enc_comp_200.csv", top_n=1, dataset_size="200", save=False)
    """Example usages"""
    df = run_top_15_comp(
        folder_path="/home/ivan_marcel/thesis/src/autoencoder/data/encoded_l2_sim/testset_size_2k", latent_dim=2048, train_size=1500)

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
