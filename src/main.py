
from baseline import analysis
import json
import logging
import os
from datetime import date, datetime

import numpy as np
from numpy.lib.arraysetops import intersect1d
from numpy.lib.utils import info

import utils
from baseline import baseline, baseline_parallel
from constants import DICE_SCORE_DATADUMP_PATH_TEMPLATE, ENV, REAL_TUMOR_BASE_PATH, TEST_SET_RANGES
# from faiss_src import index_builder, playground
from utils import DSValueType, SimilarityMeasureType

# import cProfile
# import pstats


DICE_SCORE_DATADUMP_PATH_TEMPLATE = DICE_SCORE_DATADUMP_PATH_TEMPLATE[ENV]
REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]
TEST_SET_RANGES = TEST_SET_RANGES[ENV]


def run_parallel_comparison(similarity_measure_type, is_test=False):
    process_counts = [1, 2, 4, 8, 16, 32]

    print("Starting parallel baseline comparison for process_counts= {} and similarity_measure_type= {}".format(
          process_counts, similarity_measure_type))
    results = {}
    for p_count in process_counts:
        start = datetime.now()
        best = baseline_parallel.run(
            processes=p_count, similarity_measure_type=similarity_measure_type, is_test=is_test)
        end = datetime.now()
        total_seconds = str((end-start).total_seconds())
        results[p_count] = {
            'runtime': total_seconds,
            'partner': best['partner']
        }

    now_date = datetime.now().strftime("%Y-%m-%d")
    now_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = "data/{date}/{datetime}_comparison_{metric}.json".format(
        date=now_date, datetime=now_datetime, metric=similarity_measure_type.value)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        json.dump(results, file)


def run_baseline(real_tumor, is_test=True):
    baseline.run(real_tumor=real_tumor, is_test=is_test)


def faiss_comparison(real_tumor):
    logging.info(
        "Running faiss comparison for real_tumor {}".format(real_tumor))
    max_t1c_dice = utils.find_n_max_dice_score_ids(
        path=DICE_SCORE_DATADUMP_PATH_TEMPLATE.format(id=real_tumor),
        value_type=DSValueType.T1C,
        n_max=10)
    logging.info(max_t1c_dice)

    D, max_t1c_faiss = playground.run(real_tumor=real_tumor)
    intersection = intersect1d(np.asarray(
        [int(e) for e in max_t1c_dice]), max_t1c_faiss)
    logging.info("intersection: {}".format(intersection))


def run_faiss_test(real_tumor):
    index_builder.build_index()


def run_top_10_l2_dice_comp():
    best_dice = utils.find_n_best_score_ids(
        path='/home/rosierm/thesis/src/data/2021-10-06 11:18:55_parallel_datadump.json',
        value_type=DSValueType.COMBINED,
        order_func=max,
        n_best=10
    )

    best_l2 = utils.find_n_best_score_ids(
        path='/home/rosierm/thesis/src/data/2021-10-06 11:21:24_parallel_datadump.json',
        value_type=DSValueType.COMBINED,
        order_func=min,
        n_best=10
    )

    print("best_dice_matches: ", best_dice)
    print("---")
    print("best_l2_matches: ", best_l2)
    print("---")
    print("intersection: ", list(set(best_dice) & set(best_l2)))


def run_parallel_base():
    tumor_ids = os.listdir(REAL_TUMOR_BASE_PATH)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    # tumor_ids = tumor_ids[:10]

    for real_tumor in tumor_ids:
        print(f"running for {real_tumor}")
        real_tumor_path = os.path.join(REAL_TUMOR_BASE_PATH, real_tumor)
        testset_size = "50k"
        subset = (TEST_SET_RANGES[testset_size]['START'],
                  TEST_SET_RANGES[testset_size]['END'])
        baseline_parallel.run(
            processes=42,
            similarity_measure_type=SimilarityMeasureType.DICE,
            tumor_path=real_tumor_path,
            subset=subset,
            downsample_to=32
        )


logging.basicConfig(level=utils.LOG_LEVEL)

###
# Exec
###
run_parallel_base()
# analysis.compare_best_match_for_downsampling()
# run_parallel_base()
# gt_path = "/home/ivan_marcel/thesis/src/data/2021-11-24/tgm_001_vs_3000_3199_l2/2021-11-24 14:25:47_parallel_datadump_l2.json"
# new_path = "/home/ivan_marcel/thesis/src/autoencoder/data/2021-11-24/2021-11-24 15:35:18_dump_l2.json"
# gt_res = utils.find_n_best_score_ids(
#     path=gt_path,
#     n_best=15,
#     value_type=utils.DSValueType.T1C,
#     order_func=min
# )
# new_res = utils.find_n_best_score_ids(
#     path=new_path,
#     n_best=15,
#     value_type=utils.DSValueType.T1C,
#     order_func=min
# )
# print(gt_res)
# print(new_res)

# run_faiss_test(real_tumor='test')

# run_parallel_comparison(
#     similarity_measure_type=SimilarityMeasureType.L2,
#     is_test=True)
# faiss_comparison(real_tumor='tgm001_preop')
# faiss_comparison(real_tumor='tgm028_preop')
# faiss_comparison(real_tumor='tgm042_preop')
# faiss_comparison(real_tumor='tgm057_preop')
# faiss_comparison(real_tumor='tgm071_preop')
# run_baseline(real_tumor='tgm042_preop')
# run_baseline(real_tumor='tgm057_preop')
# run_baseline(real_tumor='tgm071_preop')
# run_top_10_l2_dice_comp()
