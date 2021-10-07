
import json
import logging
import os
from datetime import date, datetime

import numpy as np
from numpy.lib.arraysetops import intersect1d
from numpy.lib.utils import info
# import cProfile
# import pstats


import utils
from baseline import baseline, baseline_parallel
from faiss_src import playground, index_builder
from utils import DSValueType, SimilarityMeasureType

# '/home/marcel/Projects/uni/thesis/src/data/{id}_datadump.json'
DICE_SCORE_DATADUMP_PATH_TEMPLATE = '~/thesis/src/data/{id}_datadump.json'


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


logging.basicConfig(level=utils.LOG_LEVEL)
run_parallel_comparison(
    similarity_measure_type=SimilarityMeasureType.L2,
    is_test=True)

# faiss_comparison(real_tumor='tgm001_preop')
# faiss_comparison(real_tumor='tgm028_preop')
# faiss_comparison(real_tumor='tgm042_preop')
# faiss_comparison(real_tumor='tgm057_preop')
# faiss_comparison(real_tumor='tgm071_preop')
# run_baseline(real_tumor='tgm042_preop')
# run_baseline(real_tumor='tgm057_preop')
# run_baseline(real_tumor='tgm071_preop')
# run_top_10_l2_dice_comp()

# best_l2 = utils.find_n_best_score_ids(
#     path='/home/marcel/Projects/uni/thesis/src/data/baseline_data/2021-10-06 16:33:52_parallel_datadump.json',
#     value_type=DSValueType.COMBINED,
#     order_func=min,
#     n_best=10
# )

# print("best_l2_matches: ", best_l2)
# print("---")

# best_dice = utils.find_n_best_score_ids(
#     path='/home/marcel/Projects/uni/thesis/src/data/baseline_data/2021-09-30 19:23:06_parallel_datadump.json',
#     value_type=DSValueType.COMBINED,
#     order_func=max,
#     n_best=10
# )

# print("best_dice_matches: ", best_dice)
# print("---")
# print("intersection: ", list(set(best_dice) & set(best_l2)))
