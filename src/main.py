import json
from datetime import date, datetime
import logging

from numpy.lib.arraysetops import intersect1d
from numpy.lib.utils import info

import utils
from baseline import baseline, baseline_parallel
from faiss_src import playground
from utils import DSValueType
import numpy as np

# baseline.run()
# baseline_parallel.run()
DICE_SCORE_DATADUMP_PATH_TEMPLATE = '/home/marcel/Projects/uni/thesis/src/data/{id}_datadump.json'


def run_parallel_comparison(is_test=False):
    process_counts = [1, 2, 4, 8, 16, 32]

    results = {}
    for p_count in process_counts:
        start = datetime.now()
        maximum = baseline_parallel.run(processes=p_count, is_test=is_test)
        end = datetime.now()
        total_seconds = str((end-start).total_seconds())
        results[p_count] = {
            'runtime': total_seconds,
            'partner': maximum['partner']
        }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("data/{}_comparison.json".format(now), "w") as file:
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


logging.basicConfig(level=utils.LOG_LEVEL)
# run_parallel_comparison(is_test=True)
faiss_comparison(real_tumor='tgm001_preop')
faiss_comparison(real_tumor='tgm028_preop')
faiss_comparison(real_tumor='tgm042_preop')
faiss_comparison(real_tumor='tgm057_preop')
faiss_comparison(real_tumor='tgm071_preop')
# run_baseline(real_tumor='tgm042_preop')
# run_baseline(real_tumor='tgm057_preop')
# run_baseline(real_tumor='tgm071_preop')
