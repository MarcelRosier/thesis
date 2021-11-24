import json
import multiprocessing
import os
from datetime import datetime
from functools import partial

import numpy as np
import utils
from utils import (SimilarityMeasureType, calc_dice_coef, load_real_tumor,
                   time_measure)
from constants import (ENV, REAL_TUMOR_PATH, SYN_TUMOR_BASE_PATH,
                       SYN_TUMOR_PATH_TEMPLATE)
REAL_TUMOR_PATH = REAL_TUMOR_PATH[ENV]
SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]
SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]


def get_scores_for_pair(measure_func, t1c, flair, tumor_folder):
    """
    Calculate the similarity score of the passed tumor pair
    """

    # load tumor data
    tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(
        id=tumor_folder))['data']

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

    # calc and update dice scores and partners
    cur_flair = measure_func(tumor_02, flair)
    cur_t1c = measure_func(tumor_06, t1c)
    combined = cur_t1c + cur_flair
    scores = {}
    scores[tumor_folder] = {
        't1c': cur_t1c,
        'flair': cur_flair,
        'combined': combined
    }
    return scores


@time_measure(log=True)
def get_scores_for_real_tumor_parallel(similarity_measure, processes, tumor_path, is_test=False, subset=None):
    """
    Calculate the best similarity measure score of the given tumor based on the given dataset and return tuple (scores, best_score)
    @similarity_measure determines the used comparison function 
    scores - dump of the individual scores
    best_score - info about the best combined score
    """
    (t1c, flair) = load_real_tumor(tumor_path)

    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders = [f for f in folders if f.isnumeric()]
    folders.sort(key=lambda f: int(f))
    # cap test set to 50k
    folders = folders[:50000]
    # only get a subset of the data if its a test
    # if is_test:
    #     folders = folders[:20]
    if subset is not None:
        folders = folders[subset[0]: subset[1]]
    print(f"{len(folders)=}")
    scores = {}

    print("Starting parallel loop for {} folders with {} processes".format(
        len(folders), processes))

    measure_func = utils.calc_dice_coef if similarity_measure == SimilarityMeasureType.DICE else utils.calc_l2_norm
    func = partial(get_scores_for_pair, measure_func, t1c, flair)
    with multiprocessing.Pool(processes) as pool:
        results = pool.map_async(func, folders)
        single_scores = results.get()
        scores = {k: v for d in single_scores for k, v in d.items()}

    # find best
    best_key = 0
    if similarity_measure == SimilarityMeasureType.DICE:
        best_key = max(scores.keys(), key=lambda k: scores[k]['combined'])
    elif similarity_measure == SimilarityMeasureType.L2:
        best_key = min(scores.keys(), key=lambda k: scores[k]['combined'])

    best_score = {
        'best_score': scores[best_key],
        'partner': best_key
    }
    return scores, best_score


def run(processes, similarity_measure_type=SimilarityMeasureType.DICE, tumor_path=REAL_TUMOR_PATH, is_test=False, subset=None):
    scores, best_score = get_scores_for_real_tumor_parallel(
        similarity_measure=similarity_measure_type,
        processes=processes,
        tumor_path=tumor_path,
        is_test=is_test,
        subset=subset)
    print(best_score)
    now_date = datetime.now().strftime("%Y-%m-%d")
    now_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename_dump = "data/{date}/{datetime}_parallel_datadump_{metric}.json".format(
        date=now_date, datetime=now_datetime, metric=similarity_measure_type.value)
    os.makedirs(os.path.dirname(filename_dump), exist_ok=True)

    with open(filename_dump, "w") as file:
        json.dump(scores, file)

    filename_best = "data/{date}/{datetime}_parallel_best_{metric}.json".format(
        date=now_date, datetime=now_datetime, metric=similarity_measure_type.value)
    os.makedirs(os.path.dirname(filename_best), exist_ok=True)

    with open(filename_best, "w") as file:
        json.dump(best_score, file)
    return best_score
