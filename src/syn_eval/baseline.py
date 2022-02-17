import json
import multiprocessing
import os
from datetime import datetime
from functools import partial
from typing import Tuple
from monai.metrics import compute_meandice
from monai.losses.dice import DiceLoss
import numpy as np
import torch
import utils
from constants import (BASELINE_SIMILARITY_BASE_PATH, ENV, REAL_TUMOR_PATH,
                       SYN_TUMOR_BASE_PATH, SYN_TUMOR_PATH_TEMPLATE)
from scipy.ndimage import zoom
from utils import (SimilarityMeasureType, load_real_tumor,
                   time_measure)

REAL_TUMOR_PATH = REAL_TUMOR_PATH[ENV]
SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]
SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]
BASELINE_SIMILARITY_BASE_PATH = BASELINE_SIMILARITY_BASE_PATH[ENV]


def load_syn_tumor(tumor_id, downsample_to: int = None):
    """returns 02,06 (flair,t1c)"""
    # load tumor data
    tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(
        id=tumor_id))['data']

    # crop 129^3 to 128^3 if needed
    if tumor.shape != (128, 128, 128):
        tumor = np.delete(np.delete(
            np.delete(tumor, 128, 0), 128, 1), 128, 2)

    if downsample_to:
        tumor = zoom(tumor, zoom=downsample_to/128, order=0)
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

    return tumor_02, tumor_06


def get_scores_for_pair(t1c, flair, downsample_to, tumor_id):
    """
    Calculate the similarity score of the passed tumor pair
    """

    tumor_02, tumor_06 = load_syn_tumor(
        tumor_id=tumor_id, downsample_to=downsample_to)

    cur_flair = utils.calc_dice_coef(tumor_02, flair)
    cur_t1c = utils.calc_dice_coef(tumor_06, t1c)
    combined = cur_t1c + cur_flair

    scores = {}
    scores[tumor_id] = {
        't1c': cur_t1c,
        'flair': cur_flair,
        'combined': combined
    }
    return scores


def get_scores_for_real_tumor_parallel(input_tumor_id: str, downsample_to: int = None):
    t1c, flair = load_syn_tumor(input_tumor_id, downsample_to=downsample_to)

    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders = [f for f in folders if f.isnumeric()]
    folders.sort(key=lambda f: int(f))
    # cap test set to 50k
    folders = folders[:50000]
    scores = {}

    func = partial(get_scores_for_pair, t1c, flair, downsample_to)

    with multiprocessing.Pool(processes=50) as pool:
        results = pool.map_async(func, folders)
        single_scores = results.get()
        scores = {k: v for d in single_scores for k, v in d.items()}
    # scores = {}
    # for f in folders:
    #     r = func(f)
    #     scores[f] = r[f]

    # find best
    best_key = max(scores.keys(), key=lambda k: scores[k]['combined'])
    best_score = {
        'best_score': scores[best_key],
        'partner': best_key
    }
    return scores, best_score


def run(input_tumor_id, downsample_to: int = None, validation: bool = False):
    scores, best_score = get_scores_for_real_tumor_parallel(
        input_tumor_id=input_tumor_id,
        downsample_to=downsample_to)

    print(best_score)

    if validation:
        data_path = "/home/ivan_marcel/thesis/src/syn_eval/val_data"
    else:
        data_path = "/home/ivan_marcel/thesis/src/syn_eval/data"
    save_path = f"{data_path}/{'down'+str(downsample_to) if downsample_to else 'baseline'}/{input_tumor_id}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        json.dump(scores, file)
    return best_score
