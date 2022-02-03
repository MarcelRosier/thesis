import json
import multiprocessing
import os
from datetime import datetime
from functools import partial
from typing import Tuple

import numpy as np
import utils
from constants import (BASELINE_SIMILARITY_BASE_PATH, ENV, REAL_TUMOR_PATH,
                       SYN_TUMOR_BASE_PATH, SYN_TUMOR_PATH_TEMPLATE, REAL_TUMOR_BASE_PATH)
from scipy.ndimage import zoom
from utils import (SimilarityMeasureType, load_real_tumor,
                   time_measure)

REAL_TUMOR_PATH = REAL_TUMOR_PATH[ENV]
REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]
SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]
SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]
BASELINE_SIMILARITY_BASE_PATH = BASELINE_SIMILARITY_BASE_PATH[ENV]


def load_syn_tumor(syn_tumor_id):
    # load tumor data
    syn_tumor = np.load(
        SYN_TUMOR_PATH_TEMPLATE.format(id=syn_tumor_id))['data']

    # crop 129^3 to 128^3 if needed
    if syn_tumor.shape != (128, 128, 128):
        syn_tumor = np.delete(np.delete(
            np.delete(syn_tumor, 128, 0), 128, 1), 128, 2)

    # normalize
    max_val = syn_tumor.max()
    if max_val != 0:
        syn_tumor *= 1.0/max_val

    # threshold
    flair = np.copy(syn_tumor)
    flair[flair < 0.2] = 0
    flair[flair >= 0.2] = 1
    t1c = np.copy(syn_tumor)
    t1c[t1c < 0.6] = 0
    t1c[t1c >= 0.6] = 1
    return t1c, flair


def get_best_match_for_syn_tumor(measure_func, syn_tumor_id):
    input_t1c, input_flair = load_syn_tumor(syn_tumor_id)
    # load X syn tumors
    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders = [f for f in folders if f.isnumeric()]
    folders.sort(key=lambda f: int(f))
    # cap test set to 50k subset (0,50k)
    size = 40
    folders = folders[:size]
    # remove input from list
    # assert syn_tumor_id in folders
    folders.remove(syn_tumor_id)
    # assert syn_tumor_id not in folders

    # init best_mach
    best_match = {
        't1c': -1,
        'flair': -1,
        'combined': -1,
        'partner': -1,
    }
    for ref_syn_tumor_id in folders:
        (ref_t1c, ref_flair) = load_syn_tumor(ref_syn_tumor_id)
        cur_flair = measure_func(input_flair, ref_flair)
        cur_t1c = measure_func(input_t1c, ref_t1c)
        cur_combined = cur_flair + cur_t1c
        if cur_combined > best_match['combined']:
            best_match = {
                't1c': cur_t1c,
                'flair': cur_flair,
                'combined': cur_combined,
                'partner': ref_syn_tumor_id,
            }
    return {
        syn_tumor_id: best_match
    }


def run(processes):
    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders = [f for f in folders if f.isnumeric()]
    folders.sort(key=lambda f: int(f))
    # cap test set to 50k subset (0,50k)
    size = 40
    folders = folders[:size]
    # use dice metric
    measure_func = utils.calc_dice_coef
    func = partial(get_best_match_for_syn_tumor, measure_func)
    with multiprocessing.Pool(processes) as pool:
        results = pool.map_async(func, folders)
        best_matches = results.get()

    print(best_matches)
    save_path = f"/home/ivan_marcel/thesis/src/lucas_exp/data/{size}/dice/syn_tumor_mapping.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        json.dump(best_matches, file)
    return best_matches
