import json
import os
from pickle import PROTO
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from utils import time_measure, calc_dice_coef
import multiprocessing
from functools import partial
from datetime import datetime


# tumor_mask_f_to_atlas229 ; tumor_mask_t_to_atlas229
REAL_TUMOR_PATH = '/home/rosierm/marcel_tgm/tgm001_preop'
SYN_TUMOR_BASE_PATH = '/home/rosierm/samples_extended/Dataset'
SYN_TUMOR_PATH_TEMPLATE = '/home/rosierm/samples_extended/Dataset/{id}/Data_0001.npz'

T1C_PATH = '/home/rosierm/kap_2021/dice_analysis/tumor_mask_t_to_atlas.nii'
FLAIR_PATH = '/home/rosierm/kap_2021/dice_analysis/tumor_mask_f_to_atlas.nii'
PROCESSES = 7


def load_real_tumor(base_path):
    """Return pair (t1c,flair) of a real tumor"""
    t1c = nib.load(os.path.join(
        base_path, 'tumor_mask_t_to_atlas229.nii')).get_fdata()
    flair = nib.load(os.path.join(
        base_path, 'tumor_mask_f_to_atlas229.nii')).get_fdata()

    flair = torch.from_numpy(flair)
    t1c = torch.from_numpy(t1c)

    t1c = zoom(F.pad(t1c, (32, 31, 14, 13, 32, 31)), zoom=0.5, order=0)
    flair = zoom(F.pad(flair, (32, 31, 14, 13, 32, 31)), zoom=0.5, order=0)

    return (t1c, flair)


def get_dice_scores_for_pair(t1c, flair, tumor_folder):
    """
    Calculate the max dice score of the given tumor based on the given ***subset*** of the dataset and return tuple (scores, maximum)
    scores - dump of the individual scores
    maximum - info about the best combined dice score
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
    cur_flair = calc_dice_coef(tumor_02, flair)
    cur_t1c = calc_dice_coef(tumor_06, t1c)
    combined = cur_t1c + cur_flair
    scores = {}
    scores[tumor_folder] = {
        't1c': cur_t1c,
        'flair': cur_flair,
        'combined': combined
    }
    return scores


@time_measure(log=True)
def get_dice_scores_for_real_tumor_parallel(tumor_path, is_test=False):
    """
    Calculate the max dice score of the given tumor based on the given dataset and return tuple (scores, maximum)
    scores - dump of the individual scores
    maximum - info about the best combined dice score
    """
    (t1c, flair) = load_real_tumor(tumor_path)

    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders.sort(key=lambda f: int(f))
    # only get a subset of the data if its a test
    if is_test:
        folders = folders[:1000]
    scores = {}

    print("Starting parallel loop for {} folders".format(len(folders)))
    func = partial(get_dice_scores_for_pair, t1c, flair)
    with multiprocessing.Pool(PROCESSES) as pool:
        results = pool.map_async(func, folders)
        single_scores = results.get()
        scores = {k: v for d in single_scores for k, v in d.items()}

    # find max
    max_key = max(scores.keys(), key=lambda k: scores[k]['combined'])
    maximum = {
        'max_score': scores[max_key],
        'partner': max_key
    }
    return scores, maximum


def run(is_test=False):
    scores, maximum = get_dice_scores_for_real_tumor_parallel(
        tumor_path=REAL_TUMOR_PATH, is_test=is_test)
    print(maximum)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("data/{}_parallel_datadump.json".format(now), "w") as file:
        json.dump(scores, file)
    with open("data/{}_parallel_maximum.json".format(now), "w") as file:
        json.dump(maximum, file)
