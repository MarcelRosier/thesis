import json
import logging
import os
from datetime import datetime

import numpy as np
from utils import calc_dice_coef, load_real_tumor, time_measure

# tumor_mask_f_to_atlas229 ; tumor_mask_t_to_atlas229
# tgm001_preop'
REAL_TUMOR_PATH = '/home/marcel/Projects/uni/thesis/real_tumors/{id}'
SYN_TUMOR_BASE_PATH = '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset'
SYN_TUMOR_PATH_TEMPLATE = '//home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset/{id}/Data_0001.npz'

T1C_PATH = '/home/rosierm/kap_2021/dice_analysis/tumor_mask_t_to_atlas.nii'
FLAIR_PATH = '/home/rosierm/kap_2021/dice_analysis/tumor_mask_f_to_atlas.nii'
TUMOR_SUBSET_TESTING_SIZE = 200


@time_measure(log=True)
def get_dice_scores_for_real_tumor(tumor_path, is_test=False):
    """
    Calculate the max dice score of the given tumor based on the given dataset and return tuple (max_t1c, max_flair),
    each is a dict containing (max_score, partner_tumor)
    """
    (t1c, flair) = load_real_tumor(tumor_path)

    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders.sort(key=lambda f: int(f))

    # only get a subset of the data if its a test
    if is_test:
        folders = folders[:TUMOR_SUBSET_TESTING_SIZE]

    # init dicts
    scores = {}
    maximum = {
        'max_score': 0,
        'partner': None
    }

    print("Starting loop for {} folders".format(len(folders)))
    # loop through synthetic tumors
    for f in folders:
        # load tumor data
        tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(id=f))['data']

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
        scores[f] = {
            't1c': cur_t1c,
            'flair': cur_flair,
            'combined': combined
        }
        if maximum['max_score'] < combined:
            maximum['max_score'] = combined
            maximum['partner'] = f

    return scores, maximum


def run(real_tumor, is_test=False):
    (scores, maximum) = get_dice_scores_for_real_tumor(
        tumor_path=REAL_TUMOR_PATH.format(id=real_tumor), is_test=is_test)
    logging.info("maximum dice score for tumor {}: {}".format(
        real_tumor, maximum))
    # now = datetime.now().strftime("%Y-%m-%d")
    with open("data/{tumor}_datadump.json".format(tumor=real_tumor), "w") as file:
        json.dump(scores, file)
    with open("data/{tumor}_maximum.json".format(tumor=real_tumor), "w") as file:
        json.dump(maximum, file)

    # print(calc_tumor_score_map(working=False))
