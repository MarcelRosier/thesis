import json
import os
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import time
from datetime import datetime


def time_measure(log=False):
    def timing_base(f):
        def wrap(*args, **kwargs):
            start = time.time()
            ret = f(*args, **kwargs)
            end = time.time()
            output = '{:s} function took {:.3f} ms'.format(
                f.__name__, (end-start)*1000.0)
            print(output)
            if log:
                with open("run_time.log", "a+") as file:
                    file.write('started: {}\nend: {} \n{}\n{}\n'.format(
                        datetime.fromtimestamp(start), datetime.fromtimestamp(end), output, ("-"*10)))

            return ret
        return wrap
    return timing_base


@time_measure(log=True)
def test():
    for i in range(1000000):
        pass


# tumor_mask_f_to_atlas229 ; tumor_mask_t_to_atlas229
REAL_TUMOR_PATH = '/home/rosierm/marcel_tgm/tgm001_preop'
SYN_TUMOR_BASE_PATH = '/home/rosierm/samples_extended/Dataset'
SYN_TUMOR_PATH_TEMPLATE = '/home/rosierm/samples_extended/Dataset/{id}/Data_0001.npz'

T1C_PATH = '/home/rosierm/kap_2021/dice_analysis/tumor_mask_t_to_atlas.nii'
FLAIR_PATH = '/home/rosierm/kap_2021/dice_analysis/tumor_mask_f_to_atlas.nii'


def calc_dice_coef(syn_data, real_data):
    """calcualte the dice coefficient of the two input data"""
    combined = syn_data + real_data
    intersection = np.count_nonzero(combined == 2)
    union = np.count_nonzero(syn_data) + np.count_nonzero(real_data)
    if union == 0:
        return 0
    return (2 * intersection) / union


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


@time_measure(log=True)
def get_dice_scores_for_real_tumor(tumor_path):
    """
    Calculate the max dice score of the given tumor based on the given dataset and return tuple (max_t1c, max_flair),
    each is a dict containing (max_score, partner_tumor)
    """
    (t1c, flair) = load_real_tumor(tumor_path)

    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders.sort(key=lambda f: int(f))

    # init dicts
    scores = {}
    maximum = {
        'max_score': 0,
        'partner': None
    }

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


def run():
    (scores, maximum) = get_dice_scores_for_real_tumor(tumor_path=REAL_TUMOR_PATH)
    print(maximum)
    with open("datadump.json", "w") as file:
        json.dump(scores, file)
    with open("maximum.json", "w") as file:
        json.dump(maximum, file)

    # print(calc_tumor_score_map(working=False))
