import time
from datetime import datetime
import numpy as np
import json
from scipy.ndimage import zoom
import torch
import nibabel as nib
import os
import torch.nn.functional as F


def time_measure(log=False):
    def timing_base(f):
        def wrap(*args, **kwargs):
            start = time.time()
            ret = f(*args, **kwargs)
            end = time.time()
            output = '{:s} function took {:.3f} ms -> {:.0f} h {:.0f} m {:.0f} s'.format(
                f.__name__, (end-start)*1000.0, (end-start) // (60 * 60), ((end-start) // 60) % 60, (end-start) % 60)
            print(output)
            if log:
                with open("run_time.log", "a+") as file:
                    file.write('started: {}\nend: {} \n{}\n{}\n'.format(
                        datetime.fromtimestamp(start), datetime.fromtimestamp(end), output, ("-"*10)))
            return ret
        return wrap
    return timing_base


def calc_dice_coef(syn_data, real_data):
    """calculate the dice coefficient of the two input data"""
    combined = syn_data + real_data
    intersection = np.count_nonzero(combined == 2)
    union = np.count_nonzero(syn_data) + np.count_nonzero(real_data)
    if union == 0:
        return 0
    return (2 * intersection) / union


def get_number_of_entries(path):
    data = {}
    with open(path) as json_file:
        data = json.load(json_file)
    print("data_len: ", len(data))
    return len(data)


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
