import json
import logging
import os
import time
from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from constants import ENV, SYN_TUMOR_PATH_TEMPLATE

SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]

LOG_LEVEL = logging.INFO


class DSValueType(Enum):
    """Dice score value types"""
    T1C = 't1c'
    FLAIR = 'flair'
    COMBINED = 'combined'


class SimilarityMeasureType(Enum):
    "Different similarity measures"
    DICE = 'dice'
    L2 = 'l2'


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


def calc_l2_norm(syn_data, real_data):
    """calculate the l2 norm of the two input data"""
    combined = syn_data - real_data
    # linalg.norm uses the Frobenius norm as default ord
    return np.linalg.norm(combined)


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


def find_n_best_score_ids(path, value_type, order_func, n_best=1):
    """
    Finds the n (@n_max) maximum values of given type stored in the given json file
    @path - path to json file
    @value_type - t1c/flair/combined
    @n_max - number of max_values
    """
    best_keys = []
    # cast enum to str if necessary
    if not isinstance(value_type, str):
        value_type = value_type.value
    with open(path) as json_file:
        data = json.load(json_file)
        for _ in range(n_best):
            best_key = order_func(
                data.keys(), key=lambda k: data[k][value_type])
            best_keys.append(best_key)
            del data[best_key]
    return best_keys


def plot_tumor(tumor, zoom_factor: float = 1.):
    """
    Usage: plot_tumor(tumor=load_single_tumor(tumor_id=1234), zoom_factor=0.5)
    """
    tumor = load_single_tumor(tumor_id=42)
    print(np.unique(tumor))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(zoom_factor)
    # downsample
    tumor = norm_and_threshold_tumor(zoom(tumor, zoom_factor))
    pos = np.where(tumor == 1)
    ax.scatter(pos[0], pos[1], pos[2], s=1.5)
    # ax.plot_wireframe(pos[0], pos[1], pos[2])
    plt.show()


def load_single_tumor(tumor_id):
    tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(id=tumor_id))[
        'data']

    # crop 129^3 to 128^3 if needed
    if tumor.shape != (128, 128, 128):
        tumor = np.delete(np.delete(
            np.delete(tumor, 128, 0), 128, 1), 128, 2)
    return norm_and_threshold_tumor(tumor)


def norm_and_threshold_tumor(tumor):
    # normalize
    max_val = tumor.max()
    if max_val != 0:
        tumor *= 1.0/max_val

    # threshold
    tumor[tumor < 0.6] = 0
    tumor[tumor >= 0.6] = 1
    return tumor


def pretty_print_params(BASE_CHANNELS=None,
                        MAX_EPOCHS=None,
                        LATENT_DIM=None,
                        MIN_DIM=None,
                        BATCH_SIZE=None,
                        TRAIN_SIZE=None,
                        VAL_SIZE=None,
                        LEARNING_RATE=None,
                        CHECKPOINT_FREQUENCY=None,
                        TEST_SIZE=None):
    from rich.console import Console
    from rich.table import Column, Table

    params_table = Table(
        show_header=True, header_style="bold #1DB954")
    params_table.add_column("Param")
    params_table.add_column("Value", style="#ffffff")
    if BASE_CHANNELS:
        params_table.add_row("BASE_CHANNELS", str(BASE_CHANNELS))
    if MAX_EPOCHS:
        params_table.add_row("MAX_EPOCHS", str(MAX_EPOCHS))
    if LATENT_DIM:
        params_table.add_row("LATENT_DIM", str(LATENT_DIM))
    if MIN_DIM:
        params_table.add_row("MIN_DIM", str(MIN_DIM))
    if BATCH_SIZE:
        params_table.add_row("BATCH_SIZE", str(BATCH_SIZE))
    if TRAIN_SIZE:
        params_table.add_row("TRAIN_SIZE", str(TRAIN_SIZE))
    if VAL_SIZE:
        params_table.add_row("VAL_SIZE", str(VAL_SIZE))
    if TEST_SIZE:
        params_table.add_row("TEST_SIZE", str(TEST_SIZE))
    if LEARNING_RATE:
        params_table.add_row("LEARNING_RATE", str(LEARNING_RATE))
    if CHECKPOINT_FREQUENCY:
        params_table.add_row("CHECKPOINT_FREQUENCY", str(CHECKPOINT_FREQUENCY))

    console = Console()
    console.print(params_table)


def pretty_print_gpu_info(info_list):
    from rich.console import Console
    from rich.table import Column, Table

    info_table = Table(
        show_header=True, header_style="bold #1DB954")
    info_table.add_column("GPU INFO")
    info_table.add_column("Value", style="#ffffff")

    for attr, val in info_list:
        info_table.add_row(attr, val)

    console = Console()
    console.print(info_table)
