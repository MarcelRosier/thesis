import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from constants import ENV, SYN_TUMOR_BASE_PATH, SYN_TUMOR_PATH_TEMPLATE

SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]
SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]


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


def time_measure(log: bool = False):
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


def get_number_of_entries(path: str):
    data = {}
    with open(path) as json_file:
        data = json.load(json_file)
    print("data_len: ", len(data))
    return len(data)


def load_real_tumor(base_path: str, downsample_to: int = None):
    """
    @base_path: path to the real tumor folder, e.g. /tgm001_preop/ \n
    Return pair (t1c,flair) of a real tumor
    """
    t1c = nib.load(os.path.join(
        base_path, 'tumor_mask_t_to_atlas229.nii')).get_fdata()
    flair = nib.load(os.path.join(
        base_path, 'tumor_mask_f_to_atlas229.nii')).get_fdata()

    flair = torch.from_numpy(flair)
    t1c = torch.from_numpy(t1c)

    t1c = zoom(F.pad(t1c, (32, 31, 14, 13, 32, 31)), zoom=0.5, order=0)
    flair = zoom(F.pad(flair, (32, 31, 14, 13, 32, 31)), zoom=0.5, order=0)

    if downsample_to:
        t1c = zoom(t1c, zoom=downsample_to/128, order=0)
        flair = zoom(flair, zoom=downsample_to/128, order=0)
    return (t1c, flair)


def find_n_best_score_ids(path: str, value_type: DSValueType, order_func, n_best=1):
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


def load_single_tumor(tumor_id, threshold=0.6):
    """
    Loads a single syntethic, thresholded, normalized tumor with dim 128^3 \n
    @tumor_id = folder_id of the syn tumor\n
    @threshold = threshold value (typically 0.6 (t1c) / 0.2 (flair))
    """
    tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(id=tumor_id))[
        'data']

    # crop 129^3 to 128^3 if needed
    if tumor.shape != (128, 128, 128):
        tumor = np.delete(np.delete(
            np.delete(tumor, 128, 0), 128, 1), 128, 2)
    return norm_and_threshold_tumor(tumor, threshold)


def normalize(v):
    """ Return a unit vector of @v"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def load_reconstructed_tumor(path):
    """
    Load a reconstructed syntethic tumor\n
    This differs from the usual tumor loading in the format of the saved tumor
    """
    tumor = np.load(path)[0][0]
    # crop 129^3 to 128^3 if needed
    if tumor.shape != (128, 128, 128):
        tumor = np.delete(np.delete(
            np.delete(tumor, 128, 0), 128, 1), 128, 2)
    return norm_and_threshold_tumor(tumor)


def norm_and_threshold_tumor(tumor, threshold=0.6):
    # normalize
    max_val = tumor.max()
    if max_val != 0:
        tumor *= 1.0/max_val

    # threshold
    tumor[tumor < threshold] = 0
    tumor[tumor >= threshold] = 1
    return tumor


def get_sorted_syn_tumor_list() -> List[str]:
    """ Return a list of all syntethic tumors sorted by id"""
    folders = os.listdir(SYN_TUMOR_BASE_PATH)
    folders = [f for f in folders if f.isnumeric()]
    folders.sort(key=lambda f: int(f))
    return folders


##################################
# Pretty printing
##################################


def pretty_print_params(BASE_CHANNELS=None,
                        MAX_EPOCHS=None,
                        LATENT_DIM=None,
                        MIN_DIM=None,
                        BATCH_SIZE=None,
                        TRAIN_SIZE=None,
                        VAL_SIZE=None,
                        LEARNING_RATE=None,
                        CHECKPOINT_FREQUENCY=None,
                        TEST_SIZE=None,
                        SYNTHETIC=None,
                        VAE=None,
                        BETA=None):
    """Print a table with all passed Parameters\n If the rich module is not installed the info will be printed to the console"""

    try:
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
            params_table.add_row("CHECKPOINT_FREQUENCY",
                                 str(CHECKPOINT_FREQUENCY))
        if SYNTHETIC is not None:
            params_table.add_row("SYNTHETIC", str(SYNTHETIC))
        if VAE is not None:
            params_table.add_row("VAE", str(VAE))
        if BETA is not None:
            params_table.add_row("BETA", str(BETA))

        console = Console()
        console.print(params_table)

    except ImportError:
        print(f"INFO:\n{BASE_CHANNELS=}\n{MAX_EPOCHS=}\n{LATENT_DIM=}\n{MIN_DIM=}\n{BATCH_SIZE=}\n{TRAIN_SIZE=}\n{VAL_SIZE=}\n{LEARNING_RATE=}\n{CHECKPOINT_FREQUENCY=}")


def pretty_print_gpu_info(device):
    """Print a table with the passed GPU info\n If the rich module is not installed the info will be printed to the console"""

    info_list = [("CUDA_VISIBLE_DEVICES",
                  f"[{os.environ['CUDA_VISIBLE_DEVICES']}]"),
                 ("Device:", str(device)),
                 ("Active CUDA Device: GPU", torch.cuda.get_device_name())]
    try:
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

    except ImportError:
        for attr, val in info_list:
            print(f"{attr}={val}")


##################################
# Plotting
##################################

def plot_tumor(tumor, cmp_tumor=None, title="title", c_base='b', zoom_factor: float = 0.5):
    """
    Plot a tumor in a 3D view.\n
    If cmp_tumor is passed this tumor will be plotted in the same view in the color magenta.\n
    The base tumor has color blue or c_base if used.\n
    @tumor - tumor that will be plotted
    @cmp_tumor - [optional] a second tumor for comparsion
    @title - title of the plot
    @c_base - color of the scatterplot dots, default is blue\n
    Usage: plot_tumor(tumor=load_single_tumor(tumor_id=1234))
    """
    # tumor = load_single_tumor(tumor_id=42)
    # print(np.unique(tumor))
    fig = plt.figure()
    plt.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    # downsample
    pos = np.where(tumor == 1)
    ax.scatter(pos[0], pos[1], pos[2], s=1.5, c=c_base)
    if cmp_tumor is not None:
        cmp_pos = np.where(cmp_tumor == 1)
        ax.scatter(cmp_pos[0], cmp_pos[1], cmp_pos[2], s=1.5, c='m')
    # ax.plot_wireframe(pos[0], pos[1], pos[2])
    # plt.show()


def plot_tumor_overview(latend_dim):
    """
    Plot an overview for tumor encoding:\n
    - input tumor
    - reconstructed (output) tumor
    - differences between input and output
    - both tumors in the same plot, blue= input, magenta= output
    """
    rec_path = "/home/marcel/Projects/uni/thesis/src/data/test_3000.npy" if latend_dim == 4096 else "/home/marcel/Projects/uni/thesis/media/3000_reconstructed_2048.npy"
    base = load_single_tumor(tumor_id=3000)
    output = load_reconstructed_tumor(
        path=rec_path)
    intersection = (base != output)

    plot_tumor(base, title=f"input, {latend_dim}")
    plot_tumor(output, title=f"output, {latend_dim}", c_base='m')
    plot_tumor(tumor=intersection, title=f"input != output, {latend_dim}")
    plot_tumor(base, cmp_tumor=output, title=f"input and output, {latend_dim}")

    # plt.show()


def plot_tumor_list(tumor_list: List[int]):
    """ Plot each of the given tumors in a separate view """
    tumors = [load_single_tumor(tumor_id=tumor_id) for tumor_id in tumor_list]
    for i, tumor in enumerate(tumors):
        plot_tumor(tumor, title=tumor_list[i])
    # dice
    dice = calc_dice_coef(tumors[0], tumors[0])
    print(f"{dice=}")

    # plot
    plt.show()


# plot_tumor_list(tumor_list=[3048, 3144])

def test():
    tumor, _ = load_real_tumor(
        base_path="/home/marcel/Projects/uni/thesis/real_tumors/tgm001_preop")
    print(tumor.shape)


# test()
# plot_tumor_overview(latend_dim=4096)
# plot_tumor_overview(latend_dim=2048)
# plt.show()
