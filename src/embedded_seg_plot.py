import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from glob import glob

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pickle5 as pickle
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom

import utils

# mpl.rc('image', cmap='terrain')

# sys.path.append('/home/ivan/.local/lib/python3.8/site-packages')

#import tensorflow as tf
#import torch
#import surface_distance
# torch.set_num_threads(2) #uses max. 2 cpus for inference! no gpu inference!
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
#from masseffect_solver import masseffect_solver


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def custom_cmap():
    colors = [(116, 197, 247), (214, 10, 10)]  # R -> G -> B
    cmap_name = 'my_list'

    cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    return cmap


def load_syn(tid):
    # load tumor data
    tumor = np.load(
        f"/Users/marcelrosier/Projects/uni/tumor_data/samples_extended/Dataset/{tid}/Data_0001.npz")['data']

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
    return tumor_06, tumor_02


mpl.rcParams['text.color'] = 'black'
mpl.rcParams['xtick.color'] = 'grey'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
plt.rcParams["font.family"] = "Laksaman"
# path = "/home/ivan/ibbm/learnmorph/bene/"
path_to_orig = "/Users/marcelrosier/Projects/uni/tumor_data/real_tumors/"
dirs = os.listdir(path_to_orig)
dirs.sort()
ii = 1
for folder in dirs:
    folder = "tgm001_preop"
    # # load

    # print(queried_tumor_mri.shape)
    # MRI SCAN
    # flair_seg = nib.load(path_to_orig + folder +
    #                      "/tumor_mask_f_to_atlas229.nii")
    # flair_seg = flair_seg.get_fdata()
    # t1c_seg = nib.load(path_to_orig + folder +
    #                    "/tumor_mask_t_to_atlas229.nii")
    # t1c_seg = t1c_seg.get_fdata()
    t1c_seg, flair_seg = utils.load_real_tumor(path_to_orig + folder)
    data_segm = flair_seg + t1c_seg
    flair_scan_dep = data_segm > 0

    flair_scan_dep = flair_seg
    t1gd_scan_dep = t1c_seg

    atlas_path = "/Users/marcelrosier/Projects/uni/tumor_data/Atlas/atlas_t1_masked.nii"
    # data_brain = nib.load(path_to_orig + folder + "/atlas_t1.nii")
    data_brain = nib.load(atlas_path)
    # data_brain = nib.load(
    #     "/Users/marcelrosier/Projects/uni/tumor_data/Atlas/atlas_t1_masked.nii")
    data_brain = data_brain.get_fdata()
    data_brain = torch.from_numpy(data_brain)
    data_brain = zoom(
        F.pad(data_brain, (32, 31, 14, 13, 32, 31)), zoom=0.5, order=0)

    data_brain = (data_brain - np.min(data_brain)) / \
        (np.max(data_brain) - np.min(data_brain))
    # syn tumor baseline
    baseline_t1c, baseline_flair = load_syn(tid=39725)
    baseline_data_segm = baseline_flair + baseline_t1c
    baseline_flair_scan_dep = baseline_data_segm > 0

    baseline_flair_scan_dep = baseline_flair
    baseline_t1gd_scan_dep = baseline_t1c
    # process
    mri_scan_dep = flair_scan_dep
    mask_mri = np.ma.masked_where((t1gd_scan_dep.astype(int) + flair_scan_dep.astype(
        int)) == 0, t1gd_scan_dep.astype(int)+flair_scan_dep.astype(int))
    mask_baseline = np.ma.masked_where((baseline_t1gd_scan_dep.astype(int) + baseline_flair_scan_dep.astype(
        int)) == 0, baseline_t1gd_scan_dep.astype(int)+baseline_flair_scan_dep.astype(int))

    meanvalues = np.mean(mri_scan_dep, axis=(0, 1))
    s = np.argmax(meanvalues)
    showparameters = 0

    print(t1c_seg.shape)
    print(baseline_t1c.shape)
    print(data_brain.shape)
    # fig = plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 4),
                             gridspec_kw={"width_ratios": [1, 1]})
    # fig, (ax, ax1, ax2, cax2) = plt.subplots(ncols=4, nrows=5, figsize=(15, 4),
    #                                          gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    # outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    # fig.subplots_adjust(wspace=0.02)
    axes[0][0].set_ylabel("MRI".format(ii), fontsize=18)
    axes[1][0].set_ylabel("Best match", fontsize=18)
    # axes[2][0].set_ylabel("DS 64", fontsize=18)

    for i in range(2):
        axes[0][i].set_title(i)
        axes[0][i].imshow(data_brain[:, :, s].T[10:120, 15:115], cmap=cm.gray)
        remove_ticks(axes[0][i])
        axes[0][i].imshow(mask_mri[:, :, s].T[10:120, 15:115],
                          cmap=cm.bwr)  # tab20, bwr?
        # print(data_brain[:,:,s].T.shape)

        axes[1][i].imshow(
            data_brain[:, :, s].T[10:120, 15:115], cmap=cm.gray)
        axes[1][i].imshow(mask_baseline[:, :, s].T[10:120, 15:115],
                          cmap=cm.bwr)  # tab20, bwr?
        dice_t1gd = 0.00
        dice_flair = 0.00
        axes[1][i].set_xlabel(
            r'$Dice_{T1Gd}$' + f" = {dice_flair} " + r'$Dice_{FLAIR}$' + f" = {dice_flair}")
        remove_ticks(axes[1][i])

        # axes[2][i].imshow(data_brain[:, :, s].T[10:120, 15:115], cmap=cm.gray)
        # # im2 = axes[2][i].imshow(fk[:, :, s].T[10:220, 15:225], cmap=cm.jet)
        # remove_ticks(axes[2][i])

        # if i >= 0:
        #     cb = fig.colorbar(
        #         im2, cax=axes[i][3], ticks=list(np.linspace(0, 1, 11)))
        #     cb.ax.tick_params(labelsize=14)
        # else:
        #     axes[i][3].axis('off')
    # fig.savefig("./plot_{}.png".format(ii), bbox_inches='tight', dpi=600)
    plt.show()
    ii += 1
    if ii == 2:
        break
