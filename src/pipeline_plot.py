import sys
from datetime import datetime
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import subprocess
import pickle5 as pickle
from glob import glob
import nibabel as nib
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# mpl.rc('image', cmap='terrain')

# sys.path.append('/home/ivan/.local/lib/python3.8/site-packages')

# import tensorflow as tf
# import torch
# import surface_distance
# torch.set_num_threads(2) #uses max. 2 cpus for inference! no gpu inference!
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from masseffect_solver import masseffect_solver


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def custom_cmap():
    colors = [(116, 197, 247), (214, 10, 10)]  # R -> G -> B
    cmap_name = 'my_list'

    cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    return cmap


mpl.rcParams['text.color'] = 'black'
mpl.rcParams['xtick.color'] = 'grey'
mpl.rcParams['ytick.color'] = 'grey'
mpl.rcParams['axes.labelcolor'] = 'grey'
plt.rcParams["font.family"] = "Laksaman"

# path = "/home/ivan/ibbm/learnmorph/bene/"
path_to_orig = "/Users/marcelrosier/Projects/uni/tumor_data/real_tumors/"
# dirs = os.listdir(path_to_orig)
# dirs.sort()
dirs = [
    'tgm047_preop',  # 1.4
    'tgm008_preop',  # 1.2
    'tgm051_preop',    # 1.0
    'tgm025_preop',  # 0.8
    'tgm023_preop',  # 0.6
]

# fig = plt.figure(figsize=(10, 8))
fig, axes = plt.subplots(ncols=4, nrows=5, figsize=(15, 4),
                         gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
# fig, (ax, ax1, ax2, cax2) = plt.subplots(ncols=4, nrows=5, figsize=(15, 4),
#                                          gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
# outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

# fig.subplots_adjust(wspace=0.02)
axes[0][0].set_title("Patient's brain MRI", fontsize=18)
axes[0][1].set_title("MRI segmentations", fontsize=18)
axes[0][2].set_title("Queried tumor", fontsize=18)

for i, folder in enumerate(dirs):
    # # loads
    axes[i][0].set_ylabel(i+1, rotation=0, fontsize=20,
                          labelpad=20, color='black')
    data_fk = nib.load(path_to_orig + folder +
                       "/queried_tumor_patientspace_mri_s.nii")
    data_fk = data_fk.get_fdata()
    print(data_fk.shape)

    flair_seg = nib.load(path_to_orig + folder +
                         "/Tum_Combo.nii.gz")
    flair_seg = flair_seg.get_fdata()
    t1c_seg = nib.load(path_to_orig + folder +
                       "/Tum_T1_binarized.nii.gz")
    t1c_seg = t1c_seg.get_fdata()
    data_segm = flair_seg + t1c_seg
    flair_scan_dep = data_segm > 0

    flair_scan_dep = flair_seg
    t1gd_scan_dep = t1c_seg

    data_brain = nib.load(path_to_orig + folder + "/t1c.nii.gz")
    data_brain = data_brain.get_fdata()
    data_brain = (data_brain - np.min(data_brain)) / \
        (np.max(data_brain) - np.min(data_brain))
    print(data_brain.shape)

    # process
    mri_scan_dep = flair_scan_dep
    mask = np.ma.masked_where((t1gd_scan_dep.astype(int) + flair_scan_dep.astype(
        int)) == 0, t1gd_scan_dep.astype(int)+flair_scan_dep.astype(int))
    fk = np.ma.masked_where(data_fk <= 0.02, data_fk)

    meanvalues = np.mean(mri_scan_dep, axis=(0, 1))
    s = np.argmax(meanvalues)
    showparameters = 0

    axes[i][0].imshow(data_brain[:, :, s].T[15:215, 20:220], cmap=cm.gray)
    remove_ticks(axes[i][0])
    # print(data_brain[:,:,s].T.shape)

    axes[i][1].imshow(data_brain[:, :, s].T[10:220, 15:225], cmap=cm.gray)
    remove_ticks(axes[i][1])
    axes[i][1].imshow(mask[:, :, s].T[10:220, 15:225],
                      cmap=cm.bwr)  # tab20, bwr?

    axes[i][2].imshow(data_brain[:, :, s].T[10:220, 15:225], cmap=cm.gray)
    im2 = axes[i][2].imshow(fk[:, :, s].T[10:220, 15:225], cmap=cm.jet)
    remove_ticks(axes[i][2])

    if i == 2:
        cb = fig.colorbar(
            im2, cax=axes[i][3], ticks=list(np.linspace(0, 1, 11)))
        cb.ax.tick_params(labelsize=14)
    else:
        axes[i][3].axis('off')
    # fig.savefig("./plot_{}.png".format(ii), bbox_inches='tight', dpi=600)
plt.show()
# if ii == 2:
#     break
