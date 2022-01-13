import sys
from datetime import datetime
import os
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
mpl.rc('image', cmap='terrain')

# sys.path.append('/home/ivan/.local/lib/python3.8/site-packages')

#import tensorflow as tf
#import torch
#import surface_distance
# torch.set_num_threads(2) #uses max. 2 cpus for inference! no gpu inference!
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
#from masseffect_solver import masseffect_solver


def plot_tumors_mae(gt, sim, mriscan, i, mae, modelname):

    meanvalues = np.mean(mriscan, axis=(0, 1))
    print(mriscan.shape)
    print(meanvalues.shape)
    s = np.argmax(meanvalues)
    showparameters = 0

    for l in range(3):
        print(s)
        if l == 1:
            s -= 10
        elif l == 2:
            s += 20

        fig, (ax, ax2, ax3, cax2) = plt.subplots(ncols=4, figsize=(20, 4),
                                                 gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
        fig.subplots_adjust(wspace=0.2)
        im = ax.imshow(mriscan[:, :, s].T)
        ax.set_title("MRI segmentations")
        remove_ticks(ax)
        im2 = ax2.imshow(sim[:, :, s].T)
        ax2.set_title(
            "Simulated tumor with {} \n MAE = {}".format(modelname, mae))
        # if showparameters:
        #     ax2.set_title("Simulated tumor with\nDw={}, rho={}, T={}\n mu1={},mu2={}".format(
        #         round(Dw, 3), round(rho, 3), round(T), round(mu1, 3), round(mu2, 3)))
        remove_ticks(ax2)
        im3 = ax3.imshow(gt[:, :, s].T)
        ax3.set_title("Ground truth tumor")
        remove_ticks(ax3)
        fig.colorbar(im3, cax=cax2)
        #fig.savefig( "./plots_tumor/" + "{}_plot{}_{}.png".format(modelname, i,l))


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


mpl.rcParams['text.color'] = 'grey'
mpl.rcParams['xtick.color'] = 'grey'
mpl.rcParams['ytick.color'] = 'grey'
mpl.rcParams['axes.labelcolor'] = 'grey'
plt.rcParams["font.family"] = "Laksaman"

# path = "/home/ivan/ibbm/learnmorph/bene/"
path_to_orig = "/Users/marcelrosier/Projects/uni/tumor_data/real_tumors/"
ii = 1
for folder in os.listdir(path_to_orig):
    # print(folder)

    # # load
    # data_fk = nib.load(path + folder + "/inferred_tumor_patientspace_mri.nii")
    # data_fk = data_fk.get_fdata()
    # data_me = nib.load(
    #     path + folder + "/inferred_tumor_patientspace_m_mri_s.nii")
    # data_me = data_me.get_fdata()
    # "/brats_seg.nii.gz")
    data_segm = nib.load(path_to_orig + folder +
                         "/tumor_mask_f_to_atlas229.nii")
    data_segm = data_segm.get_fdata()
    flair_scan_dep = data_segm > 0
    t1gd_scan_dep = (data_segm == 1) | (data_segm == 4)
    # "/brats_t1c.nii.gz")
    data_brain = nib.load(path_to_orig + folder +
                          "/tumor_mask_t_to_atlas229.nii")
    data_brain = data_brain.get_fdata()
    data_brain = (data_brain - np.min(data_brain)) / \
        (np.max(data_brain) - np.min(data_brain))

    # process
    mri_scan_dep = flair_scan_dep
    mask = np.ma.masked_where((t1gd_scan_dep.astype(int) + flair_scan_dep.astype(
        int)) == 0, t1gd_scan_dep.astype(int)+flair_scan_dep.astype(int))
    # fk = np.ma.masked_where(data_fk <= 0.02, data_fk)
    # me = np.ma.masked_where(data_me <= 0.02, data_me)

    # plot
    modelname_indep = "FK"
    modelname_dep = "ME"
    meanvalues = np.mean(mri_scan_dep, axis=(0, 1))
    s = np.argmax(meanvalues)
    showparameters = 0

    fig, (ax, ax1, ax2, ax3, cax2) = plt.subplots(ncols=5, figsize=(15, 4),
                                                  gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.05]})
    fig.subplots_adjust(wspace=0.02)

    ax.imshow(data_brain[:, :, s].T[15:215, 20:220], cmap=cm.gray)
    ax.set_title("Patient's brain MRI".format(ii), fontsize=18)
    remove_ticks(ax)
    # print(data_brain[:,:,s].T.shape)

    ax1.imshow(data_brain[:, :, s].T[10:220, 15:225], cmap=cm.gray)
    ax1.set_title("MRI segmentations", fontsize=18)
    remove_ticks(ax1)
    ax1.imshow(mask[:, :, s].T[10:220, 15:225], cmap=cm.tab20)

    # ax2.imshow(data_brain[:, :, s].T[10:220, 15:225], cmap=cm.gray)
    # ax2.set_title("Inferred tumor \n with FK model", fontsize=18)
    # #ax2.set_title("Simulated tumor with {} \n MAE = {}".format(modelname, d[id]['maeresults'][:-1]))
    # im2 = ax2.imshow(fk[:, :, s].T[10:220, 15:225])
    # remove_ticks(ax2)

    # ax3.imshow(data_brain[:, :, s].T[10:220, 15:225], cmap=cm.gray)
    # ax3.set_title("Inferred tumor \n with ME model", fontsize=18)
    # ax3.imshow(me[:, :, s].T[10:220, 15:225])
    # remove_ticks(ax3)
    # cb = fig.colorbar(im2, cax=cax2)
    # cb.ax.tick_params(labelsize=16)
    # fig.savefig("./plot_{}.png".format(ii), bbox_inches='tight', dpi=600)
    plt.show()
    ii += 1
    if ii == 2:
        break
