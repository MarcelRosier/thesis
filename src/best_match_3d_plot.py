from matplotlib.image import imread
import torch.nn.functional as F
import nibabel as nib
import torch
from scipy.ndimage import zoom
from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pyparsing import opAssoc

import utils


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


def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def get_data(real_tumor_id, best_match_id):
    real_tumor_path = f"/Users/marcelrosier/Projects/uni/tumor_data/real_tumors/{real_tumor_id}"
    t1c_seg, flair_seg = utils.load_real_tumor(real_tumor_path)
    baseline_t1c, baseline_flair = load_syn(tid=best_match_id)
    return t1c_seg, flair_seg, baseline_t1c, baseline_flair


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

ncols = 5
fig, axes = plt.subplots(ncols=ncols, nrows=2)

selected_tumors = {
    'tgm001_preop': 39725,
    'tgm002_preop': 26858,
    'tgm003_preop': 49389,
    'tgm004_preop': 43626,
    'tgm005_preop': 36183,
}

for i, key in enumerate(selected_tumors.keys()):
    rt, rf, st, sf = get_data(
        real_tumor_id=key, best_match_id=selected_tumors[key])

    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    # color=(40/255, 247/255, 1))
    mlab.contour3d(data_brain, opacity=0.15, color=(1, 1, 1))
    mlab.contour3d(rf, opacity=0.2, color=(0, 0, 1))
    mlab.contour3d(rt, opacity=0.6, color=(1, 0, 0))
    mlab.savefig(filename="temp_r.png", figure=fig)
    mlab.close(fig)

    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    # color=(40/255, 247/255, 1))
    mlab.contour3d(data_brain, opacity=0.15, color=(1, 1, 1))
    mlab.contour3d(sf, opacity=0.2, color=(0, 0, 1))
    mlab.contour3d(st, opacity=0.6, color=(1, 0, 0))
    mlab.savefig(filename="temp_s.png", figure=fig)
    mlab.close(fig)

    r_img = imread("temp_r.png")
    s_img = imread("temp_s.png")
    axes[0][i].set_title(key)
    axes[0][i].imshow(r_img[50:750][100:700])
    remove_ticks(axes[0][i])
    axes[1][i].imshow(s_img[50:750][100:700])
    remove_ticks(axes[1][i])

    import os
    # os.remove("temp_r.png")
    # os.remove("temp_s.png")

axes[0][0].set_ylabel('Input')
axes[1][0].set_ylabel('Best match')

plt.show()
