import json
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


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

ncols = 4
fig, axes = plt.subplots(ncols=2, nrows=2)
selected_tumors = ['tgm008_preop',  # 1.4
                   'tgm008_preop',  # 1.2
                   ]
# selected_tumors = {
#     'tgm047_preop': 21359,
#     'tgm008_preop': 37987,
#     'tgm051_preop': 39571,
#     'tgm025_preop': 48179,
#     'tgm023_preop': 9115,
# }

baseline_ranking_path = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice/"
encoded_ranking_base_path = "/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_enc_sim/"
# TODO switch orientation
for i, tumor_id in enumerate(selected_tumors):
    # get best_match data
    path = f"{baseline_ranking_path}/{tumor_id}.json"
    best_match_id = utils.find_n_best_score_ids(
        path=path, value_type='combined', order_func=max, n_best=1)[0]
    best_match_id = 42685
    rt, rf, st, sf = get_data(
        real_tumor_id=tumor_id, best_match_id=best_match_id)

    # ae_best_match_id = utils.find_n_best_score_ids(
    #     path=encoded_ranking_base_path + "ae/" + tumor_id + ".json", value_type='combined', order_func=min, n_best=1)[0]
    # vae_best_match_id = utils.find_n_best_score_ids(
    #     path=encoded_ranking_base_path + "vae/" + tumor_id + ".json", value_type='combined', order_func=min, n_best=1)[0]
    # ae_t, ae_f = load_syn(tid=ae_best_match_id)
    # vae_t, vae_f = load_syn(tid=vae_best_match_id)

    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    # color=(40/255, 247/255, 1))
    mlab.contour3d(data_brain, opacity=0.15, color=(1, 1, 1))
    mlab.contour3d(rf, opacity=0.2, color=(0, 0, 1))
    # mlab.contour3d(rt, opacity=0.6, color=(1, 0, 0))
    mlab.savefig(filename="temp_r.png", figure=fig)
    mlab.close(fig)

    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    # color=(40/255, 247/255, 1))
    mlab.contour3d(data_brain, opacity=0.15, color=(1, 1, 1))
    mlab.contour3d(sf, opacity=0.2, color=(0, 0, 1))
    # mlab.contour3d(st, opacity=0.6, color=(1, 0, 0))
    mlab.savefig(filename="temp_s.png", figure=fig)
    mlab.close(fig)
    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    # # color=(40/255, 247/255, 1))
    # mlab.contour3d(data_brain, opacity=0.15, color=(1, 1, 1))
    # mlab.contour3d(ae_f, opacity=0.2, color=(0, 0, 1))
    # mlab.contour3d(ae_t, opacity=0.6, color=(1, 0, 0))
    # mlab.savefig(filename="temp_ae.png", figure=fig)
    # mlab.close(fig)
    # fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    # # color=(40/255, 247/255, 1))
    # mlab.contour3d(data_brain, opacity=0.15, color=(1, 1, 1))
    # mlab.contour3d(vae_f, opacity=0.2, color=(0, 0, 1))
    # mlab.contour3d(vae_t, opacity=0.6, color=(1, 0, 0))
    # mlab.savefig(filename="temp_vae.png", figure=fig)
    # mlab.close(fig)

    with open(path) as file:
        data = json.load(file)

    r_img = imread("temp_r.png")
    s_img = imread("temp_s.png")
    # ae_img = imread("temp_ae.png")
    # vae_img = imread("temp_vae.png")
    axes[i][0].imshow(r_img)  # [50:750][100:700])
    remove_ticks(axes[i][0])
    axes[i][0].set_ylabel(f"{i+1}", fontsize=32, rotation=0, labelpad=-10)
    axes[i][1].imshow(s_img)  # [50:750][100:700])
    remove_ticks(axes[i][1])
    # axes[i][2].imshow(ae_img[50:750][100:700])
    # remove_ticks(axes[i][2])
    # axes[i][3].imshow(vae_img[50:750][100:700])
    # remove_ticks(axes[i][3])
    # axes[1][i].set_xlabel(r"$Dice_{T1Gd}=$" + f"{str(data[best_match_id]['t1c'])[:5]}\n" +
    #                       r"$ Dice_{FLAIR}=$" + f"{str(data[best_match_id]['flair'])[:5]}", fontsize=28)

    import os
    # os.remove("temp_r.png")
    # os.remove("temp_s.png")

axes[0][0].set_title('tgm008', fontsize=32)
axes[0][1].set_title('42685', fontsize=32)
# axes[0][2].set_title('AE', fontsize=32)
# axes[0][3].set_title('VAE', fontsize=32)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()
