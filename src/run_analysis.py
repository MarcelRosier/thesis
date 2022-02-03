from baseline import analysis
from autoencoder import recon_dice_analysis
from utils import DSValueType
analysis.compare_best_match_for_enc(
    value_type=DSValueType.COMBINED, is_ae=False, is_1024=True)
# analysis.compare_best_match_for_enc(value_type=DSValueType.T1C)
# recon_dice_analysis.compute_recon_dice_scores(is_t1c=True, cuda_id=6)
# recon_dice_analysis.compare_custom_monai_ranking()
# recon_dice_analysis.gen_recons(cuda_id=3)


###
# Space usage tests
###

# import json
# import multiprocessing
# import os
# from datetime import datetime
# from functools import partial
# from typing import Tuple
# from monai.metrics import compute_meandice
# from monai.losses.dice import DiceLoss
# import numpy as np
# import torch
# import utils
# from constants import (BASELINE_SIMILARITY_BASE_PATH, ENV, REAL_TUMOR_PATH,
#                        SYN_TUMOR_BASE_PATH, SYN_TUMOR_PATH_TEMPLATE)
# from scipy.ndimage import zoom
# from utils import (SimilarityMeasureType, load_real_tumor,
#                    time_measure)

# REAL_TUMOR_PATH = REAL_TUMOR_PATH[ENV]
# SYN_TUMOR_BASE_PATH = SYN_TUMOR_BASE_PATH[ENV]
# SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]
# BASELINE_SIMILARITY_BASE_PATH = BASELINE_SIMILARITY_BASE_PATH[ENV]


# def test_ds_size():

#     for i in range(100):
#         # load tumor data
#         # tumor = np.load(SYN_TUMOR_PATH_TEMPLATE.format(
#         #     id=i))['data']
#         tumor = np.load(
#             f"/mnt/Drive3/ivan_marcel/final_encs/encoded_VAE_FLAIR_8_1500/syn_50k/{i}.npy")  # encoded_FLAIR_1024_1500
#         # crop 129^3 to 128^3 if needed
#         # if tumor.shape != (128, 128, 128):
#         #     tumor = np.delete(np.delete(
#         #         np.delete(tumor, 128, 0), 128, 1), 128, 2)

#         # tumor = zoom(tumor, zoom=32/128, order=0)
#         # # normalize
#         # max_val = tumor.max()
#         # if max_val != 0:
#         #     tumor *= 1.0/max_val
#         path = "/home/ivan_marcel/thesis/src/data/test_compression_size/vae/"
#         np.savez_compressed(path+str(i), tumor)


# test_ds_size()
