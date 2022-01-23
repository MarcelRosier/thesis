import json
import os

from constants import ENV, REAL_TUMOR_BASE_PATH
from utils import find_n_best_score_ids, DSValueType
import pandas as pd

REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]


def compare_best_match_for_downsampling(downsample_to: int = 64, value_type=DSValueType.FLAIR, n_best=3):
    tumor_ids = os.listdir(REAL_TUMOR_BASE_PATH)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    # tumor_ids = tumor_ids[:10]
    base_path = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/testset_size_50000"
    top_gt_list = []
    top_downsampled_list = []
    for tumor_id in tumor_ids:
        path_downsampled = f"{base_path}/dim_{downsample_to}/dice/{tumor_id}.json"
        top_downsampled = find_n_best_score_ids(
            path_downsampled,
            value_type,
            max,
            n_best
        )

        path_gt = f"{base_path}/dim_{128}/dice/{tumor_id}.json"
        top_gt = find_n_best_score_ids(
            path_gt,
            value_type,
            max,
            n_best
        )
        # same_best_match = top_gt[0] == top_downsampled[0]
        top_gt_list.append(top_gt)
        top_downsampled_list.append(top_downsampled)
    return tumor_ids, top_gt_list, top_downsampled_list


def compare_best_match_for_enc(is_ae=True, value_type: DSValueType = DSValueType.COMBINED):
    tumor_ids = os.listdir(REAL_TUMOR_BASE_PATH)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    # tumor_ids = tumor_ids[:10]
    assert len(tumor_ids) == 62
    base_path_gt = "/home/ivan_marcel/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice"
    base_path_enc = f"/home/ivan_marcel/thesis/src/autoencoder/data/final_50k_enc_sim/{'ae' if is_ae else 'vae'}"

    data = {}

    for tumor_id in tumor_ids:
        path_enc = f"{base_path_enc}/{tumor_id}.json"
        top_enc = find_n_best_score_ids(
            path_enc,
            value_type,
            min,
            n_best=15
        )

        path_gt = f"{base_path_gt}/{tumor_id}.json"
        top_gt = find_n_best_score_ids(
            path_gt,
            value_type,
            max,
            n_best=15
        )
        # same_best_match = top_gt[0] == top_downsampled[0]
        data[tumor_id] = {
            'top_gt': top_gt,
            'top_enc': top_enc
        }

    path = f"/home/ivan_marcel/thesis/src/autoencoder/data/final_50k_top15/{'ae' if is_ae else 'vae'}/top_15_{value_type}.json"
    with open(path, "w") as file:
        json.dump(data, file)
