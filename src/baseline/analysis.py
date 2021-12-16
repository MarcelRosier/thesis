import os

from constants import ENV, REAL_TUMOR_BASE_PATH
from utils import find_n_best_score_ids, DSValueType
import pandas as pd

REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]


def compare_best_match_for_downsampling(downsample_to=64, value_type=DSValueType.FLAIR, n_best=3):
    tumor_ids = os.listdir(REAL_TUMOR_BASE_PATH)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    # tumor_ids = tumor_ids[:10]
    base_path = "/home/marcel/Projects/uni/thesis/src/baseline/data/testset_size_2000"
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
