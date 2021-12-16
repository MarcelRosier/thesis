import os

from constants import ENV, REAL_TUMOR_BASE_PATH
from utils import find_n_best_score_ids, DSValueType
import pandas as pd

REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]


def compare_best_match_for_downsampling(downsample_to=64, value_type=DSValueType.FLAIR, n_best=3):
    tumor_ids = os.listdir(REAL_TUMOR_BASE_PATH)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    # tumor_ids = tumor_ids[:10]
    base_path = "/home/ivan_marcel/thesis/src/baseline/data/testset_size_2000"
    table = []
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
        same_best_match = top_gt[0] == top_downsampled[0]
        table.append([same_best_match, top_gt, top_downsampled])
    df = pd.DataFrame(
        table, columns=['Same best match', 'top 3 128^3', f'top 3 {downsample_to}^3'])
    print(df)
