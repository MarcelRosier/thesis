import json
import os
from tkinter import Y

from matplotlib import pyplot as plt

from constants import ENV, REAL_TUMOR_BASE_PATH
from utils import find_n_best_score_ids, DSValueType
import pandas as pd

REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]


def compare_best_match_for_downsampling(downsample_to: int = 64, value_type=DSValueType.FLAIR, n_best=3):
    tumor_ids = os.listdir(REAL_TUMOR_BASE_PATH)
    tumor_ids.remove('.DS_Store')
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    # tumor_ids = tumor_ids[:10]
    base_path = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000"
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
        # same_best_match = top_gt[0] == top_downsam‘pled[0]
        top_gt_list.append(top_gt)
        top_downsampled_list.append(top_downsampled)
    return tumor_ids, top_gt_list, top_downsampled_list


def compare_best_match_for_enc(is_ae=True, value_type: DSValueType = DSValueType.COMBINED, is_1024=False, is_20k=False):
    tumor_ids = os.listdir(REAL_TUMOR_BASE_PATH)
    tumor_ids = [t for t in tumor_ids if t[3:6].isnumeric()]
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    # tumor_ids = tumor_ids[:10]
    assert len(tumor_ids) == 62
    base_path_gt = "/home/ivan_marcel/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice"
    base_path_enc = f"/home/ivan_marcel/thesis/src/autoencoder/data/final_50k_enc_sim/{'ae' if is_ae else ('vae/1024' if is_1024 else ('vae20k/8' if is_20k else'vae/8)'))}"

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

    path = f"/home/ivan_marcel/thesis/src/autoencoder/data/final_50k_top15/{'ae' if is_ae else ('vae/1024' if is_1024 else ('vae20k/8' if is_20k else'vae/8)'))}/top_15_{value_type}.json"
    with open(path, "w") as file:
        json.dump(data, file)


def get_top_2_baseline_dice_scores():
    df = pd.DataFrame(columns=['tumor', 'first_combined', 'second_combined', ])
    base_path = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice"
    value_type = DSValueType.COMBINED
    tumor_list = os.listdir(base_path)  # [:10]
    tumor_list.sort(key=lambda f: int(f[3:6]))
    for tumor in tumor_list:
        path = f"{base_path}/{tumor}"
        [first, second] = find_n_best_score_ids(
            path=path, value_type=value_type, order_func=max, n_best=2)
        with open(path) as file:
            data = json.load(file)
        df = df.append({
            'tumor': tumor.split('.')[0],
            'first_combined': data[first][value_type.value],
            'second_combined': data[second][value_type.value],
        }, ignore_index=True)
    # print(first)
    # print(data[first])
    # print(second)
    # print(data[second])
    # df.to_csv("baseline_top_2_combined_dice_scores.csv")
    import seaborn as sns
    sns.set(rc={'figure.figsize': (16, 9)})
    sns.set_theme(style='whitegrid')
    ax = plt.subplots()
    first_plot = sns.scatterplot(data=df, x='tumor', y='first_combined')
    second_plot = sns.barplot(data=df, x='tumor', y='second_combined',
                              color='#329ea8', alpha=0.8)
    second_plot.set_xticklabels([])
    second_plot.set_ylabel("Combined Dice Score")
    second_plot.legend(['Best match', '2nd best match'])
    plt.show()
