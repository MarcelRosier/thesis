import utils
import json

selected_tumors = [
    'tgm047_preop',  # 1.4
    'tgm008_preop',  # 1.2
    'tgm051_preop',    # 1.0
    'tgm025_preop',  # 0.8
    'tgm023_preop',  # 0.6
]

baseline_ranking_path = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice/"
encoded_ranking_base_path = "/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_enc_sim/"

# for i, tumor_id in enumerate(selected_tumors):
#     # get best_match data
#     path = f"{baseline_ranking_path}/{tumor_id}.json"
#     gt_ranking = utils.find_n_best_score_ids(
#         path=path, value_type='combined', order_func=max, n_best=100)

#     ae_best_match_id = utils.find_n_best_score_ids(
#         path=encoded_ranking_base_path + "ae/" + tumor_id + ".json", value_type='combined', order_func=min, n_best=1)[0]
#     vae_best_match_id = utils.find_n_best_score_ids(
#         path=encoded_ranking_base_path + "vae/" + tumor_id + ".json", value_type='combined', order_func=min, n_best=1)[0]
#     with open(path) as file:
#         data = json.load(file)
#     dice_gt = data[gt_ranking[0]]['combined']
#     dice_ae = data[ae_best_match_id]['combined']
#     dice_vae = data[vae_best_match_id]['combined']

#     print(tumor_id)
#     print(gt_ranking.index(ae_best_match_id))
#     print(gt_ranking.index(vae_best_match_id))
#     print(f"{dice_gt=}")
#     print(f"{dice_ae=}")
#     print(f"{dice_vae=}")
#     print(f"delta_ae= {dice_gt-dice_ae}")
#     print(f"delta_vae= {dice_gt-dice_vae}")
#     print("---")
base_path = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000"

path_downsampled = f"{base_path}/dim_{64}/dice/{'tgm023_preop'}.json"
top_downsampled = utils.find_n_best_score_ids(
    path_downsampled,
    utils.DSValueType.COMBINED,
    max,
    n_best=5
)[0]

path_gt = f"{base_path}/dim_{128}/dice/{'tgm023_preop'}.json"
top_gt = utils.find_n_best_score_ids(
    path_gt,
    utils.DSValueType.COMBINED,
    max,
    n_best=5
)
print(top_gt.index(top_downsampled))
path = f"{base_path}/dim_{128}/dice/{'tgm023_preop'}.json"
with open(path) as file:
    data = json.load(file)

dice_gt = data[top_gt[0]]['combined']
dice_down = data[top_downsampled]['combined']
print(f"{dice_gt=}")
print(f"{dice_down=}")
print(f"delta = {dice_gt - dice_down}")
