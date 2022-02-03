import os
import json
from utils import DSValueType

base_dir = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice"


def run():
    print_details()
    return
    tumor_ids = os.listdir(base_dir)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    mapping = {}
    for tumor_id in tumor_ids:
        with open(f"{base_dir}/{tumor_id}") as json_file:
            data = json.load(json_file)
            best_key = max(
                data.keys(), key=lambda k: data[k][DSValueType.FLAIR.value])
            mapping[tumor_id.split('.')[0]] = best_key
    with open('mapping_flair.json', 'w') as file:
        json.dump(mapping, file)


def monai_dice_score(real, syn):
    from monai.losses.dice import DiceLoss
    import torch
    criterion = DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=False)
    real = torch.from_numpy(real)
    real.unsqueeze_(0)
    real.unsqueeze_(0)
    syn = torch.from_numpy(syn)
    syn.unsqueeze_(0)
    syn.unsqueeze_(0)
    return 1 - criterion(real, syn).item()


def print_details():
    import utils
    tumor_id = 'tgm008_preop'
    syn_id = 42685

    _, real_f = utils.load_real_tumor(
        base_path=f'/Users/marcelrosier/Projects/uni/tumor_data/real_tumors/{tumor_id}')
    syn_f = utils.load_single_tumor(syn_id, threshold=0.2)
    # my dice implementation
    print("my dice: ", utils.calc_dice_coef(real_f, syn_f))
    # monai dice loss
    print("monai dice: ", monai_dice_score(real_f, syn_f))
