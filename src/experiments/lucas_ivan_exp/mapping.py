import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import torch.nn.functional as F
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


def test_load_real_tumor(base_path: str, downsample_to: int = None):
    """
    @base_path: path to the real tumor folder, e.g. /tgm001_preop/ \n
    Return pair (t1c,flair) of a real tumor
    """
    import torch
    t1c = nib.load(os.path.join(
        base_path, 'tumor_mask_t_to_atlas.nii')).get_fdata()
    flair = nib.load(os.path.join(
        base_path, 'tumor_mask_f_to_atlas.nii')).get_fdata()

    flair = torch.from_numpy(flair)
    t1c = torch.from_numpy(t1c)
    return (t1c, flair)


def print_details():
    import utils
    tumor_id = 'tgm008_preop'
    syn_id = 42685

    # _, real_f = utils.load_real_tumor(
    #     base_path=f'/Users/marcelrosier/Projects/uni/tumor_data/real_tumors/{tumor_id}')
    real_f = nib.load(os.path.join(
        f'/Users/marcelrosier/Projects/uni/tumor_data/real_tumors/{tumor_id}', 'tumor_mask_f_to_atlas.nii')).get_fdata()
    syn_f = utils.load_single_tumor(syn_id, threshold=0.2, cut=False)
    syn_f = np.delete(np.delete(
        np.delete(syn_f, 128, 0), 128, 1), 128, 2)
    real_f = np.delete(np.delete(
        np.delete(real_f, 128, 0), 128, 1), 128, 2)
    print(real_f.shape)
    print(syn_f.shape)
    # my dice implementation
    print("my dice: ", utils.calc_dice_coef(real_f, syn_f))
    # monai dice loss
    print("monai dice: ", monai_dice_score(real_f, syn_f))
