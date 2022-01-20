from baseline import analysis
from autoencoder import recon_dice_analysis
from utils import DSValueType
# analysis.compare_best_match_for_enc(value_type=DSValueType.FLAIR)
# analysis.compare_best_match_for_enc(value_type=DSValueType.T1C)
recon_dice_analysis.compute_recon_dice_scores(is_t1c=False, cuda_id=6)
