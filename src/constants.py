
import socket

# env var
ENV = socket.getfqdn()
# envs
LOCAL = 'lx'
IBBM = 'atnavab66.informatik.tu-muenchen.de'
GIGA = 'ibbm-giga'

DICE_SCORE_DATADUMP_PATH_TEMPLATE = {
    LOCAL: '/home/marcel/Projects/uni/thesis/src/data/{id}_datadump.json',
    IBBM: '~/thesis/src/data/{id}_datadump.json',
    GIGA: None
}

# Baseline constants
REAL_TUMOR_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/real_tumors/tgm001_preop',
    IBBM: '/home/rosierm/marcel_tgm/tgm001_preop',
    GIGA: None
}

SYN_TUMOR_BASE_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset',
    IBBM: '/home/rosierm/samples_extended/Dataset',
    GIGA: '/mnt/Drive3/ivan/samples_extended/Dataset'
}

SYN_TUMOR_PATH_TEMPLATE = {
    LOCAL: '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset/{id}/Data_0001.npz',
    IBBM: '/home/rosierm/samples_extended/Dataset/{id}/Data_0001.npz',
    GIGA: '/mnt/Drive3/ivan/samples_extended/Dataset/{id}/Data_0001.npz'
}

T1C_PATH = {
    LOCAL: None,
    IBBM: '/home/rosierm/kap_2021/dice_analysis/tumor_mask_t_to_atlas.nii',
    GIGA: None
}

FLAIR_PATH = {
    LOCAL: None,
    IBBM: '/home/rosierm/kap_2021/dice_analysis/tumor_mask_f_to_atlas.nii',
    GIGA: None
}

# Autoencoder constants
AE_CHECKPOINT_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/src/autoencoder/checkpoints',
    IBBM: '/home/rosierm/thesis/src/autoencoder/checkpoints',
    GIGA: '/mnt/Drive3/ivan_marcel/checkpoints'
}

# Dataset size constants
TUMOR_SUBSET_200 = 200
TUMOR_SUBSET_1K = 1000
TUMOR_SUBSET_10K = 10000
TUMOR_SUBSET_50K = 50000
