
import socket

# envs
LOCAL = 'lx'
IBBM = 'atnavab66.informatik.tu-muenchen.de'
GIGA = 'ibbm-giga'
MAC = 'marcel-mbp.local'

# env var
ENV = socket.getfqdn()  # MAC  #

DICE_SCORE_DATADUMP_PATH_TEMPLATE = {
    LOCAL: '/home/marcel/Projects/uni/thesis/src/data/{id}_datadump.json',
    IBBM: '~/thesis/src/data/{id}_datadump.json',
    GIGA: None
}

# Baseline constants
REAL_TUMOR_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/real_tumors/tgm001_preop',
    IBBM: '/home/rosierm/marcel_tgm/tgm001_preop',
    GIGA: '/mnt/Drive3/ivan_marcel/real_tumors/tgm001_preop',
    MAC: ''
}

REAL_TUMOR_BASE_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/real_tumors',
    IBBM: '/home/rosierm/marcel_tgm',
    GIGA: '/mnt/Drive3/ivan_marcel/real_tumors',
    MAC: '/Users/marcelrosier/Projects/uni/tumor_data/real_tumors'
}

SYN_TUMOR_BASE_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset',
    IBBM: '/home/rosierm/samples_extended/Dataset',
    GIGA: '/mnt/Drive3/ivan/samples_extended/Dataset',
    MAC: '/Users/marcelrosier/Projects/uni/samples_extended/Dataset'
}

SYN_TUMOR_PATH_TEMPLATE = {
    LOCAL: '/home/marcel/Projects/uni/thesis/tumor_data/samples_extended/Dataset/{id}/Data_0001.npz',
    IBBM: '/home/rosierm/samples_extended/Dataset/{id}/Data_0001.npz',
    GIGA: '/mnt/Drive3/ivan/samples_extended/Dataset/{id}/Data_0001.npz',
    MAC: '/Users/marcelrosier/Projects/uni/tumor_data/samples_extended/Dataset/{id}/Data_0001.npz',
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

BASELINE_SIMILARITY_BASE_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/src/baseline/data',
    IBBM: None,
    GIGA: '/home/ivan_marcel/thesis/src/baseline/data',
    MAC: '/Users/marcelrosier/Projects/uni/thesis/src/baseline/data',
}

# Autoencoder constants
AE_CHECKPOINT_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/src/autoencoder/checkpoints',
    IBBM: '/home/rosierm/thesis/src/autoencoder/checkpoints',
    GIGA: '/mnt/Drive3/ivan_marcel/checkpoints/tumor_autoencoder/torch_logs'
}

AE_MODEL_SAVE_PATH = {
    LOCAL: None,
    IBBM: None,
    GIGA: '/mnt/Drive3/ivan_marcel/models'
}

ENCODED_BASE_PATH = {
    LOCAL: None,
    IBBM: None,
    GIGA: '/mnt/Drive3/ivan_marcel/encoded',
    MAC: None,
}

MEDIA_BASE_PATH = {
    LOCAL: '/home/marcel/Projects/uni/thesis/media',
    MAC: '/Users/marcelrosier/Projects/uni/thesis/media'
}

default = {
    '200': {
        'START': 3000,
        'END': 3200
    },
    '2k': {
        'START': 4000,
        'END': 6000
    },
    '20k': {
        'START': 8000,
        'END': 28000
    },
    '50k': {
        'START': 0,
        'END': 50000
    },
}
TEST_SET_RANGES = {
    LOCAL: default,
    IBBM: default,
    GIGA: default,
    MAC: default,
}

# Dataset size constants
TUMOR_SUBSET_200 = 200
TUMOR_SUBSET_1K = 1000
TUMOR_SUBSET_10K = 10000
TUMOR_SUBSET_50K = 50000
