from email.mime import base
from constants import (ENV, SYN_TUMOR_PATH_TEMPLATE)
from syn_eval import baseline

SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]

STARTING_ID = 75000


def run_baseline(downsample_to: int = None):
    for i in range(62):
        cur_id = STARTING_ID + i
        baseline.run(input_tumor_id=cur_id, downsample_to=downsample_to)

# run_baseline(downsample_to=None)
# run_baseline(downsample_to=64)
# run_baseline(downsample_to=32)
