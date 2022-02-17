from constants import (ENV, SYN_TUMOR_PATH_TEMPLATE)
from syn_eval import baseline, test, query

SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]

STARTING_ID = 75000
STARTING_ID_VAL = 80000


def run_baseline(downsample_to: int = None):
    for i in range(62):
        cur_id = STARTING_ID + i
        baseline.run(input_tumor_id=cur_id, downsample_to=downsample_to)


def run_baseline_validation(downsample_to: int = None):
    for i in range(62):
        cur_id = STARTING_ID_VAL + i
        baseline.run(input_tumor_id=cur_id,
                     downsample_to=downsample_to, validation=True)


query.run(processes=32)
# test.check_real_ae_bm_quality()
# run_baseline_validation(downsample_to=None)
# run_baseline(downsample_to=64)
# run_baseline(downsample_to=32)
# test.bm_presence()
