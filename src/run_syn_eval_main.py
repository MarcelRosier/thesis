from constants import (ENV, SYN_TUMOR_PATH_TEMPLATE)
from syn_eval import baseline, query, helper, real_query_dc, test

SYN_TUMOR_PATH_TEMPLATE = SYN_TUMOR_PATH_TEMPLATE[ENV]

STARTING_ID = 75000
STARTING_ID_VAL = 80000
STARTING_LARGE_ID = 90000


def run_baseline(downsample_to: int = None):
    for i in range(62):
        cur_id = STARTING_ID + i
        baseline.run(input_tumor_id=cur_id, downsample_to=downsample_to)


def run_baseline_validation(downsample_to: int = None):
    for i in range(62):
        cur_id = STARTING_ID_VAL + i
        baseline.run(input_tumor_id=cur_id,
                     downsample_to=downsample_to, validation=True)


def run_large_syn_eval(downsample_to: int = None):
    for i in range(1000):
        cur_id = STARTING_LARGE_ID + i
        baseline.run(input_tumor_id=cur_id,
                     downsample_to=downsample_to)


# run_large_syn_eval(downsample_to=32)
# real_query_dc.run(processes=32)
# helper.run(val=True)
# run_baseline_validation(downsample_to=None)
# run_baseline(downsample_to=64)
# run_baseline(downsample_to=32)
# test.bm_presence()
# query.run(40)
helper.run_1k()
