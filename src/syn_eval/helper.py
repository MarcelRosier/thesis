from re import T
import utils
import json
import os
from constants import REAL_TUMOR_BASE_PATH, ENV
REAL_TUMOR_BASE_PATH = REAL_TUMOR_BASE_PATH[ENV]

base_path = "/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data"


def run(val: bool = False):

    ids = [str((80000 if val else 75000) + i) for i in range(62)]

    baseline = {}
    down64 = {}
    down32 = {}
    ae = {}
    vae = {}
    for tid in ids:
        baseline[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/../val_data/baseline/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        # down64[tid] = utils.find_n_best_score_ids(
        #     path=f"{base_path}/down64/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        # down32[tid] = utils.find_n_best_score_ids(
        #     path=f"{base_path}/down32/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        ae[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/ae1024/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=min, n_best=15)
        vae[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/vae1024/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=min, n_best=15)

    sub_path = f"{base_path}/top_15_lists{'_val' if val else ''}"
    with open(f"{sub_path}/baseline.json", 'w') as file:
        json.dump(baseline, file)
    # with open(f"{base_path}/top_15_lists/down64.json", 'w') as file:
    #     json.dump(down64, file)
    # with open(f"{base_path}/top_15_lists/down32.json", 'w') as file:
    #     json.dump(down32, file)
    with open(f"{sub_path}/ae1024.json", 'w') as file:
        json.dump(ae, file)
    with open(f"{sub_path}/vae1024.json", 'w') as file:
        json.dump(vae, file)


def run_1k():

    ids = [str((90000) + i) for i in range(1000)]
    base_path = "/home/ivan_marcel/thesis/src/syn_eval/data"

    baseline = {}
    down64 = {}
    down32 = {}
    ae = {}
    vae = {}
    for tid in ids:
        baseline[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/baseline/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        down64[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/down64/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        down32[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/down32/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        ae[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/ae1024/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=min, n_best=15)
        vae[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/vae1024/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=min, n_best=15)

    sub_path = f"{base_path}/top_15_lists_1k"
    with open(f"{sub_path}/baseline.json", 'w') as file:
        json.dump(baseline, file)
    with open(f"{base_path}/top_15_lists/down64.json", 'w') as file:
        json.dump(down64, file)
    with open(f"{base_path}/top_15_lists/down32.json", 'w') as file:
        json.dump(down32, file)
    with open(f"{sub_path}/ae1024.json", 'w') as file:
        json.dump(ae, file)
    with open(f"{sub_path}/vae1024.json", 'w') as file:
        json.dump(vae, file)


def run_real_dc_lists():

    tumor_ids = os.listdir(REAL_TUMOR_BASE_PATH)
    tumor_ids = [f for f in tumor_ids if f[3:6].isnumeric()]
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    print(tumor_ids)
    baseline = {}
    down64 = {}
    down32 = {}
    ae = {}
    vae = {}
    base_path = "/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/real_dc_data"
    for tid in tumor_ids:
        baseline[tid] = utils.find_n_best_score_ids(
            path=f"/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        # down64[tid] = utils.find_n_best_score_ids(
        #     path=f"{base_path}/down64/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        # down32[tid] = utils.find_n_best_score_ids(
        #     path=f"{base_path}/down32/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=max, n_best=15)
        ae[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/ae1024/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=min, n_best=15)
        vae[tid] = utils.find_n_best_score_ids(
            path=f"{base_path}/vae1024/{tid}.json", value_type=utils.DSValueType.COMBINED, order_func=min, n_best=15)

    with open(f"{base_path}/baseline.json", 'w') as file:
        json.dump(baseline, file)
    # with open(f"{base_path}/top_15_lists/down64.json", 'w') as file:
    #     json.dump(down64, file)
    # with open(f"{base_path}/top_15_lists/down32.json", 'w') as file:
    #     json.dump(down32, file)
    with open(f"{base_path}/ae1024.json", 'w') as file:
        json.dump(ae, file)
    with open(f"{base_path}/vae1024.json", 'w') as file:
        json.dump(vae, file)


def check():
    base_path = "/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data"

    with open(f"{base_path}/top_15_lists/baseline.json") as file:
        data = json.load(file)
        print(data['75000'])
    with open(f"{base_path}/top_15_lists/down64.json") as file:
        data = json.load(file)
        print(data['75000'])
    with open(f"{base_path}/top_15_lists/down32.json") as file:
        data = json.load(file)
        print(data['75000'])
    with open(f"{base_path}/top_15_lists/ae1024.json") as file:
        data = json.load(file)
        print(data['75000'])
    with open(f"{base_path}/top_15_lists/vae1025.json") as file:
        data = json.load(file)
        print(data['75000'])


def test_enc_bm_quality():
    with open(f"{base_path}/top_15_lists/ae1024.json") as file:
        data = json.load(file)
        print(data['75000'])
