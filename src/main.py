import json
from datetime import date, datetime

import utils
from baseline import baseline, baseline_parallel
from faiss_src import playground
from utils import DSValueType

# baseline.run()
# baseline_parallel.run()


def run_parallel_comparison(is_test=False):
    process_counts = [1, 2, 4, 8, 16, 32]

    results = {}
    for p_count in process_counts:
        start = datetime.now()
        maximum = baseline_parallel.run(processes=p_count, is_test=is_test)
        end = datetime.now()
        total_seconds = str((end-start).total_seconds())
        results[p_count] = {
            'runtime': total_seconds,
            'partner': maximum['partner']
        }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("data/{}_comparison.json".format(now), "w") as file:
        json.dump(results, file)


# run_parallel_comparison(is_test=True)
# baseline.run(is_test=True)
max_t1c = utils.find_n_max_dice_score_ids(
    path='/home/marcel/Projects/uni/thesis/src/data/2021-07-09 11:07:20_datadump.json',
    value_type=DSValueType.T1C,
    n_max=10)
print(max_t1c)

playground.run()
