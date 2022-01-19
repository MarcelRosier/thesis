import os
import json
from utils import DSValueType


def run():
    base_dir = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/testset_size_50000/dim_128/dice"
    tumor_ids = os.listdir(base_dir)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    mapping = {}
    for tumor_id in tumor_ids:
        with open(f"{base_dir}/{tumor_id}") as json_file:
            data = json.load(json_file)
            best_key = max(
                data.keys(), key=lambda k: data[k][DSValueType.COMBINED.value])
            mapping[tumor_id.split('.')[0]] = best_key
    with open('mapping.json', 'w') as file:
        json.dump(mapping, file)
