import matplotlib.pyplot as plt
import json
import numpy as np

DATA_PATH = '/home/marcel/Projects/uni/thesis/src/data/2021-06-26 04:47:47_comparison.json'


def load_json_data(path):
    data = {}
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def plot_runtime_vs_threads(data):
    """
    input data is a dict: {
        '$thread_number':{
            'runtime': $total_seconds,
            'partner': $partner_tumor_id # must be the same for all
        },
    }
    """
    x_data = [k for k in data.keys()]
    y_data = [float(v['runtime']) for _, v in data.items()]
    objects = np.arange(len(x_data))
    plt.bar(objects, y_data, align='center', alpha=0.5)
    plt.xticks(objects, x_data)
    plt.xlabel('Number of processes')
    plt.ylabel('Runtime in seconds')
    plt.title('Process count and corresponding runtime')

    plt.show()


data = load_json_data(DATA_PATH)
plot_runtime_vs_threads(data=data)
