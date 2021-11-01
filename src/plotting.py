import json

import matplotlib.pyplot as plt
import numpy as np

DICE_DATA_PATH = '/home/marcel/Projects/uni/thesis/src/data/baseline_data/2021-09-30 19:47:08_comparison.json'
L2_DATA_PATH = '/home/marcel/Projects/uni/thesis/src/data/baseline_data/2021-10-06 22:33:30_comparison_l2.json'


def load_json_data(path):
    data = {}
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def plot_runtime_vs_threads_single_input(data):
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
    plt.title('Process count and corresponding runtime for 50k synthetic tumors')

    plt.show()


def plot_runtime_vs_threads_dual_input(data_1, data_2):
    """
    input data is a dict: {
        '$thread_number':{
            'runtime': $total_seconds,
            'partner': $partner_tumor_id # must be the same for all
        },
    }
    """
    x_data_1 = [k for k in data_1.keys()]
    y_data_1 = [float(v['runtime']) for _, v in data_1.items()]
    # x_data_2 = [k for k in data_2.keys()]
    y_data_2 = [float(v['runtime']) for _, v in data_2.items()]
    objects = np.arange(len(x_data_1))
    # add legend
    colors = {'DICE': 'royalblue', 'L2': 'indigo'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label])
               for label in labels]
    plt.legend(handles, labels)

    plt.bar(objects, y_data_1, align='center',
            alpha=0.5, width=0.3, color='royalblue')
    plt.bar(objects + 0.3, y_data_2, align='center',
            alpha=0.5, width=0.3, color='indigo')
    plt.xticks(objects, x_data_1)
    plt.xlabel('Number of processes')
    plt.ylabel('Runtime in seconds')
    plt.title('Process count and corresponding runtime for 50k synthetic tumors')

    plt.show()


dice_data = load_json_data(DICE_DATA_PATH)
l2_data = load_json_data(L2_DATA_PATH)
# plot_runtime_vs_threads_single_input(data=dice_data)
plot_runtime_vs_threads_dual_input(data_1=dice_data, data_2=l2_data)
