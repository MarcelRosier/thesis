import time
from datetime import datetime
import numpy as np
import json


def time_measure(log=False):
    def timing_base(f):
        def wrap(*args, **kwargs):
            start = time.time()
            ret = f(*args, **kwargs)
            end = time.time()
            output = '{:s} function took {:.3f} ms -> {:.0f} h {:.0f} m {:.0f} s'.format(
                f.__name__, (end-start)*1000.0, (end-start) // 60 * 60, ((end-start) / 60) % 60, (end-start) % 60)
            print(output)
            if log:
                with open("run_time.log", "a+") as file:
                    file.write('started: {}\nend: {} \n{}\n{}\n'.format(
                        datetime.fromtimestamp(start), datetime.fromtimestamp(end), output, ("-"*10)))
            return ret
        return wrap
    return timing_base


def calc_dice_coef(syn_data, real_data):
    """calcualte the dice coefficient of the two input data"""
    combined = syn_data + real_data
    intersection = np.count_nonzero(combined == 2)
    union = np.count_nonzero(syn_data) + np.count_nonzero(real_data)
    if union == 0:
        return 0
    return (2 * intersection) / union


def get_number_of_entries(path):
    data = {}
    with open(path) as json_file:
        data = json.load(json_file)
    print("data_len: ", len(data))
    return len(data)
