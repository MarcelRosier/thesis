import json

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

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


def plot_gt_enc_comp(enc: str, test_set_size: str):
    table = pd.read_csv(
        f"/home/marcel/Projects/uni/thesis/media/{enc}/gt_{enc}_comp_{test_set_size}.csv").to_numpy()
    # transform list strings to length
    for row in table:
        row[0] = int(row[0][3:6])
        row[1] = len(row[1].split(','))
        row[2] = len(row[2].split(','))
        row[3] = len(row[3].split(' '))
    df_table = pd.DataFrame(
        {'tumor': table[:, 0], 'intersection': table[:, 3]})
    df_table.astype({'tumor': 'int32'})
    df_table.sort_values(by='tumor')
    print(df_table)
    ax = sns.barplot(x='tumor', y='intersection', data=df_table)
    ax.set(xlabel='real tumor (tgmXXX_preop)',
           ylabel='#intersection in top 15', title=f'#Intersection between top 15 groundtruth and encoded L2 comparison for a test dataset_size= {test_set_size}, {enc}')
    plt.show()


def plot_gt_enc_rbo_scores():
    json_data = {}
    with open("/home/marcel/Projects/uni/thesis/media/rbo_comp_200_top_1.json") as file:
        json_data = json.load(file)
    lists = zip(json_data.keys(), json_data.values())
    df = pd.DataFrame(data=lists, columns=['tumor', 'rbo'])
    df['tumor'] = df['tumor'].apply(lambda c: int(c[3:6]))

    avg = mean(df['rbo'].to_list())
    print(avg)
    print(sum(df['rbo'].to_list()))

    ax = sns.barplot(x='tumor', y='rbo', data=df)
    ax.axhline(avg)
    ax.set(xlabel='real tumor (tgmXXX_preop)',
           ylabel='RBO of top 1', title='RBO score for top 1 groundtruth and encoded L2 comparison for a test dataset_size= 200')
    # plt.show()


def plot_enc4096_gt_best_matches(test_set_size: str, enc: str):
    json_data = {}
    with open(f"/home/marcel/Projects/uni/thesis/media/{enc}/{enc}_gt_match_pairs/testset_size_{test_set_size}.json") as file:
        json_data = json.load(file)
    tumors = []
    gt_indices = []
    for key in json_data.keys():
        tumors.append(key)
        gt_indices.append(json_data[key]['unencoded_rank'])
    print(f"{max(gt_indices)=}")
    print(f"{np.sum(np.array(gt_indices) == 0)=}")
    df = pd.DataFrame(columns=['tumor', 'gt_index'],
                      data=zip(tumors, gt_indices))
    df['tumor'] = df['tumor'].apply(lambda c: int(c[3:6]))
    ax = sns.barplot(x='tumor', y='gt_index', data=df)
    avg = (sum(gt_indices)/len(gt_indices))
    ax.axhline(avg)
    ax.text(0, avg + 0.05, str(avg)[:4])
    ax.set(
        title=f"Index in the groundtruth ranking of the best encoded match (#syntethic tumors={test_set_size}) , {enc}",)
    plt.show()


def plot_best_match_presence(enc: str, test_set_size: str, top_n: int, ax):
    from autoencoder.encoded_similarity_check import load_top_15_lists
    tumor_ids, gt_lists, encoded_lists = load_top_15_lists(
        csv_path=f"/home/marcel/Projects/uni/thesis/media/{enc}/gt_{enc}_comp_{test_set_size}.csv")
    # transform list strings to length
    is_present = []
    for tumor, gt_list, enc_list in zip(tumor_ids, gt_lists, encoded_lists):
        gt_best = gt_list[0]
        gt_best_in_top_n_enc = gt_best in enc_list[:top_n]
        is_present.append(float(gt_best_in_top_n_enc))

    tumor_ids = [int(tumor[3:6]) for tumor in tumor_ids]
    avg = sum(is_present) / len(is_present)
    sns.barplot(ax=ax, x=tumor_ids, y=is_present, color="#2a9c2c")
    ax.axhline(avg)
    ax.text(0, avg + 0.05, str(avg*100)[:4] + "%")
    ax.set_title(f"gt best match in top {top_n} encoded matches")
    plt.show()


def plot_best_match_presence_overview(enc, test_set_size):
    fig, axes = plt.subplots(3, 1, figsize=(15, 5), sharey=True)
    fig.suptitle(
        f'GT best match present in encoded top n ranking\n Datasetsize={test_set_size}, {enc} ')

    plot_best_match_presence(enc, test_set_size, top_n=15, ax=axes[0])
    plot_best_match_presence(enc, test_set_size, top_n=5, ax=axes[1])
    plot_best_match_presence(enc, test_set_size, top_n=1, ax=axes[2])

    plt.show()


# plot_best_match_presence_overview(enc="enc_2048_1500", test_set_size="2k")
# plot_gt_enc_comp(enc="enc_2048_1500", test_set_size="2k")
plot_enc4096_gt_best_matches(test_set_size="2k", enc="enc_2048_1500")
