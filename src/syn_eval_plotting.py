from email.mime import base
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors

from utils import DSValueType


def plot_bmp_overview():

    def plot_enc_best_match_presence(tumor_ids, top_gt_list, top_enc_list, top_n: int, ax, value_type: DSValueType):

        is_present = []
        for gt, enc in zip(top_gt_list, top_enc_list):
            enc_best = enc[0]
            enc_best_in_top_n_gt = enc_best in gt[:top_n]
            if top_n == 15 and not enc_best_in_top_n_gt:
                # print("test")
                pass
            is_present.append(float(enc_best_in_top_n_gt))
        tumor_ids = [int(tumor[3:6]) for tumor in tumor_ids]
        x_ax = np.linspace(1, len(tumor_ids), 62)
        cmap = colors.ListedColormap(['red', 'green'])
        assignedColors = [cmap(int(t)) for t in is_present]
        plot = sns.scatterplot(ax=ax, x=x_ax, y=is_present,
                               c=assignedColors,  cmap=cmap)
        plot.set_yticks([1.0, 0.0], ["True",
                                     "False"])
        plot.set_xticklabels([])
        plot.set_xlabel("Tumors")
        avg = sum(is_present) / len(is_present)
        # print(cmap(1))
        legend = plot.legend([str(avg*100)[:4] + "%"], loc="center right")
        legend.legendHandles[0].set_color('green')

    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 3))
    cols = ["DS64", "DS32", "AE", "VAE"]
    rows = ["Top 1", "Top 5", "Top 15"]
    print(axes[:, 0])
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0)
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    # Combined 8
    value_type = DSValueType.COMBINED
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists/baseline.json') as file:
        baseline: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists/down64.json') as file:
        down64: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists/down32.json') as file:
        down32: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists/ae1024.json') as file:
        ae1024: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists/vae1024.json') as file:
        vae1024: dict = json.load(file)

    tumor_ids = baseline.keys()
    top_baseline_lists = []
    top_ds64_lists = []
    top_ds32_lists = []
    top_ae_lists = []
    top_vae_lists = []
    for tumor_id in tumor_ids:
        top_baseline_lists.append(baseline[tumor_id])
        top_ds64_lists.append(down64[tumor_id])
        top_ds32_lists.append(down32[tumor_id])
        top_ae_lists.append(ae1024[tumor_id])
        top_vae_lists.append(vae1024[tumor_id])

    # down64
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ds64_lists, top_n=1, ax=axes[0][0], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ds64_lists, top_n=5, ax=axes[1][0], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ds64_lists, top_n=15, ax=axes[2][0], value_type=value_type)
    # down32
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ds32_lists, top_n=1, ax=axes[0][1], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ds32_lists, top_n=5, ax=axes[1][1], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ds32_lists, top_n=15, ax=axes[2][1], value_type=value_type)
    # ae
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=1, ax=axes[0][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=5, ax=axes[1][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=15, ax=axes[2][2], value_type=value_type)
    # vae
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=1, ax=axes[0][3], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=5, ax=axes[1][3], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=15, ax=axes[2][3], value_type=value_type)

    fig.tight_layout()


def plot_bmp_val_overview():

    def plot_enc_best_match_presence(tumor_ids, top_gt_list, top_enc_list, top_n: int, ax, value_type: DSValueType):

        is_present = []
        for gt, enc in zip(top_gt_list, top_enc_list):
            enc_best = enc[0]
            enc_best_in_top_n_gt = enc_best in gt[:top_n]
            if top_n == 15 and not enc_best_in_top_n_gt:
                # print("test")
                pass
            is_present.append(float(enc_best_in_top_n_gt))
        tumor_ids = [int(tumor[3:6]) for tumor in tumor_ids]
        x_ax = np.linspace(1, len(tumor_ids), 62)
        cmap = colors.ListedColormap(['red', 'green'])
        assignedColors = [cmap(int(t)) for t in is_present]
        plot = sns.scatterplot(ax=ax, x=x_ax, y=is_present,
                               c=assignedColors,  cmap=cmap)
        plot.set_yticks([1.0, 0.0], ["True",
                                     "False"])
        plot.set_xticklabels([])
        plot.set_xlabel("Tumors")
        avg = sum(is_present) / len(is_present)
        # print(cmap(1))
        legend = plot.legend([str(avg*100)[:4] + "%"], loc="center right")
        legend.legendHandles[0].set_color('green')

    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 3))
    cols = ["DS64", "DS32", "AE", "VAE"]
    rows = ["Top 1", "Top 5", "Top 15"]
    print(axes[:, 0])
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0)
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    # Combined 8
    value_type = DSValueType.COMBINED
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists_val/baseline.json') as file:
        baseline: dict = json.load(file)
    # with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists_val/down64.json') as file:
    #     down64: dict = json.load(file)
    # with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists_val/down32.json') as file:
    #     down32: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists_val/ae1024.json') as file:
        ae1024: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/data/top_15_lists_val/vae1024.json') as file:
        vae1024: dict = json.load(file)

    tumor_ids = baseline.keys()
    top_baseline_lists = []
    top_ds64_lists = []
    top_ds32_lists = []
    top_ae_lists = []
    top_vae_lists = []
    for tumor_id in tumor_ids:
        top_baseline_lists.append(baseline[tumor_id])
        # top_ds64_lists.append(down64[tumor_id])
        # top_ds32_lists.append(down32[tumor_id])
        top_ae_lists.append(ae1024[tumor_id])
        top_vae_lists.append(vae1024[tumor_id])

    # down64
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds64_lists, top_n=1, ax=axes[0][0], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds64_lists, top_n=5, ax=axes[1][0], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds64_lists, top_n=15, ax=axes[2][0], value_type=value_type)
    # # down32
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds32_lists, top_n=1, ax=axes[0][1], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds32_lists, top_n=5, ax=axes[1][1], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds32_lists, top_n=15, ax=axes[2][1], value_type=value_type)
    # ae
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=1, ax=axes[0][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=5, ax=axes[1][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=15, ax=axes[2][2], value_type=value_type)
    # vae
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=1, ax=axes[0][3], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=5, ax=axes[1][3], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=15, ax=axes[2][3], value_type=value_type)

    fig.tight_layout()


def plot_bmp_realdc_overview():

    def plot_enc_best_match_presence(tumor_ids, top_gt_list, top_enc_list, top_n: int, ax, value_type: DSValueType):

        is_present = []
        for gt, enc in zip(top_gt_list, top_enc_list):
            enc_best = enc[0]
            enc_best_in_top_n_gt = enc_best in gt[:top_n]
            if top_n == 15 and not enc_best_in_top_n_gt:
                # print("test")
                pass
            is_present.append(float(enc_best_in_top_n_gt))
        tumor_ids = [int(tumor[3:6]) for tumor in tumor_ids]
        x_ax = np.linspace(1, len(tumor_ids), 62)
        cmap = colors.ListedColormap(['red', 'green'])
        assignedColors = [cmap(int(t)) for t in is_present]
        plot = sns.scatterplot(ax=ax, x=x_ax, y=is_present,
                               c=assignedColors,  cmap=cmap)
        plot.set_yticks([1.0, 0.0], ["True",
                                     "False"])
        plot.set_xticklabels([])
        plot.set_xlabel("Tumors")
        avg = sum(is_present) / len(is_present)
        # print(cmap(1))
        legend = plot.legend([str(avg*100)[:4] + "%"], loc="center right")
        legend.legendHandles[0].set_color('green')

    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(12, 3))
    cols = ["DS64", "DS32", "AE", "VAE"]
    rows = ["Top 1", "Top 5", "Top 15"]
    print(axes[:, 0])
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0)
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    # Combined 8
    value_type = DSValueType.COMBINED
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/real_dc_data/baseline.json') as file:
        baseline: dict = json.load(file)
    # with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/real_dc_data/down64.json') as file:
    #     down64: dict = json.load(file)
    # with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/real_dc_data/down32.json') as file:
    #     down32: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/real_dc_data/ae1024.json') as file:
        ae1024: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/syn_eval/real_dc_data/vae1024.json') as file:
        vae1024: dict = json.load(file)

    tumor_ids = baseline.keys()
    top_baseline_lists = []
    top_ds64_lists = []
    top_ds32_lists = []
    top_ae_lists = []
    top_vae_lists = []
    for tumor_id in tumor_ids:
        top_baseline_lists.append(baseline[tumor_id])
        # top_ds64_lists.append(down64[tumor_id])
        # top_ds32_lists.append(down32[tumor_id])
        top_ae_lists.append(ae1024[tumor_id])
        top_vae_lists.append(vae1024[tumor_id])

    # down64
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds64_lists, top_n=1, ax=axes[0][0], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds64_lists, top_n=5, ax=axes[1][0], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds64_lists, top_n=15, ax=axes[2][0], value_type=value_type)
    # # down32
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds32_lists, top_n=1, ax=axes[0][1], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds32_lists, top_n=5, ax=axes[1][1], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
    #                              top_ds32_lists, top_n=15, ax=axes[2][1], value_type=value_type)
    # ae
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=1, ax=axes[0][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=5, ax=axes[1][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_ae_lists, top_n=15, ax=axes[2][2], value_type=value_type)
    # vae
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=1, ax=axes[0][3], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=5, ax=axes[1][3], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_baseline_lists,
                                 top_vae_lists, top_n=15, ax=axes[2][3], value_type=value_type)

    fig.tight_layout()


sns.set(rc={'figure.figsize': (16, 9)})
sns.set_theme(style='whitegrid')

# plot_bmp_overview()
plot_bmp_val_overview()
# plot_bmp_realdc_overview()

plt.show()
