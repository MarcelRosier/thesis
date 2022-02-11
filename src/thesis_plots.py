import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import col
import seaborn as sns
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
from seaborn.palettes import color_palette

from utils import DSValueType


def plot_train_and_val_loss_ae():
    base_pkl_path = "/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/log_train_loss_pkl/ae"
    train_loss_data = pd.DataFrame()
    val_loss_data = pd.DataFrame()

    exp_name = "T1C_BC_24_LD_32_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_BETA_0001_1641399376"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["32"] = df['train_loss']
    val_loss_data["32"] = df['val_loss']

    exp_name = "T1C_BC_24_LD_128_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_BETA_0001_1641399246"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["128"] = df['train_loss']
    val_loss_data["128"] = df['val_loss']

    exp_name = "T1C_BC_24_LD_512_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_BETA_0001_1641390171"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["512"] = df['train_loss']
    val_loss_data["512"] = df['val_loss']

    exp_name = "T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_600_BETA_0001_1642258438"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["1024"] = df['train_loss']
    val_loss_data["1024"] = df['val_loss']

    exp_name = "BC_24_LD_2048_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_1638352499"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["2048"] = df['train_loss']
    val_loss_data["2048"] = df['val_loss']

    exp_name = "BC_24_LD_4096_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_1636735907"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["4096"] = df['train_loss']
    val_loss_data["4096"] = df['val_loss']

    ticks = list(np.linspace(0, 120, 13))
    # yticks = list(np.linspace(0, 1, 11))

    fig, axes = plt.subplots(1, 2)

    # train
    train_plot = sns.lineplot(ax=axes[0], data=train_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.03, 0.13)
    train_plot.set_title("Training loss per epoch")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("Dice Loss")
    train_plot.set_yticks(list(np.linspace(0.03, 0.13, 11)))

    # val
    val_plot = sns.lineplot(ax=axes[1], data=val_loss_data, dashes=False)
    val_plot.set_xlim(-1)
    val_plot.set_ylim(0.07, 0.17)
    val_plot.set_title("Validation loss per epoch")
    val_plot.set_xticks(ticks)
    val_plot.set_xlabel("epoch")
    val_plot.set_ylabel("Dice Loss")
    val_plot.set_yticks(list(np.linspace(0.07, 0.17, 11)))
    # plt.savefig("test.png", bbox_inches='tight', dpi=800)
    # plt.show()


def plot_train_and_val_loss_vae():
    base_pkl_path = "/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/log_train_loss_pkl/vae"
    train_loss_data = pd.DataFrame()
    val_loss_data = pd.DataFrame()
    kld_loss_data = pd.DataFrame()

    exp_name = "VAE_T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641757232"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["1024"] = df['train_loss']
    val_loss_data["1024"] = df['val_loss']
    kld_loss_data["1024"] = df['kld_loss']

    exp_name = "VAE_T1C_BC_24_LD_512_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641550824"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["512"] = df['train_loss']
    val_loss_data["512"] = df['val_loss']
    kld_loss_data["512"] = df['kld_loss']

    exp_name = "VAE_T1C_BC_24_LD_128_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641806733"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["128"] = df['train_loss']
    val_loss_data["128"] = df['val_loss']
    kld_loss_data["128"] = df['kld_loss']

    exp_name = "VAE_T1C_BC_24_LD_32_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641806824"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["32"] = df['train_loss']
    val_loss_data["32"] = df['val_loss']
    kld_loss_data["32"] = df['kld_loss']

    exp_name = "VAE_T1C_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641830476"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["8"] = df['train_loss']
    val_loss_data["8"] = df['val_loss']
    kld_loss_data["8"] = df['kld_loss']

    exp_name = "VAE_T1C_BC_24_LD_4_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641900178"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["4"] = df['train_loss']
    val_loss_data["4"] = df['val_loss']
    kld_loss_data["4"] = df['kld_loss']

    # exp_name = "VAE_T1C_BC_24_LD_2_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641834797"
    # df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    # train_loss_data["2"] = df['train_loss']
    # val_loss_data["2"] = df['val_loss']
    # kld_loss_data["2"] = df['kld_loss']

    ticks = list(np.linspace(0, 600, 13))
    # yticks = list(np.linspace(0, 1, 11))

    fig, axes = plt.subplots(2, 2, sharex=True)

    # train sum
    train_plot = sns.lineplot(
        ax=axes[0][0], data=train_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.10, 0.20)
    train_plot.set_title("Training loss per epoch")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel(r"Dice + 0.001* $D_{KL}$ Loss")
    train_plot.set_yticks(list(np.linspace(0.1, 0.2, 11)))
    train_plot.legend(loc='lower left')

    # val
    val_plot = sns.lineplot(
        ax=axes[0][1], data=val_loss_data, dashes=False, alpha=0.8)
    val_plot.set_xlim(-1)
    val_plot.set_ylim(0.08, 0.13)
    val_plot.set_title("Validation loss per epoch")
    val_plot.set_xticks(ticks)
    val_plot.set_xlabel("epoch")
    val_plot.set_ylabel(r"Dice + 0.001* $D_{KL}$ Loss")
    val_plot.set_yticks(list(np.linspace(0.08, 0.13, 11)))
    val_plot.legend(loc='lower left')

    # train dice
    dice_loss_data = train_loss_data - kld_loss_data
    train_plot = sns.lineplot(
        ax=axes[1][0], data=dice_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.06, 0.16)
    train_plot.set_title("Training loss per epoch (Dice part)")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("Dice Loss")
    train_plot.set_yticks(list(np.linspace(0.06, 0.16, 11)))
    train_plot.legend(loc='lower left')

    # train kld
    dice_loss_data = train_loss_data - kld_loss_data
    train_plot = sns.lineplot(
        ax=axes[1][1], data=kld_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.0, 0.10)
    train_plot.set_title(r"Training loss per epoch ($D_{KL}$ part)")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel(r"0.001 * $D_{KL}$ Loss")
    train_plot.set_yticks(list(np.linspace(0, 0.1, 11)))
    train_plot.legend(loc='lower left')

    # plt.savefig("test_vae.png", bbox_inches='tight', dpi=800)
    # plt.show()


def plot_t1c_and_flair_train_and_val_loss_vae():
    base_pkl_path = "/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/log_train_loss_pkl/vae"
    train_loss_data = pd.DataFrame()
    val_loss_data = pd.DataFrame()
    kld_loss_data = pd.DataFrame()

    exp_name = "VAE_T1C_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642602180"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["T1Gd"] = df['train_loss']
    val_loss_data["T1Gd"] = df['val_loss']
    kld_loss_data["T1Gd"] = df['kld_loss']

    exp_name = "VAE_FLAIR_BC_24_LD_8_MD_16_BS_2_TS_1500_LR_3e-05_ME_1000_BETA_0001_1642709006"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["FLAIR"] = df['train_loss']
    val_loss_data["FLAIR"] = df['val_loss']
    kld_loss_data["FLAIR"] = df['kld_loss']

    # exp_name = "VAE_T1C_BC_24_LD_2_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641834797"
    # df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    # train_loss_data["2"] = df['train_loss']
    # val_loss_data["2"] = df['val_loss']
    # kld_loss_data["2"] = df['kld_loss']

    ticks = list(np.linspace(0, 1000, 11))
    # yticks = list(np.linspace(0, 1, 11))

    fig, axes = plt.subplots(2, 2, sharex=True)

    # train sum
    train_plot = sns.lineplot(
        ax=axes[0][0], data=train_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.08, 0.18)
    train_plot.set_title("Training loss per epoch")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel(r"Dice + 0.001* $D_{KL}$ Loss")
    train_plot.set_yticks(list(np.linspace(0.08, 0.18, 11)))
    train_plot.legend(loc='lower left')

    # val
    val_plot = sns.lineplot(
        ax=axes[0][1], data=val_loss_data, dashes=False, alpha=0.8)
    val_plot.set_xlim(-1)
    val_plot.set_ylim(0.08, 0.13)
    val_plot.set_title("Validation loss per epoch")
    val_plot.set_xticks(ticks)
    val_plot.set_xlabel("epoch")
    val_plot.set_ylabel(r"Dice + 0.001* $D_{KL}$ Loss")
    val_plot.set_yticks(list(np.linspace(0.08, 0.13, 11)))
    val_plot.legend(loc='lower left')

    # train dice
    dice_loss_data = train_loss_data - kld_loss_data
    train_plot = sns.lineplot(
        ax=axes[1][0], data=dice_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.04, 0.14)
    train_plot.set_title("Training loss per epoch (Dice part)")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("Dice Loss")
    train_plot.set_yticks(list(np.linspace(0.04, 0.14, 11)))
    train_plot.legend(loc='lower left')

    # train kld
    dice_loss_data = train_loss_data - kld_loss_data
    train_plot = sns.lineplot(
        ax=axes[1][1], data=kld_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.0, 0.10)
    train_plot.set_title(r"Training loss per epoch ($D_{KL}$ part)")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel(r"0.001 * $D_{KL}$ Loss")
    train_plot.set_yticks(list(np.linspace(0, 0.1, 11)))
    train_plot.legend(loc='lower left')

    # plt.savefig("test_vae.png", bbox_inches='tight', dpi=800)
    # plt.show()


def plot_best_match_input_dice():
    base_dir = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/custom_dice/testset_size_50000/dim_128/dice"
    tumor_ids = os.listdir(base_dir)
    tumor_ids.sort(key=lambda f: int(f[3:6]))
    t1c_scores = []
    flair_scores = []

    for tumor_id in tumor_ids:
        with open(f"{base_dir}/{tumor_id}") as json_file:
            data = json.load(json_file)
            best_key = max(
                data.keys(), key=lambda k: data[k][DSValueType.COMBINED.value])
            t1c_scores.append(data[best_key][DSValueType.T1C.value])
            flair_scores.append(data[best_key][DSValueType.FLAIR.value])

    fig, axes = plt.subplots(3, 1, sharex=True)
    # fig.suptitle("Dice score between input and best match")
    # fig.text(0.5135, 0.075, 'Tumors', ha='center')
    x_labels = [int(tumor[3:6]) for tumor in tumor_ids]
    y_ticks = np.linspace(0, 1, 11)

    t1c_avg = sum(t1c_scores) / len(t1c_scores)
    print(t1c_avg)
    # t1c_plot = sns.barplot(
    #     ax=axes[0], x=x_labels, y=t1c_scores, color="#3070B3")
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    # red_green_pal = sns.diverging_palette(0, 150, l=50, n=32, as_cmap=True)
    spectral_palette = sns.color_palette("Spectral", as_cmap=True)
    # rg2 = sns.color_palette("RdYlGn_r", 5)
    colors = [spectral_palette(norm(c)) for c in t1c_scores]
    p = sns.color_palette("Spectral", as_cmap=True)
    t1c_plot = sns.barplot(
        ax=axes[0], x=x_labels, y=t1c_scores, palette=colors)
    t1c_plot.set_xlim(-0.5)
    t1c_plot.set_ylim(-0.01, 1)
    # t1c_plot.set_title("T1Gd")
    # t1c_plot.set_xlabel("tumors")
    t1c_plot.set_xticklabels([])
    t1c_plot.set_ylabel("T1Gd dice score", size=16)
    t1c_plot.set_yticks(y_ticks)
    t1c_plot.axhline(t1c_avg, color="#446EB0")  # "#5ba56e")

    colors = [spectral_palette(norm(c)) for c in flair_scores]
    flair_avg = sum(flair_scores) / len(flair_scores)
    print(flair_avg)
    flair_plot = sns.barplot(
        ax=axes[1], x=x_labels, y=flair_scores, palette=colors)
    flair_plot.set_xlim(-0.5)
    flair_plot.set_ylim(-0.01, 1)
    # flair_plot.set_title("FLAIR")
    # flair_plot.set_xlabel("tumors")
    flair_plot.set_xticklabels([])
    flair_plot.set_ylabel("FLAIR dice score", size=16)
    flair_plot.set_yticks(y_ticks)
    flair_plot.axhline(flair_avg, color="#446EB0")  # "#5ba56e")

    print(t1c_scores)
    print(flair_scores)
    # combined plot
    y_ticks = np.linspace(0, 2, 21)
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)
    combined_scores = [t + f for t, f in zip(t1c_scores, flair_scores)]
    colors = [spectral_palette(norm(c)) for c in combined_scores]
    combined_scores_avg = sum(combined_scores) / len(combined_scores)
    print(combined_scores_avg)
    combined_scores_plot = sns.barplot(
        ax=axes[2], x=x_labels, y=combined_scores, palette=colors)
    combined_scores_plot.set_xlim(-0.5)
    combined_scores_plot.set_ylim(-0.01, 2)
    # combined_scores_plot.set_title("Combined")
    # combined_scores_plot.set_xlabel("tumors")
    combined_scores_plot.set_xticklabels([])
    combined_scores_plot.set_ylabel("Combined dice score", size=16)
    combined_scores_plot.set_xlabel("Tumors")
    combined_scores_plot.set_yticks(y_ticks)

    for label in combined_scores_plot.get_yticklabels()[1:][::2]:
        label.set_visible(False)
    combined_scores_plot.axhline(
        combined_scores_avg, color="#446EB0")  # "#5ba56e")
    # plt.show()
    # plt.savefig("test_dice.png", bbox_inches='tight', dpi=800)


def plot_t1c_flair_train_and_val_loss_ae():
    base_pkl_path = "/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/log_train_loss_pkl/ae"
    train_loss_data = pd.DataFrame()
    val_loss_data = pd.DataFrame()

    exp_name = "T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_600_BETA_0001_1642258438"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["T1Gd"] = df['train_loss']
    val_loss_data["T1Gd"] = df['val_loss']

    exp_name = "FLAIR_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_300_BETA_0001_1642493260"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["FLAIR"] = df['train_loss']
    val_loss_data["FLAIR"] = df['val_loss']

    ticks = list(np.linspace(0, 300, 13))
    # yticks = list(np.linspace(0, 1, 11))

    fig, axes = plt.subplots(1, 2)
    # print(train_loss_data)
    # print(val_loss_data)
    # train
    train_plot = sns.lineplot(ax=axes[0], data=train_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.0, 0.1)
    train_plot.set_title("Training loss per epoch")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("Dice Loss")
    train_plot.set_yticks(list(np.linspace(0.0, 0.1, 11)))

    # val
    val_plot = sns.lineplot(ax=axes[1], data=val_loss_data, dashes=False)
    val_plot.set_xlim(-1)
    val_plot.set_ylim(0.04, 0.14)
    val_plot.set_title("Validation loss per epoch")
    val_plot.set_xticks(ticks)
    val_plot.set_xlabel("epoch")
    val_plot.set_ylabel("Dice Loss")
    val_plot.set_yticks(list(np.linspace(0.04, 0.14, 11)))
    # plt.savefig("test.png", bbox_inches='tight', dpi=800)
    # plt.show()


def plot_enc_best_match_presence_overview(is_ae: bool, is_1024: bool = False):

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

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 3))
    cols = ["Top 1", "Top 5", "Top 15"]
    rows = [r"$64^3$", r"$32^3$"]
    for ax, col in zip(axes, cols):
        ax.set_ylabel(col, rotation=0)
    # for ax, row in zip(axes

    # Combined
    value_type = DSValueType.COMBINED
    # fig.suptitle("AE FLAIR")
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/{"ae" if is_ae else ("vae/1024" if is_1024 else "vae/8")}/top_15_{value_type.value}.json') as file:
        data: dict = json.load(file)
    tumor_ids = data.keys()
    top_gt_lists = []
    top_enc_lists = []
    for tumor_id in tumor_ids:
        top_gt_lists.append(data[tumor_id]['top_gt'])
        top_enc_lists.append(data[tumor_id]['top_enc'])
    # chosen_ids = ['tgm036_preop', 'tgm008_preop',
    #               'tgm055_preop', 'tgm025_preop', 'tgm035_preop']
    # for cid in chosen_ids:
    #     rid = list(tumor_ids).index(cid)
    #     same = top_gt_lists[rid][0] == top_enc_lists[rid][0]
    #     print(same)
    # for tumor_id in tumor_ids:
    #     same = top_gt_lists[list(tumor_ids).index(
    #         tumor_id)][0] == top_enc_lists[list(tumor_ids).index(tumor_id)][0]
    #     if same:
    #         print(tumor_id)
    # print()
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=1, ax=axes[0], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=5, ax=axes[1], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=15, ax=axes[2], value_type=value_type)
    # # flair
    # value_type = DSValueType.FLAIR
    # with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/top_15_{value_type}.json') as file:
    #     data: dict = json.load(file)
    # tumor_ids = data.keys()
    # top_gt_lists = []
    # top_enc_lists = []
    # for tumor_id in tumor_ids:
    #     top_gt_lists.append(data[tumor_id]['top_gt'])
    #     top_enc_lists.append(data[tumor_id]['top_enc'])

    # plot_enc_best_match_presence(tumor_ids, top_gt_lists,
    #                              top_enc_lists, top_n=1, ax=axes[1][0], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_gt_lists,
    #                              top_enc_lists, top_n=5, ax=axes[1][1], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_gt_lists,
    #                              top_enc_lists, top_n=15, ax=axes[1][2], value_type=value_type)
    # # t1c
    # value_type = DSValueType.T1C
    # with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/top_15_{value_type}.json') as file:
    #     data: dict = json.load(file)
    # tumor_ids = data.keys()
    # top_gt_lists = []
    # top_enc_lists = []
    # for tumor_id in tumor_ids:
    #     top_gt_lists.append(data[tumor_id]['top_gt'])
    #     top_enc_lists.append(data[tumor_id]['top_enc'])

    # plot_enc_best_match_presence(tumor_ids, top_gt_lists,
    #                              top_enc_lists, top_n=1, ax=axes[2][0], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_gt_lists,
    #                              top_enc_lists, top_n=5, ax=axes[2][1], value_type=value_type)
    # plot_enc_best_match_presence(tumor_ids, top_gt_lists,
    #                              top_enc_lists, top_n=15, ax=axes[2][2], value_type=value_type)

    fig.tight_layout()


def plot_vae_enc_best_match_presence_overview():

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

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 3))
    rows = ["Top 1", "Top 5", "Top 15"]
    cols = ["Latent size 8", "Latent size 1024",
            "Trainset size 20k; latent size 8"]
    print(axes[:, 0])
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0)
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    # Combined 8
    value_type = DSValueType.COMBINED
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/vae/8/top_15_combined.json') as file:
        data_8: dict = json.load(file)
    tumor_ids = data_8.keys()
    top_gt_lists = []
    top_enc_lists = []
    for tumor_id in tumor_ids:
        top_gt_lists.append(data_8[tumor_id]['top_gt'])
        top_enc_lists.append(data_8[tumor_id]['top_enc'])
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=1, ax=axes[0][0], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=5, ax=axes[1][0], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=15, ax=axes[2][0], value_type=value_type)
    # Combined 1024
    value_type = DSValueType.COMBINED
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/vae/1024/top_15_combined.json') as file:
        data_1024: dict = json.load(file)
    tumor_ids = data_1024.keys()
    top_gt_lists = []
    top_enc_lists = []
    for tumor_id in tumor_ids:
        top_gt_lists.append(data_1024[tumor_id]['top_gt'])
        top_enc_lists.append(data_1024[tumor_id]['top_enc'])
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=1, ax=axes[0][1], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=5, ax=axes[1][1], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=15, ax=axes[2][1], value_type=value_type)
    # Combined 8 20k
    value_type = DSValueType.COMBINED
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/vae20k/8/top_15_combined.json') as file:
        data_1024: dict = json.load(file)
    tumor_ids = data_1024.keys()
    top_gt_lists = []
    top_enc_lists = []
    for tumor_id in tumor_ids:
        top_gt_lists.append(data_1024[tumor_id]['top_gt'])
        top_enc_lists.append(data_1024[tumor_id]['top_enc'])
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=1, ax=axes[0][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=5, ax=axes[1][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=15, ax=axes[2][2], value_type=value_type)
    fig.tight_layout()


def plot_vae_flair_enc_best_match_presence_overview():

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

    fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 3))
    # fig.suptitle("VAE FLAIR")
    rows = ["Top 1", "Top 5", "Top 15"]
    cols = ["Latent size 8", "Latent size 1024",
            "Trainset size 20k; Latent size 8"]
    print(axes[:, 0])
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0)
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    #  8
    value_type = DSValueType.COMBINED
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/vae/8/top_15_{value_type.value}.json') as file:
        data_8: dict = json.load(file)
    tumor_ids = data_8.keys()
    top_gt_lists = []
    top_enc_lists = []
    for tumor_id in tumor_ids:
        top_gt_lists.append(data_8[tumor_id]['top_gt'])
        top_enc_lists.append(data_8[tumor_id]['top_enc'])
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=1, ax=axes[0][0], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=5, ax=axes[1][0], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=15, ax=axes[2][0], value_type=value_type)
    #  1024
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/vae/1024/top_15_{value_type.value}.json') as file:
        data_1024: dict = json.load(file)
    tumor_ids = data_1024.keys()
    top_gt_lists = []
    top_enc_lists = []
    for tumor_id in tumor_ids:
        top_gt_lists.append(data_1024[tumor_id]['top_gt'])
        top_enc_lists.append(data_1024[tumor_id]['top_enc'])
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=1, ax=axes[0][1], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=5, ax=axes[1][1], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=15, ax=axes[2][1], value_type=value_type)
    #  8 20k
    value_type = DSValueType.FLAIR
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/final_50k_top15/vae20k/8/top_15_{value_type.value}.json') as file:
        data_1024: dict = json.load(file)
    tumor_ids = data_1024.keys()
    top_gt_lists = []
    top_enc_lists = []
    for tumor_id in tumor_ids:
        top_gt_lists.append(data_1024[tumor_id]['top_gt'])
        top_enc_lists.append(data_1024[tumor_id]['top_enc'])
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=1, ax=axes[0][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=5, ax=axes[1][2], value_type=value_type)
    plot_enc_best_match_presence(tumor_ids, top_gt_lists,
                                 top_enc_lists, top_n=15, ax=axes[2][2], value_type=value_type)
    fig.tight_layout()


def plot_recon_losses(is_ae: bool):
    tum_blue = "#446EB0"
    orange = "#F0746E"
    lime = "#7CCBA2"
    purple = "#7C1D6F"
    if is_ae:
        flair_avg_val = 1 - 0.050507
        t1c_avg_val = 1 - 0.063680
    else:
        pass

    folder = 'ae_TS_1500' if is_ae else 'vae_TS_1500'
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/recon_analysis/{folder}/syn/monai_scores_flair.json') as file:
        flair_data: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/recon_analysis/{folder}/syn/monai_scores_t1c.json') as file:
        t1c_data: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/recon_analysis/{folder}/real/monai_scores_flair.json') as file:
        real_flair_data: dict = json.load(file)
    with open(f'/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/recon_analysis/{folder}/real/monai_scores_t1c.json') as file:
        real_t1c_data: dict = json.load(file)

    # General plot
    fig, axes = plt.subplots(2, 2, sharex=False,
                             sharey=False, figsize=(12, 3))
    axes[0][0].set_title(r"Synthetic Dataset $S$")
    axes[0][1].set_title(r"Real Dataset $P$")
    axes[0][0].set_ylabel("FLAIR")
    axes[1][0].set_ylabel("T1Gd")

    # syn flair plot
    sorted_flair_list_nth = sorted(flair_data.values())  # [::100]
    flair_avg = sum(flair_data.values())/len(flair_data.values())
    print(f"{flair_avg=}")
    # print(f"{flair_avg_val=}")
    x = np.linspace(1, len(sorted_flair_list_nth), len(sorted_flair_list_nth))
    flair_plot = sns.lineplot(
        ax=axes[0][0], x=x, y=sorted_flair_list_nth, color=tum_blue)
    flair_plot.set_yticks(list(np.linspace(0, 1, 11)))
    flair_plot.set_xlim(0, 50000)
    flair_plot.axhline(flair_avg, color=orange)
    # flair_plot.axhline(flair_avg_val, color=lime)
    flair_plot.legend(
        [r'$Dice_{S}$', r'$\overline{Dice}_{S}$', r'$\overline{Dice}_{Validation}$'], loc='lower right')

    # syn t1c plot
    t1c_avg = sum(t1c_data.values())/(len(t1c_data.values()))
    print(f"{t1c_avg=}")
    # print(f"{t1c_avg_val=}")
    sorted_t1c_list_nth = sorted(t1c_data.values())  # [::100]
    x = np.linspace(1, len(sorted_t1c_list_nth), len(sorted_t1c_list_nth))
    t1c_plot = sns.lineplot(
        ax=axes[1][0], x=x, y=sorted_t1c_list_nth, color=tum_blue)
    t1c_plot.set_yticks(list(np.linspace(0, 1, 11)))
    t1c_plot.set_xlim(0, 50000)
    t1c_plot.axhline(t1c_avg, color=orange)
    # t1c_plot.axhline(t1c_avg_val, color=lime)
    t1c_plot.legend([r'$Dice_{S}$',
                    r'$\overline{Dice}_{S}$', r'$\overline{Dice}_{Validation}$'], loc='lower right')

    # real flair plot
    real_sorted_flair_list = sorted(real_flair_data.values())
    real_flair_avg = (sum(real_flair_data.values()) /
                      len(real_flair_data.values()))
    print(f"{real_flair_avg=}")
    x = np.linspace(1, len(real_sorted_flair_list),
                    len(real_sorted_flair_list))
    flair_plot = sns.lineplot(
        ax=axes[0][1], x=x, y=real_sorted_flair_list, color=tum_blue)
    flair_plot.set_yticks(list(np.linspace(0, 1, 11)))
    flair_plot.set_xlim(0, len(real_sorted_flair_list))
    flair_plot.axhline(real_flair_avg, color=orange)
    flair_plot.legend(
        [r'$Dice_{P}$', r'$\overline{Dice}_{P}$'], loc='lower right')

    # real t1cplot
    real_sorted_t1c_list = sorted(real_t1c_data.values())
    real_t1c_avg = (sum(real_t1c_data.values()) /
                    len(real_t1c_data.values()))
    print(f"{real_t1c_avg=}")
    x = np.linspace(1, len(real_sorted_t1c_list),
                    len(real_sorted_t1c_list))
    t1c_plot = sns.lineplot(
        ax=axes[1][1], x=x, y=real_sorted_t1c_list, color=tum_blue)
    t1c_plot.set_yticks(list(np.linspace(0, 1, 11)))
    t1c_plot.set_xlim(0, len(real_sorted_t1c_list))
    t1c_plot.axhline(real_t1c_avg, color=orange)
    t1c_plot.legend(
        [r'$Dice_{P}$', r'$\overline{Dice}_{P}$'], loc='lower right')


sns.set(rc={'figure.figsize': (16, 9)})
sns.set_theme(style='whitegrid')
# plot_train_and_val_loss_vae()
# plot_best_match_input_dice()
# plot_t1c_flair_train_and_val_loss_ae()
# plot_enc_best_match_presence_overview(is_ae=True, is_1024=True)
plot_vae_flair_enc_best_match_presence_overview()
# print("AE")
# plot_recon_losses(is_ae=True)
# print("VAE")
# plot_recon_losses(is_ae=False)
# plot_t1c_and_flair_train_and_val_loss_vae()
plt.show()
