import os
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns
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

    exp_name = "BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_BETA_0001_1640617079"
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

    # val
    val_plot = sns.lineplot(ax=axes[1], data=val_loss_data, dashes=False)
    val_plot.set_xlim(-1)
    val_plot.set_ylim(0.05, 0.18)
    val_plot.set_title("Validation loss per epoch")
    val_plot.set_xticks(ticks)
    val_plot.set_xlabel("epoch")
    val_plot.set_ylabel("Dice Loss")
    plt.savefig("test.png", bbox_inches='tight', dpi=800)
    # plt.show()


def plot_train_and_val_loss_vae():
    base_pkl_path = "/Users/marcelrosier/Projects/uni/thesis/src/autoencoder/data/log_train_loss_pkl/vae"
    train_loss_data = pd.DataFrame()
    val_loss_data = pd.DataFrame()
    kld_loss_data = pd.DataFrame()

    exp_name = "VAE_T1C_BC_24_LD_512_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641550824"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["512"] = df['train_loss']
    val_loss_data["512"] = df['val_loss']
    kld_loss_data["512"] = df['kld_loss']

    exp_name = "VAE_T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641757232"
    df = pd.read_pickle(f"{base_pkl_path}/{exp_name}.pkl")
    train_loss_data["1024"] = df['train_loss']
    val_loss_data["1024"] = df['val_loss']
    kld_loss_data["1024"] = df['kld_loss']

    ticks = list(np.linspace(0, 600, 13))
    # yticks = list(np.linspace(0, 1, 11))

    fig, axes = plt.subplots(2, 2, sharex=True)

    # train sum
    train_plot = sns.lineplot(
        ax=axes[0][0], data=train_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.12, 0.30)
    train_plot.set_title("Training loss per epoch")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("Dice + 0.001* D_KL Loss")

    # val
    val_plot = sns.lineplot(ax=axes[0][1], data=val_loss_data, dashes=False)
    val_plot.set_xlim(-1)
    val_plot.set_ylim(0.12, 0.30)
    val_plot.set_title("Validation loss per epoch")
    val_plot.set_xticks(ticks)
    val_plot.set_xlabel("epoch")
    val_plot.set_ylabel("Dice + 0.001* D_KL Loss")

    # train dice
    dice_loss_data = train_loss_data - kld_loss_data
    train_plot = sns.lineplot(
        ax=axes[1][0], data=dice_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.08, 0.20)
    train_plot.set_title("Training loss per epoch (Dice part)")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("Dice Loss")

    # train kld
    dice_loss_data = train_loss_data - kld_loss_data
    train_plot = sns.lineplot(
        ax=axes[1][1], data=kld_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.05, 0.10)
    train_plot.set_title("Training loss per epoch (kld part)")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("0.001 * D_KL Loss")

    plt.savefig("test_vae.png", bbox_inches='tight', dpi=800)
    # plt.show()


def plot_best_match_input_dice():
    base_dir = "/Users/marcelrosier/Projects/uni/thesis/src/baseline/data/testset_size_50000/dim_128/dice"
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

    fig, axes = plt.subplots(1, 2)
    x_labels = [int(tumor[3:6]) for tumor in tumor_ids]
    y_ticks = np.linspace(0, 1, 11)

    t1c_avg = sum(t1c_scores) / len(t1c_scores)
    # t1c_plot = sns.barplot(
    #     ax=axes[0], x=x_labels, y=t1c_scores, color="#3070B3")
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    red_green_pal = sns.diverging_palette(0, 150, l=50, n=32, as_cmap=True)
    colors = [red_green_pal(norm(c)) for c in t1c_scores]
    p = sns.color_palette("Spectral", as_cmap=True)
    t1c_plot = sns.barplot(
        ax=axes[0], x=x_labels, y=t1c_scores, palette=colors)
    t1c_plot.set_xlim(-1)
    t1c_plot.set_ylim(-0.01, 1)
    t1c_plot.set_title("Dice score between input and best match (T1Gd)")
    t1c_plot.set_xlabel("tumors")
    t1c_plot.set_xticklabels([])
    t1c_plot.set_ylabel("Dice score")
    t1c_plot.set_yticks(y_ticks)
    t1c_plot.axhline(t1c_avg, color="#5ba56e")

    colors = [red_green_pal(norm(c)) for c in flair_scores]
    flair_avg = sum(flair_scores) / len(flair_scores)
    flair_plot = sns.barplot(
        ax=axes[1], x=x_labels, y=flair_scores, palette=colors)
    flair_plot.set_xlim(-1)
    flair_plot.set_ylim(-0.01, 1)
    flair_plot.set_title("Dice score between input and best match (FLAIR)")
    flair_plot.set_xlabel("tumors")
    flair_plot.set_xticklabels([])
    flair_plot.set_ylabel("Dice score")
    flair_plot.set_yticks(y_ticks)
    flair_plot.axhline(flair_avg, color="#5ba56e")
    plt.show()
    # plt.savefig("test_dice.png", bbox_inches='tight', dpi=800)


sns.set(rc={'figure.figsize': (16, 9)})
sns.set_theme(style='whitegrid')
# plot_train_and_val_loss_vae()
plot_best_match_input_dice()
