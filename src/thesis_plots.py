import matplotlib.pyplot as plt
# plt.style.use('classic')
import numpy as np
import pandas as pd

import seaborn as sns


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

    fig, axes = plt.subplots(2, 2)

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
    train_plot.set_ylim(0.05, 0.20)
    train_plot.set_title("Training loss per epoch (Dice part)")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("Dice Loss")

    # train kld
    dice_loss_data = train_loss_data - kld_loss_data
    train_plot = sns.lineplot(
        ax=axes[1][1], data=kld_loss_data, dashes=False)
    train_plot.set_xlim(-1)
    train_plot.set_ylim(0.00, 0.10)
    train_plot.set_title("Training loss per epoch (kld part)")
    train_plot.set_xticks(ticks)
    train_plot.set_xlabel("epoch")
    train_plot.set_ylabel("0.001 * D_KL Loss")

    # plt.savefig("test.png", bbox_inches='tight', dpi=800)
    plt.show()


sns.set(rc={'figure.figsize': (12, 4.5)})
sns.set_theme(style='whitegrid')
plot_train_and_val_loss_vae()
