
from constants import AE_CHECKPOINT_PATH, ENV
CHECKPOINT_PATH = AE_CHECKPOINT_PATH[ENV]


def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ['wall_time', 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


def save_train_loss_as_pkl(exp_name):
    dir_path = CHECKPOINT_PATH
    save_path = "/home/ivan_marcel/thesis/src/autoencoder/data/log_train_loss_pkl"
    log_df = convert_tb_data(f"{dir_path}/{exp_name}")
    import pandas as pd

    df = pd.DataFrame(columns=["train_loss", "val_loss"])
    # print(log_df.loc[0].step)
    print(log_df)

    for i in range(0, 1800, 3):  # for older 600,5
        train_loss = log_df.loc[i].value
        val_loss = log_df.loc[i+1].value
        kld_loss = log_df.loc[i+2].value
        df = df.append({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "kld_loss": kld_loss
        }, ignore_index=True)
    print(df)
    df.to_pickle(f"{save_path}/{exp_name}.pkl")


if __name__ == "__main__":
    exp_name = "VAE_T1C_BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641757232"
    # "VAE_T1C_BC_24_LD_512_MD_16_BS_2_TS_1500_LR_3e-05_ME_600_BETA_0001_1641550824"

    # AE logs
    # "T1C_BC_24_LD_128_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_BETA_0001_1641399246"
    # "T1C_BC_24_LD_512_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_BETA_0001_1641390171"
    # "T1C_BC_24_LD_32_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_BETA_0001_1641399376"
    # "BC_24_LD_1024_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_BETA_0001_1640617079"
    # "BC_24_LD_2048_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_1638352499"
    # "BC_24_LD_4096_MD_16_BS_2_TS_1500_LR_1e-05_ME_120_1636735907"
    save_train_loss_as_pkl(exp_name)
