import os
import sys
import json
import datetime
import pandas as pd

from params import LOG_PATH

LOGGED_IN_CONFIG = [
    "encoder",
    "decoder",
    "num_classes",
    "activation",
    "loss",
    "optimizer",
    "batch_size",
    "epochs",
    "lr",
    "warmup_prop",
    "k",
    "random_state",
]


class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file

    Args:
        directory (str, optional): Path to save logs at. Defaults to "".
        name (str, optional): Name of the file to save the logs in. Defaults to "logs.txt".
    """

    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    """
    Creates the directory for logging.
    Logs will be saved at log_path/date_of_day/exp_id

    Args:
        log_path ([str]): Directory

    Returns:
        str: Path to the created log folder
    """
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        os.mkdir(log_today)

    exp_id = len(os.listdir(log_today))
    log_folder = log_today + f"{exp_id}/"

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    else:
        print("Experiment already exists")

    return log_folder


def update_history(history, metrics, epoch, loss, val_loss, time):
    """
    Updates a training history dataframe.

    Args:
        history (pandas dataframe or None): Previous history.
        metrics (dict): Metrics dictionary.
        epoch (int): Epoch.
        loss (float): Training loss.
        val_loss (float): Validation loss.
        time (float): Epoch duration.

    Returns:
        pandas dataframe: history
    """
    new_history = {
        "epoch": [epoch],
        "time": [time],
        "loss": [loss],
        "val_loss": [val_loss],
    }
    new_history.update(metrics)

    new_history = pd.DataFrame.from_dict(new_history)

    if history is not None:
        return pd.concat([history, new_history]).reset_index(drop=True)
    else:
        return new_history


def save_config(config, path):
    """
    Saves a config as a json and pandas dataframe

    Args:
        config (Config): Config.
        path (str): Path to save at.

    Returns:
        pandas dataframe: Config as a dataframe
    """
    dic = config.__dict__.copy()
    del dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"]

    with open(path, "w") as f:
        json.dump(dic, f)


def update_overall_logs(metrics, config_df, log_path):
    """
    Updates a .csv containing logs for several experiments.

    Args:
        metrics (pandas dataframe): Metrics dataframe.
        config_df (pandas dataframe): Config as a dataframe as returned by the save_config function.
        log_path (str): Path to save at.

    Returns:
        pandas dataframe: Updated dataframe containing logs.
    """
    filename = (
        f"{LOG_PATH}logs_{config_df['mode'][0]}_{config_df['target_name'][0]}.csv"
    )

    metrics = metrics[["auc", "accuracy", "f1"]]
    config_df = config_df[LOGGED_IN_CONFIG]
    df = pd.concat([config_df, metrics], axis=1)
    df["path"] = log_path

    try:
        logs = pd.read_csv(filename)
        logs = pd.concat([logs, df], sort=False).reset_index(drop=True)
    except FileNotFoundError:
        logs = df

    logs.to_csv(filename, index=False)

    return logs
