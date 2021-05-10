# Hardcoded stuff, paths are to adapt to your setup

import torch
import numpy as np

NUM_WORKERS = 2

DATA_PATH = "../input/"
TIFF_PATH = DATA_PATH + "train/"
TIFF_PATH_4 = DATA_PATH + "train_4/"
TIFF_PATH_2 = DATA_PATH + "train_2/"
TIFF_PATH_TEST = DATA_PATH + "test/"

LOG_PATH = "../logs/"
OUT_PATH = "../output/"

CLASSES = ["ftus"]
NUM_CLASSES = len(CLASSES)

MEAN = np.array([0.66437738, 0.50478148, 0.70114894])
STD = np.array([0.15825711, 0.24371008, 0.13832686])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH_EXTRA = DATA_PATH + "extra_tiff/"


# Additonal images for PL
EXTRA_IMGS = ["VAN0003-LK-32-21-PAS_registered.ome", "VAN0011-RK-3-10-PAS_registered.ome"]
EXTRA_IMGS_SHAPES = {
    "VAN0003-LK-32-21-PAS_registered.ome": (41220, 41500),
    "VAN0011-RK-3-10-PAS_registered.ome": (37040, 53240),
}
