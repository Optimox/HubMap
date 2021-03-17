import torch
import numpy as np

NUM_WORKERS = 2

SIZE = 256
TILE_SIZE = 256
REDUCE_FACTOR = 4


DATA_PATH = "../input/"
TIFF_PATH = DATA_PATH + "train/"
TIFF_PATH_4 = DATA_PATH + "train_4/"

LOG_PATH = "../logs/"
OUT_PATH = "../output/"
IMG_SIZE = (SIZE, SIZE)

IMG_PATH = DATA_PATH + f"train_{SIZE}_red_{REDUCE_FACTOR}"
MASK_PATH = DATA_PATH + f"masks_{SIZE}_red_{REDUCE_FACTOR}"

CLASSES = ["ftus"]

NUM_CLASSES = len(CLASSES)

MEAN = np.array([0.66437738, 0.50478148, 0.70114894])
STD = np.array([0.15825711, 0.24371008, 0.13832686])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


LAB_STATS = {
    "2f6ecfcdf": (
        np.array([145.15535701, 137.99020286, 118.54792884]),
        np.array([89.94705177, 14.23873667, 13.47610648]),
    ),
    "aaa6a05cc": (
        np.array([160.37612878, 143.02428181, 114.15084724]),
        np.array([75.359951, 17.53332275, 16.0875504]),
    ),
    "cb2d976f4": (
        np.array([149.2735253, 140.2772118, 117.25825769]),
        np.array([85.75920275, 16.05684167, 13.85500175]),
    ),
    "0486052bb": (
        np.array([150.91806562, 137.70516107, 118.47462954]),
        np.array([87.26843423, 13.69916489, 13.26900443]),
    ),
    "e79de561c": (
        np.array([157.9340989, 146.50102633, 111.57854369]),
        np.array([55.07994246, 15.72905062, 14.46999661]),
    ),
    "095bf7a1f": (
        np.array([129.58890923, 144.69630213, 116.09852381]),
        np.array([79.53312429, 18.46779254, 13.0580684]),
    ),
    "54f2eec69": (
        np.array([149.5829189, 143.24307541, 116.13484479]),
        np.array([73.73585502, 17.1945404, 14.00904746]),
    ),
    "1e2425f28": (
        np.array([133.55921634, 153.74112745, 111.1193993]),
        np.array([66.38893461, 20.50278703, 13.87867939]),
    ),
}
