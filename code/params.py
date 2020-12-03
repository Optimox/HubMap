import torch
import numpy as np

NUM_WORKERS = 2

SIZE = 256
REDUCE_FACTOR = 4

DATA_PATH = "../input/hubmap-kidney-segmentation/"
LOG_PATH = "../logs/"

IMG_SIZE = (SIZE, SIZE)

IMG_PATH = DATA_PATH + f"train_{SIZE}_red_{REDUCE_FACTOR}"

CLASSES = ["ftus"]

NUM_CLASSES = len(CLASSES)

MEAN = np.array([0.66437738, 0.50478148, 0.70114894])
STD = np.array([0.15825711, 0.24371008, 0.13832686])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"