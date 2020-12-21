from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import tifffile as tiff
import pandas as pd
import rasterio
from rasterio.windows import Window

from params import DATA_PATH


def load_image(img_path):
    """
    Load image and make sure sizes matches df_info
    """
    df_info = pd.read_csv(DATA_PATH + "HuBMAP-20-dataset_information.csv")
    image_fname = img_path.rsplit("/", -1)[-1]
    W = int(df_info[df_info.image_file == image_fname]["width_pixels"])
    H = int(df_info[df_info.image_file == image_fname]["height_pixels"])
    img = tiff.imread(img_path).squeeze()

    channel_pos = np.argwhere(np.array(img.shape) == 3)[0][0]
    W_pos = np.argwhere(np.array(img.shape) == W)[0][0]
    H_pos = np.argwhere(np.array(img.shape) == H)[0][0]

    img = np.moveaxis(img, (H_pos, W_pos, channel_pos), (0, 1, 2))
    return img


class TileDataset(Dataset):
    """
    Dataset to read from tiled images.
    """
    def __init__(self, df, img_dir="", mask_dir="", transforms=None):
        """
        Args:
            df (pandas dataframe): file_names.
            img_dir (str, optional): Images directory. Defaults to "".
            mask_dir (str, optional): Masks directory. Defaults to "".
            transforms (albumentation transforms, optional) : Transforms. Defaults to None.
        """
        self.df = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        tile_name = self.df.loc[idx, "tile_name"]

        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, tile_name)), cv2.COLOR_BGR2RGB
        )
        # img = cv2.imread(os.path.join(self.img_dir, tile_name))

        mask = cv2.imread(os.path.join(self.mask_dir, tile_name), cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        return img, mask


class PredictFromImgDataset(Dataset):
    def __init__(
        self,
        original_img_path,
        mask_name=None,
        overlap_factor=1,
        tile_size=256,
        reduce_factor=4,
        transforms=None,
    ):
        self.original_img = rasterio.open(original_img_path)
        self.orig_size = self.original_img.shape
        self.raw_tile_size = tile_size
        self.reduce_factor = reduce_factor
        self.tile_size = tile_size * reduce_factor
        self.overlap_factor = overlap_factor
        self.positions = self.get_positions()
        self.transforms = transforms
        if mask_name is not None:
            df_mask = pd.read_csv(DATA_PATH + "train.csv")
            self.msk_encoding = df_mask[df_mask.id == mask_name].encoding
            self.mask = self.enc2mask()
        else:
            self.mask = None

    def __len__(self):
        return len(self.positions)

    def get_positions(self):
        top_x = np.arange(
            0,
            self.orig_size[0],  # +self.tile_size,
            int(self.tile_size / self.overlap_factor),
        )
        top_y = np.arange(
            0,
            self.orig_size[1],  # +self.tile_size,
            int(self.tile_size / self.overlap_factor),
        )
        starting_positions = []
        for x in top_x:
            right_space = self.orig_size[0] - (x + self.tile_size)
            if right_space > 0:
                boundaries_x = (x, x + self.tile_size)
            else:
                boundaries_x = (x + right_space, x + right_space + self.tile_size)

            for y in top_y:
                down_space = self.orig_size[1] - (y + self.tile_size)
                if down_space > 0:
                    boundaries_y = (y, y + self.tile_size)
                else:
                    boundaries_y = (y + down_space, y + down_space + self.tile_size)
                starting_positions.append((boundaries_x, boundaries_y))
        return starting_positions

    def enc2mask(self):
        msk = np.zeros(self.orig_size[0] * self.orig_size[1], dtype=np.uint8)
        for m, enc in enumerate(self.msk_encoding):
            if isinstance(enc, np.float) and np.isnan(enc):
                continue
            enc_split = enc.split()
            for i in range(len(enc_split) // 2):
                start = int(enc_split[2 * i]) - 1
                length = int(enc_split[2 * i + 1])
                msk[start: start + length] = 1 + m
        return msk.reshape((self.orig_size[1], self.orig_size[0])).T > 0

    def __getitem__(self, idx):
        pos_x, pos_y = self.positions[idx]
        img = self.original_img.read(
            [1, 2, 3],
            window=Window.from_slices((pos_x[0], pos_x[1]), (pos_y[0], pos_y[1])),
        )
        img = np.moveaxis(img, 0, -1)
        # down scale to tile size
        img = cv2.resize(
            img, (self.raw_tile_size, self.raw_tile_size), interpolation=cv2.INTER_AREA
        )

        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]
        return img
