import os
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff

from torch.utils.data import Dataset

from params import DATA_PATH, LAB_STATS  # noqa
from utils.rle import enc2mask
from data.transforms import lab_normalization  # noqa
from skimage.morphology import convex_hull_image


def load_image(img_path, full_size=True):
    """
    Load image and make sure sizes matches df_info
    """
    df_info = pd.read_csv(DATA_PATH + "HuBMAP-20-dataset_information.csv")
    image_fname = img_path.rsplit("/", -1)[-1]
    W = int(df_info[df_info.image_file == image_fname]["width_pixels"])
    H = int(df_info[df_info.image_file == image_fname]["height_pixels"])

    if not full_size:
        W = W // 4
        H = H // 4

    img = tiff.imread(img_path).squeeze()

    channel_pos = np.argwhere(np.array(img.shape) == 3)[0][0]
    W_pos = np.argwhere(np.array(img.shape) == W)[0][0]
    H_pos = np.argwhere(np.array(img.shape) == H)[0][0]

    img = np.moveaxis(img, (H_pos, W_pos, channel_pos), (0, 1, 2))
    return img


def simple_load(img_path):
    """
    Load image and make sure channels in last position
    """
    img = tiff.imread(img_path).squeeze()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
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

        # mean, std = LAB_STATS[tile_name.split("_")[0]]
        # img = lab_normalization(img)  # , mean=mean, std=std)

        mask = cv2.imread(os.path.join(self.mask_dir, tile_name), cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        return img, mask


class InferenceDataset(Dataset):
    def __init__(
        self,
        original_img_path,
        rle=None,
        overlap_factor=1,
        tile_size=256,
        reduce_factor=4,
        transforms=None,
    ):
        self.original_img = simple_load(original_img_path)
        self.orig_size = self.original_img.shape

        # self.original_img = lab_normalization(self.original_img)

        self.raw_tile_size = tile_size
        self.reduce_factor = reduce_factor
        self.tile_size = tile_size * reduce_factor

        self.overlap_factor = overlap_factor

        self.positions = self.get_positions()

        self.transforms = transforms

        if rle is not None:
            self.mask = enc2mask(rle, (self.orig_size[1], self.orig_size[0])) > 0
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

    def __getitem__(self, idx):
        pos_x, pos_y = self.positions[idx]
        img = self.original_img[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1], :]

        # img = lab_normalization(img)

        # down scale to tile size
        if self.reduce_factor > 1:
            img = cv2.resize(
                img,
                (self.raw_tile_size, self.raw_tile_size),
                interpolation=cv2.INTER_AREA,
            )

        if self.transforms:
            img = self.transforms(image=img)["image"]

        pos = np.array([pos_x[0], pos_x[1], pos_y[0], pos_y[1]])

        return img, pos


class InMemoryTrainDataset(Dataset):
    """
    In Memory Dataset, which allows to smart tile sampling of any size and reduction factor.

    Everything must be loaded into RAM once (takes about 4 minutes).

    The self.train method allows to easily switch from training/validation mode.
    It changes the image used for tiles and disable transformation.

    The self.update_fold_nb allows to change fold without reloading everything

    Params
    ------
        - train_img_names : images to use for CV
        - df_rle : training pandas df with rle encoded masks
        - train_tile_size : int, size of images to input model
        - reduce_factor : reduction factor to apply before entering the model
        - transforms : albu transfo, augmentation scheme
        - train_path : path to folder with the original tiff images
        - iter_per_epoch : int, number of tiles that constitue an epoch
        - on_sport_samping : float between 0 and 1, probability of rejection outside conv hull
                             (1. means only intersting tiles, 0 purely random tiles)
        - fold_nb : which fold are we considereing at the moment (everything is in RAM)
    """

    def __init__(
        self,
        train_img_names,
        df_rle,
        train_tile_size=256,
        reduce_factor=4,
        train_transfo=None,
        valid_transfo=None,
        train_path="../input/train/",
        iter_per_epoch=1000,
        on_spot_sampling=0.9,
        fold_nb=0,
    ):
        self.iter_per_epoch = iter_per_epoch
        self.train_tile_size = train_tile_size
        # Allows to make heavier transfo without artefact by center cropping
        self.before_crop_size = int(1.5 * self.train_tile_size)

        self.reduce_factor = reduce_factor
        self.tile_size = train_tile_size * reduce_factor
        self.on_spot_sampling = on_spot_sampling
        self.train_transfo = train_transfo
        self.valid_transfo = valid_transfo

        self.train_img_names = train_img_names
        self.fold_nb = fold_nb

        self.imgs = []
        self.image_sizes = []
        self.masks = []
        self.conv_hulls = []

        # Load in memory all resized images, masks and conv_hulls
        for img_name in self.train_img_names:
            img = simple_load(os.path.join(train_path, img_name + ".tiff"))
            orig_img_size = img.shape

            # img = cv2.resize(
            #     img,
            #     (
            #         orig_img_size[0] // self.reduce_factor,
            #         orig_img_size[1] // self.reduce_factor,
            #     ),
            #     interpolation=cv2.INTER_AREA,
            # )
            img_size = img.shape

            rle = df_rle.loc[df_rle.id == img_name, "encoding"]
            mask = enc2mask(rle, (orig_img_size[1], orig_img_size[0]))

            # mask = cv2.resize(
            #     mask,
            #     (
            #         orig_img_size[0] // self.reduce_factor,
            #         orig_img_size[1] // self.reduce_factor,
            #     ),
            #     interpolation=cv2.INTER_NEAREST,
            # )

            conv_hull = convex_hull_image(mask)
            self.imgs.append(img)
            self.image_sizes.append(img_size)
            self.masks.append(mask)
            self.conv_hulls.append(conv_hull)

        # Deal with fold inside this to avoid reloading for each fold (time consuming)
        self.update_fold_nb(self.fold_nb)
        self.train(True)

    def train(self, is_train):
        # Switch to train mode
        if is_train:
            self.used_img_idx = self.train_img_idx
            self.transforms = self.train_transfo
            self.sampling_thresh = self.on_spot_sampling
        else:
            # switch tile, disable transformation and on_spot_sampling (should we?)
            self.used_img_idx = self.valid_img_idx
            self.transforms = self.valid_transfo
            self.sampling_thresh = 0

    def update_fold_nb(self, fold_nb):
        """
        Allows switching fold without reloading everything
        """
        # 5 fold cv hard coded
        self.train_img_idx = [
            tile_nb
            for tile_nb in range(len(self.train_img_names))
            if tile_nb % 5 != fold_nb
        ]
        self.valid_img_idx = [
            tile_nb
            for tile_nb in range(len(self.train_img_names))
            if tile_nb % 5 == fold_nb
        ]
        self.train_set = [self.train_img_names[idx] for idx in self.train_img_idx]
        self.valid_set = [self.train_img_names[idx] for idx in self.valid_img_idx]

    def __len__(self):
        return self.iter_per_epoch

    def __getitem__(self, idx):

        image_nb = self.used_img_idx[idx % len(self.used_img_idx)]
        img_dim = self.image_sizes[image_nb]

        is_point_ok = False

        while not is_point_ok:
            # Sample random point
            x1 = np.random.randint(img_dim[0] - self.before_crop_size)
            x2 = x1 + self.before_crop_size

            y1 = np.random.randint(img_dim[1] - self.before_crop_size)
            y2 = y1 + self.before_crop_size

            if self.conv_hulls[image_nb][int((x1 + x2) / 2), int((y1 + y2) / 2)]:
                # this is inside the convhull so keep it
                is_point_ok = True
            else:
                # Allow only a fraction of images to be outisde
                should_keep = np.random.rand()
                if should_keep > self.sampling_thresh:
                    is_point_ok = True

        img = self.imgs[image_nb][x1:x2, y1:y2]
        mask = self.masks[image_nb][x1:x2, y1:y2]

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask
