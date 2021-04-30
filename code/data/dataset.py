import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from torch.utils.data import Dataset

from params import DATA_PATH, LAB_STATS  # noqa
from utils.rle import enc2mask
from data.transforms import lab_normalization  # noqa
from skimage.morphology import convex_hull_image


def load_image(img_path, full_size=True, reduce_factor=4):
    """
    Load image and make sure sizes matches df_info
    """
    df_info = pd.read_csv(DATA_PATH + "HuBMAP-20-dataset_information.csv")
    image_fname = img_path.rsplit("/", -1)[-1]

    img = tiff.imread(img_path).squeeze()

    try:
        W = int(df_info[df_info.image_file == image_fname]["width_pixels"])
        H = int(df_info[df_info.image_file == image_fname]["height_pixels"])

        if not full_size:
            W = W // reduce_factor
            H = H // reduce_factor

        channel_pos = np.argwhere(np.array(img.shape) == 3)[0][0]
        W_pos = np.argwhere(np.array(img.shape) == W)[0][0]
        H_pos = np.argwhere(np.array(img.shape) == H)[0][0]

        img = np.moveaxis(img, (H_pos, W_pos, channel_pos), (0, 1, 2))

    except TypeError:  # image_fname not in df
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

    return img


def simple_load(img_path):
    """
    Load image and make sure channels in last position
    """
    img = tiff.imread(img_path).squeeze()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img


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
        - sampling_mode:
            - centered : center of image should contain glomuleri
            - convhull : will use conv_hull only
            - random : any
            - visible : tile should have at least 2K pixels as glomuleri
        - use_external : None or float of probability
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
        sampling_mode="convhull",
        use_external=0,
        oof_folder=None,
        df_rle_test=None,
        test_path="../input/test/",
        use_pl=0,
    ):
        """"""
        # Hard coded external path for now
        self.ext_img_path = "../input/external_data/images_1024/"
        self.ext_msk_path = "../input/external_data/masks_1024/"
        self.external_names = [p.name for p in Path(self.ext_img_path).glob("*")]
        self.use_external = use_external
        self.use_pl = use_pl

        self.sampling_mode = sampling_mode
        assert self.sampling_mode in ["centered", "convhull", "random", "visible"]
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
            img_size = img.shape

            rle = df_rle.loc[df_rle.id == img_name, "encoding"]
            mask = enc2mask(rle, (orig_img_size[1], orig_img_size[0]))

            if self.sampling_mode == "convhull":
                conv_hull = convex_hull_image(mask)
                self.conv_hulls.append(conv_hull)
            self.imgs.append(img)
            self.image_sizes.append(img_size)
            self.masks.append(mask)

        self.images_areas = [h * w for (h, w, c) in self.image_sizes]

        # Deal with fold inside this to avoid reloading for each fold (time consuming)
        self.update_fold_nb(self.fold_nb)
        self.train(True)

        # Load in memory all oof preds. Expects them to be of the same reduce factor for now
        if oof_folder is not None:
            config = json.load(open(oof_folder + "config.json", "r"))
            assert (
                config["reduce_factor"] == reduce_factor
            ), "Different prediction reduce factor"

            self.oof_preds = []
            for img_name in self.train_img_names:
                self.oof_preds.append(np.load(oof_folder + f"pred_{img_name}.npy"))
        else:
            self.oof_preds = None

        # Test data
        self.imgs_test = []
        self.image_sizes_test = []
        self.masks_test = []
        self.conv_hulls_test = []

        if df_rle_test is not None:
            for img_name in df_rle_test.id.values:
                img = simple_load(os.path.join(test_path, img_name + ".tiff"))
                orig_img_size = img.shape

                rle = df_rle_test.loc[df_rle_test.id == img_name, "predicted"]
                mask = enc2mask(rle, (orig_img_size[1], orig_img_size[0]))

                img = cv2.resize(
                    img,
                    (img.shape[1] // reduce_factor, img.shape[0] // reduce_factor),
                    interpolation=cv2.INTER_AREA,
                )
                mask = cv2.resize(
                    mask,
                    (mask.shape[1] // reduce_factor, mask.shape[0] // reduce_factor),
                    interpolation=cv2.INTER_NEAREST,
                )

                self.imgs_test.append(img)
                self.image_sizes_test.append(img.shape)
                self.masks_test.append(mask)

                if self.sampling_mode == "convhull":
                    conv_hull = convex_hull_image(mask)
                    self.conv_hulls_test.append(conv_hull)

        self.sampling_probs_test = np.array([np.sum(mask) for mask in self.masks_test])
        self.sampling_probs_test = self.sampling_probs_test / np.sum(self.sampling_probs_test)

    def train(self, is_train):
        # Switch to train mode
        if is_train:
            self.used_img_idx = self.train_img_idx
            self.transforms = self.train_transfo
            self.sampling_thresh = self.on_spot_sampling
            self.sampling_probs = np.array(
                [
                    np.sum(self.masks[idx])
                    for idx, area in enumerate(self.masks)
                    if idx in self.used_img_idx
                ]
            )
            self.sampling_probs = self.sampling_probs / np.sum(self.sampling_probs)
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

    def accept_tile_policy(self, image_nb, x1, x2, y1, y2, masks, convhulls):
        if self.sampling_thresh == 0:
            return True

        if self.sampling_mode == "centered":
            if masks[image_nb][int((x1 + x2) / 2), int((y1 + y2) / 2)]:
                return True
        elif self.sampling_mode == "convhull":
            if convhulls[image_nb][int((x1 + x2) / 2), int((y1 + y2) / 2)]:
                return True
        elif self.sampling_mode == "visible":
            if masks[image_nb][x1:x2, y1:y2].sum() > 2000:
                return True
        elif self.sampling_mode == "random":
            return True

        should_keep = np.random.rand()
        if should_keep > self.sampling_thresh:
            return True
        else:
            return False

    def get_external_item(self):
        """
        Randomly selects an external image
        """
        img_name = np.random.choice(self.external_names)
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.ext_img_path, img_name)), cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            os.path.join(self.ext_msk_path, img_name), cv2.IMREAD_GRAYSCALE
        )

        h, w, _ = img.shape
        img = cv2.resize(
            img,
            (h // self.reduce_factor, w // self.reduce_factor),
            interpolation=cv2.INTER_AREA,
        )
        mask = cv2.resize(
            mask,
            (h // self.reduce_factor, w // self.reduce_factor),
            interpolation=cv2.INTER_NEAREST,
        )

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if self.oof_preds is not None:  # not actually available
            mask = mask.float()
            oof_pred = mask.clone()
        else:
            oof_pred = 0

        return img, mask, oof_pred

    def getitem_pl(self):
        image_nb = np.random.choice(
            range(len(self.imgs_test)), replace=True, p=self.sampling_probs_test
        )

        img_dim = self.image_sizes_test[image_nb]
        is_point_ok = False
        while not is_point_ok:
            # Sample random point
            x1 = np.random.randint(img_dim[0] - self.before_crop_size)
            x2 = x1 + self.before_crop_size

            y1 = np.random.randint(img_dim[1] - self.before_crop_size)
            y2 = y1 + self.before_crop_size

            is_point_ok = self.accept_tile_policy(
                image_nb, x1, x2, y1, y2, self.masks_test, self.conv_hulls_test
            )

        img = self.imgs_test[image_nb][x1:x2, y1:y2]
        mask = self.masks_test[image_nb][x1:x2, y1:y2]

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if self.oof_preds is not None:  # not actually available
            mask = mask.float()
            oof_pred = mask.clone()
        else:
            oof_pred = 0

        return img, mask, oof_pred

    def getitem_normal(self, image_nb):
        img_dim = self.image_sizes[image_nb]
        is_point_ok = False

        while not is_point_ok:
            # Sample random point
            x1 = np.random.randint(img_dim[0] - self.before_crop_size)
            x2 = x1 + self.before_crop_size

            y1 = np.random.randint(img_dim[1] - self.before_crop_size)
            y2 = y1 + self.before_crop_size

            is_point_ok = self.accept_tile_policy(
                image_nb, x1, x2, y1, y2, self.masks, self.conv_hulls
            )

        img = self.imgs[image_nb][x1:x2, y1:y2]
        mask = self.masks[image_nb][x1:x2, y1:y2]

        if self.oof_preds is not None:
            oof_pred = self.oof_preds[image_nb][x1:x2, y1:y2].astype(np.float32)
            mask = np.array([mask, oof_pred]).transpose(1, 2, 0)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if self.oof_preds is not None:
            mask, oof_pred = mask[:, :, 0], mask[:, :, 1]
        else:
            oof_pred = 0

        return img, mask, oof_pred

    def __getitem__(self, idx):

        if self.sampling_thresh == 0:  # take uniformly from images for validation
            image_nb = self.used_img_idx[idx % len(self.used_img_idx)]
        else:
            if np.random.rand() < self.use_external:
                return self.get_external_item()

            if np.random.rand() < self.use_pl:
                return self.getitem_pl()

            image_nb = self.used_img_idx[
                np.random.choice(
                    range(len(self.used_img_idx)), replace=True, p=self.sampling_probs
                )
            ]

        return self.getitem_normal(image_nb)


class TileClsDataset(Dataset):
    """
    Dataset to read from tiled images.
    """

    def __init__(self, images, root="", transforms=None):
        """
        Args:
            df (pandas dataframe): file_names.
            img_dir (str, optional): Images directory. Defaults to "".
            mask_dir (str, optional): Masks directory. Defaults to "".
            transforms (albumentation transforms, optional) : Transforms. Defaults to None.
        """
        self.df = images
        self.tiles = [root + p for p in os.listdir(root) if p.split("_")[0] in images]

        self.transforms = transforms

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = np.load(self.tiles[idx])

        img, mask = tile[:, :, :-2], tile[:, :, -2:]

        img = (img * 255).astype(np.uint8)

        target = mask[:, :, -1][mask.shape[0] // 2, mask.shape[1] // 2]

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = torch.cat([img, mask[:, :, 0].unsqueeze(0)], 0)
        mask = mask[:, :, -1]

        return img, mask, target
