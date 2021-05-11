import os
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from torch.utils.data import Dataset

from params import DATA_PATH, DATA_PATH_EXTRA
from utils.rle import enc2mask


def load_image(img_path, full_size=True, reduce_factor=4):
    """
    Loads an image, and makes sure the axis are in the same position as specified in the metadata.

    Args:
        img_path (str): Path to the image.
        full_size (bool, optional): Whether the image was downsized. Defaults to True.
        reduce_factor (int, optional): How much the image was downsized. Defaults to 4.

    Returns:
        np array [H x W x 3]: Image.
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
    Loads an image, and makes sure the channel axis is in last position.

    Args:
        img_path (str): Path to the image.

    Returns:
        np array [H x W x 3]: Image.
    """
    img = tiff.imread(img_path).squeeze()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img


class InferenceDataset(Dataset):
    """
    Dataset for inference.
    """
    def __init__(
        self,
        original_img_path,
        rle=None,
        overlap_factor=1,
        tile_size=256,
        reduce_factor=4,
        transforms=None,
    ):
        """
        Constructor.

        Args:
            original_img_path (str]): [description]
            rle (str or None, optional): rle encoding. Defaults to None.
            overlap_factor (int, optional): Overlap factor. Defaults to 1.
            tile_size (int, optional): Tile size. Defaults to 256.
            reduce_factor (int, optional): Reduce factor. Defaults to 4.
            transforms (albu transforms or None, optional): Transforms. Defaults to None.
        """
        self.original_img = simple_load(original_img_path)
        self.orig_size = self.original_img.shape

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
        """
        Computes positions of the tiles.

        Returns:
            np array: Tile starting positions.
        """
        top_x = np.arange(
            0,
            self.orig_size[0],
            int(self.tile_size / self.overlap_factor),
        )
        top_y = np.arange(
            0,
            self.orig_size[1],
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
    Everything must be loaded into RAM once.
    The self.train method allows to easily switch from training/validation mode.
    It changes the image used for tiles and disable transformation.
    The self.update_fold_nb allows to change fold without reloading everything.
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
        use_pl=0,
        pl_path=None,
        test_path="../input/test/",
        df_rle_extra=None,
        use_external=0,
    ):
        """
        Constructor.

        Args:
            train_img_names (list of str): Training data.
            df_rle (pandas dataframe): Ground truths as rles.
            train_tile_size (int, optional): Training tile size. Defaults to 256.
            reduce_factor (int, optional): Downsizing factor. Defaults to 4.
            train_transfo (albu transforms, optional): Training augmentations. Defaults to None.
            valid_transfo (albu transforms, optional): Validation transforms. Defaults to None.
            train_path (str, optional): Path to the training images. Defaults to "../input/train/".
            iter_per_epoch (int, optional): Number of images to sample per epoch. Defaults to 1000.
            on_spot_sampling (float, optional): Proba for the sampling strategy. Defaults to 0.9.
            fold_nb (int, optional): Fold number. Defaults to 0.
            use_pl (int, optional): Sampling proportion for pseudo-labeled data. Defaults to 0.
            pl_path (str or None, optional): Path to pseudo labels. Defaults to None.
            test_path (str, optional): Path to test images. Defaults to "../input/test/".
            df_rle_extra (pandas dataframe, optional): Ext ground truths as rles. Defaults to None.
            use_external (int, optional): Sampling proportion for external data. Defaults to 0.
        """
        self.use_external = use_external if df_rle_extra is not None else 0
        self.use_pl = use_pl if test_path is not None else 0

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

        # Load in memory all resized images & masks
        for img_name in self.train_img_names:
            img = simple_load(os.path.join(train_path, img_name + ".tiff"))
            orig_img_size = img.shape
            img_size = img.shape

            if isinstance(df_rle, list):  # 2 = fc, 1 = not fc
                rle = df_rle[0].loc[df_rle[0].id == img_name, "encoding"]
                mask = enc2mask(rle, (orig_img_size[1], orig_img_size[0]))
                rle = df_rle[1].loc[df_rle[1].id == img_name, "encoding"]
                mask += 2 * enc2mask(rle, (orig_img_size[1], orig_img_size[0]))
                # mask = np.clip(mask, 0, 2)
                self.num_classes = 2
            else:
                rle = df_rle.loc[df_rle.id == img_name, "encoding"]
                mask = enc2mask(rle, (orig_img_size[1], orig_img_size[0]))
                self.num_classes = 1

            self.imgs.append(img)
            self.image_sizes.append(img_size)
            self.masks.append(mask)

        self.images_areas = [h * w for (h, w, c) in self.image_sizes]

        # Test data
        self.imgs_test = []
        self.image_sizes_test = []
        self.masks_test = []
        self.pl_path = pl_path
        if pl_path is not None:
            self.img_names_test = [p[:-5] for p in os.listdir(test_path)]

            for img_name in self.img_names_test:
                img = simple_load(os.path.join(test_path, img_name + ".tiff"))
                self.imgs_test.append(img)
                self.image_sizes_test.append(img.shape)

        # Extra data
        self.imgs_extra = []
        self.image_sizes_extra = []
        self.masks_extra = []

        if df_rle_extra is not None:
            df_extra_ph = df_rle_extra[0] if isinstance(df_rle_extra, list) else df_rle_extra[0]
            for img_name in df_extra_ph.id.values:
                img = tiff.imread(DATA_PATH_EXTRA + img_name + ".tiff").squeeze()
                orig_img_size = img.shape

                if isinstance(df_rle_extra, list):  # 2 = fc, 1 = not fc
                    rle = df_rle_extra[0].loc[df_rle_extra[0].id == img_name, "encoding"]
                    mask = enc2mask(rle, (orig_img_size[1], orig_img_size[0]))
                    rle = df_rle_extra[1].loc[df_rle_extra[1].id == img_name, "encoding"]
                    mask += 2 * enc2mask(rle, (orig_img_size[1], orig_img_size[0]))
                else:
                    rle = df_rle_extra.loc[df_rle_extra.id == img_name, "encoding"]
                    mask = enc2mask(rle, (orig_img_size[1], orig_img_size[0]))

                if reduce_factor > 2:  # downsize if necessary
                    img = cv2.resize(
                        img,
                        (img.shape[1] // reduce_factor * 2, img.shape[0] // reduce_factor * 2),
                        interpolation=cv2.INTER_AREA,
                    )
                    mask = cv2.resize(
                        mask,
                        (mask.shape[1] // reduce_factor * 2, mask.shape[0] // reduce_factor * 2),
                        interpolation=cv2.INTER_NEAREST,
                    )

                self.imgs_extra.append(img)
                self.image_sizes_extra.append(img.shape)
                self.masks_extra.append(mask)

        self.sampling_probs_extra = np.array([np.sum(mask == 1) for mask in self.masks_extra])
        self.sampling_probs_extra = self.sampling_probs_extra / np.sum(self.sampling_probs_extra)

        # Deal with fold inside this to avoid reloading for each fold (time consuming)
        self.update_fold_nb(self.fold_nb, load=False)
        self.train(True)

    def train(self, is_train):
        """
        Switches the dataset between train and validation mode.

        Args:
            is_train (bool): True to switch to train and False to switch to val.
        """
        if is_train:  # Switch to train mode
            self.used_img_idx = self.train_img_idx
            self.transforms = self.train_transfo
            self.sampling_thresh = self.on_spot_sampling
            self.sampling_probs = np.array(
                [
                    np.sum(self.masks[idx] == 1)
                    for idx, area in enumerate(self.masks)
                    if idx in self.used_img_idx
                ]
            )
            self.sampling_probs = self.sampling_probs / np.sum(self.sampling_probs)
        else:  # switch tile, disable transformation and on_spot_sampling (should we?)
            self.used_img_idx = self.valid_img_idx
            self.transforms = self.valid_transfo
            self.sampling_thresh = 0

    def update_fold_nb(self, fold_nb, load=True):
        """
        Switches the fold. Updates train and val images,
        Updates pseudo-labels if load is True.

        Args:
            fold_nb (int): Fold number.
            load (bool, optional): Whether to load pseudo labels. Defaults to True.
        """
        # 5 fold cv hard coded
        self.fold_nb = fold_nb
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

        # Update PL masks
        if self.pl_path is not None and load:
            self.masks_test = [
                np.load(self.pl_path + f"pred_{name}_{fold_nb}.npy") for name in self.img_names_test
            ]
            self.sampling_probs_test = np.array([np.sum(mask > 0.5) for mask in self.masks_test])
            self.sampling_probs_test = self.sampling_probs_test / np.sum(self.sampling_probs_test)

    def __len__(self):
        return self.iter_per_epoch

    def accept_tile_policy_normal(self, image_nb, x1, x2, y1, y2):
        """
        Tile acceptation policy for the training data.
        We accept images that have a glomeruli at their middle.
        We allow for exceptions with a probability self.sampling_thresh

        Args:
            image_nb (int): Image number.
            x1 (int): Tile coordinate (top left x).
            x2 (int): Tile coordinate (bottom left x).
            y1 (int): Tile coordinate (top left y).
            y2 (int): Tile coordinate (bottom left y).

        Returns:
            bool: Whether the tile is accepted.
        """
        if self.sampling_thresh == 0:
            return True

        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        m = 10
        if self.masks[image_nb][mid_x - m: mid_x + m, mid_y - m: mid_y + m].max() > 0:
            return True

        should_keep = np.random.rand()
        if should_keep > self.sampling_thresh:
            return True
        else:
            return False

    def accept_tile_policy_pl(self, image_nb, x1, x2, y1, y2):
        """
        Tile acceptation policy for the pseudo-labeled data.
        We accept images that have >0.9 score in them.
        We allow for exceptions with a probability self.sampling_thresh.

        Args:
            image_nb (int): Image number.
            x1 (int): Tile coordinate (top left x).
            x2 (int): Tile coordinate (bottom left x).
            y1 (int): Tile coordinate (top left y).
            y2 (int): Tile coordinate (bottom left y).

        Returns:
            bool: Whether the tile is accepted.
        """
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        m = 250
        if self.masks_test[image_nb][mid_x - m: mid_x + m, mid_y - m: mid_y + m].max() > 0.9:
            return True


        should_keep = np.random.rand()
        if should_keep > self.sampling_thresh:
            return True
        else:
            return False

    def accept_tile_policy_ext(self, image_nb, x1, x2, y1, y2):
        """
        Tile acceptation policy for the extra data.
        We accept images that have a glomeruli in them.
        We allow for exceptions with a probability self.sampling_thresh.

        Args:
            image_nb (int): Image number.
            x1 (int): Tile coordinate (top left x).
            x2 (int): Tile coordinate (bottom left x).
            y1 (int): Tile coordinate (top left y).
            y2 (int): Tile coordinate (bottom left y).

        Returns:
            bool: Whether the tile is accepted.
        """
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        m = 250
        if self.masks_extra[image_nb][mid_x - m: mid_x + m, mid_y - m: mid_y + m].max() > 0:
            return True

        should_keep = np.random.rand()
        if should_keep > self.sampling_thresh:
            return True
        else:
            return False

    def getitem_normal(self, image_nb):
        """
        Returns an item from the training image image_nb.
        """
        img_dim = self.image_sizes[image_nb]
        is_point_ok = False

        while not is_point_ok:
            # Sample random point
            x1 = np.random.randint(img_dim[0] - self.before_crop_size)
            x2 = x1 + self.before_crop_size

            y1 = np.random.randint(img_dim[1] - self.before_crop_size)
            y2 = y1 + self.before_crop_size

            is_point_ok = self.accept_tile_policy_normal(
                image_nb, x1, x2, y1, y2,
            )

        img = self.imgs[image_nb][x1:x2, y1:y2]
        mask = self.masks[image_nb][x1:x2, y1:y2]

        if self.num_classes == 2:
            mask = np.array([
                mask == 1,
                ((mask == 2) + (mask == 3)) > 0,
            ]).transpose(1, 2, 0).astype(np.uint8)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask.float(), 1

    def getitem_pl(self):
        """
        Returns an item from the pseudo-labels.
        """
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

            is_point_ok = self.accept_tile_policy_pl(image_nb, x1, x2, y1, y2)

        img = self.imgs_test[image_nb][x1:x2, y1:y2]
        mask = self.masks_test[image_nb][x1:x2, y1:y2]

        if self.num_classes == 2:
            mask = np.array([mask, mask]).transpose(1, 2, 0).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask.float(), 0

    def getitem_extra(self):
        """
        Returns an item from the external data.
        """
        image_nb = np.random.choice(
            range(len(self.imgs_extra)), replace=True, p=self.sampling_probs_extra
        )

        img_dim = self.image_sizes_extra[image_nb]
        is_point_ok = False
        while not is_point_ok:
            # Sample random point
            x1 = np.random.randint(img_dim[0] - self.before_crop_size)
            x2 = x1 + self.before_crop_size

            y1 = np.random.randint(img_dim[1] - self.before_crop_size)
            y2 = y1 + self.before_crop_size

            is_point_ok = self.accept_tile_policy_ext(image_nb, x1, x2, y1, y2)

        img = self.imgs_extra[image_nb][x1:x2, y1:y2]
        mask = self.masks_extra[image_nb][x1:x2, y1:y2]

        if self.num_classes == 2:
            mask = np.array([
                ((mask == 1) + (mask == 3)) > 0,
                ((mask == 2) + (mask == 3)) > 0
            ]).transpose(1, 2, 0).astype(np.uint8)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask.float(), 1

    def __getitem__(self, idx):
        """
        Returns an item. Randomly calls one of the specific getitem functions.

        Args:
            idx ([type]): [description]

        Returns:
            img (torch tensor [3 x H x W]): image.
            mask (torch tensor [num_classes x H x W]): mask.
            w (int): 0 if an item from the pseudo-labels is sampled, else 1.
        """

        if self.sampling_thresh == 0:  # take uniformly from images for validation
            image_nb = self.used_img_idx[idx % len(self.used_img_idx)]
        else:
            value = np.random.rand()

            if value < self.use_external:
                return self.getitem_extra()

            if self.use_external < value < self.use_external + self.use_pl:
                return self.getitem_pl()

            image_nb = self.used_img_idx[
                np.random.choice(
                    range(len(self.used_img_idx)), replace=True, p=self.sampling_probs
                )
            ]

        return self.getitem_normal(image_nb)
