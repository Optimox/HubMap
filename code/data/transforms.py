import cv2
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2

from params import MEAN, STD


# MEAN_LAB = np.array([52.90164, 29.290276, -29.144949])
# STD_LAB = np.array([12.994737, 10.004983, 8.370835])
MEAN_LAB = np.array([48.16595, 34.401302, -31.495922])
STD_LAB = np.array([18.03903, 13.677521, 11.873212])


def get_lab_stats(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    mean = lab_img.mean((0, 1))
    std = lab_img.std((0, 1))

    return mean, std


def lab_normalization(img, mean=None, std=None):
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    if mean is None:
        mean = lab_img.mean((0, 1))
    if std is None:
        std = lab_img.std((0, 1))

    lab_img = (lab_img - mean) / (std + 1e-6) * STD_LAB + MEAN_LAB
    lab_img = lab_img.astype(np.float32)

    img_n = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)
    return (img_n * 255).astype(np.uint8)


def blur_transforms(p=0.5, blur_limit=5):
    """
    Applies MotionBlur or GaussianBlur random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.
        blur_limit (int, optional): Blur intensity limit. Defaults to 5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.MotionBlur(blur_limit=blur_limit, always_apply=True),
            albu.GaussianBlur(blur_limit=blur_limit, always_apply=True),
        ],
        p=p,
    )


def noise_transforms(p=0.5):
    """
    Applies GaussNoise or RandomFog random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.GaussNoise(var_limit=(1.0, 50.0), always_apply=True),
            albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.25, always_apply=True),
        ],
        p=p,
    )


def color_transforms(p=0.5):
    """
    Applies RandomGamma or RandomBrightnessContrast random with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.Compose(
                [
                    albu.RandomGamma(gamma_limit=(80, 120), p=1),
                    albu.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=1,
                    ),
                ]
            ),
            albu.RGBShift(
                r_shift_limit=10,
                g_shift_limit=0,
                b_shift_limit=10,
                p=1,
            ),
            albu.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=30,
                p=1,
            ),
            albu.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=1,
            ),
        ],
        p=p,
    )


def distortion_transforms(p=0.5):
    """
    Applies ElasticTransform with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """

    return albu.OneOf(
        [
            albu.ElasticTransform(
                alpha=20,
                sigma=5,
                alpha_affine=10,
                p=1,
            ),
        ],
        p=p,
    )


def HE_preprocess(augment=True, visualize=False, mean=MEAN, std=STD):
    """
    Returns transformations for the H&E images.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.
        mean ([type], optional): Mean for normalization. Defaults to MEAN.
        std ([type], optional): Standard deviation for normalization. Defaults to STD.

    Returns:
        albumentation transforms: transforms.
    """
    if visualize:
        normalizer = albu.Compose(
            [albu.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), ToTensorV2()], p=1
        )
    else:
        normalizer = albu.Compose(
            [albu.Normalize(mean=mean, std=std), ToTensorV2()], p=1
        )

    if augment:
        return albu.Compose(
            [
                albu.VerticalFlip(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.1, shift_limit=0.1, rotate_limit=90, p=0.5
                ),
                color_transforms(p=0.5),
                # distortion_transforms(p=0.5),
                normalizer,
            ]
        )
    else:
        return normalizer


def HE_preprocess_test(augment=False, visualize=False, mean=MEAN, std=STD):
    """
    Returns transformations for the H&E images.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.
        mean ([type], optional): Mean for normalization. Defaults to MEAN.
        std ([type], optional): Standard deviation for normalization. Defaults to STD.

    Returns:
        albumentation transforms: transforms.
    """
    if visualize:
        normalizer = albu.Compose(
            [albu.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), ToTensorV2()], p=1
        )
    else:
        normalizer = albu.Compose(
            [albu.Normalize(mean=mean, std=std), ToTensorV2()], p=1
        )

    if augment:
        raise NotImplementedError

    return normalizer
