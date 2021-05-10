import cv2
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from params import MEAN, STD


def disk(radius, alias_blur=0.1, dtype=np.float32):
    """
    From https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    and https://github.com/albumentations-team/albumentations/issues/477
    """
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


class DefocusBlur(ImageOnlyTransform):
    """
    From https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
    and https://github.com/albumentations-team/albumentations/issues/477
    """
    def __init__(
        self,
        severity=1,
        always_apply=False,
        p=1.0,
    ):
        super(DefocusBlur, self).__init__(always_apply, p)
        self.severity = severity
        self.radius, self.blur = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][
            self.severity - 1
        ]

    def apply(self, image, **params):
        image = np.array(image) / 255.0
        kernel = disk(radius=self.radius, alias_blur=self.blur)
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(image[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))
        return np.clip(channels, 0, 1) * 255

    def get_transform_init_args_names(self):
        return "severty"


def blur_transforms(p=0.5, blur_limit=5, gaussian_limit=(5, 7), severity=1):
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
            DefocusBlur(severity=severity, always_apply=True),
            albu.MotionBlur(blur_limit=blur_limit, always_apply=True),
            albu.GaussianBlur(blur_limit=gaussian_limit, always_apply=True),
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
                        brightness_limit=0.1,  # 0.3
                        contrast_limit=0.1,  # 0.3
                        p=1,
                    ),
                ]
            ),
            albu.RGBShift(
                r_shift_limit=30,
                g_shift_limit=0,
                b_shift_limit=30,
                p=1,
            ),
            albu.HueSaturationValue(
                hue_shift_limit=30,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=1,
            ),
            albu.ColorJitter(
                brightness=0.3,  # 0.3
                contrast=0.3,  # 0.3
                saturation=0.3,
                hue=0.05,
                p=1,
            ),
        ],
        p=p,
    )


def deformation_transform(p=0.5):
    """
    Applies ElasticTransform, GridDistortion or OpticalDistortion with a probability p.

    Args:
        p (float, optional): probability. Defaults to 0.5.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.OneOf(
        [
            albu.ElasticTransform(
                alpha=1,
                sigma=25,
                alpha_affine=25,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                always_apply=True,
            ),
            albu.GridDistortion(always_apply=True),
            albu.OpticalDistortion(distort_limit=1, shift_limit=0.2, always_apply=True),
        ],
        p=p,
    )


def center_crop(size):
    """
    Applies a padded center crop.

    Args:
        size (int): Crop size.

    Returns:
        albumentation transforms: transforms.
    """
    if size is None:  # disable cropping
        p = 0
    else:  # always crop
        p = 1

    return albu.Compose(
        [
            albu.PadIfNeeded(size, size, p=p, border_mode=cv2.BORDER_CONSTANT),
            albu.CenterCrop(size, size, p=p),
        ],
        p=1,
    )


def HE_preprocess(augment=True, visualize=False, mean=MEAN, std=STD, size=None):
    """
    Returns transformations for the H&E images.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.
        mean (np array, optional): Mean for normalization. Defaults to MEAN.
        std (np array, optional): Standard deviation for normalization. Defaults to STD.

    Returns:
        albumentation transforms: transforms.
    """
    if visualize:
        normalizer = albu.Compose(
            [
                center_crop(size),
                albu.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                ToTensorV2(),
            ],
            p=1,
        )
    else:
        normalizer = albu.Compose(
            [center_crop(size), albu.Normalize(mean=mean, std=std), ToTensorV2()], p=1
        )

    if augment:
        return albu.Compose(
            [
                albu.VerticalFlip(p=0.5),
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(
                    scale_limit=0.1,  # 0
                    shift_limit=0.1,  # 0.05
                    rotate_limit=90,
                    p=0.5,
                ),
                deformation_transform(p=0.5),
                color_transforms(p=0.5),
                blur_transforms(p=0.5),
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
        mean (np array, optional): Mean for normalization. Defaults to MEAN.
        std (np array, optional): Standard deviation for normalization. Defaults to STD.

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
