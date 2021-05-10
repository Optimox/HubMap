import numpy as np
import pandas as pd

from training.predict import (
    predict_entire_mask_downscaled,
    predict_entire_mask,
    threshold_resize_torch,
)

from model_zoo.models import define_model

from data.dataset import InferenceDataset
from data.transforms import HE_preprocess_test
from utils.torch import load_model_weights
from params import TIFF_PATH_TEST, DATA_PATH, EXTRA_IMGS_SHAPES


def validate_inf_test(
    model,
    config,
    images,
    fold=0,
    log_folder=None,
    use_full_size=True,
    global_threshold=None,
    use_tta=False,
    save=False
):
    """
    Performs inference with a model on a list of  test images.

    Args:
        model (torch model): Segmentation model.
        config (Config): Parameters.
        images (list of strings): Image names.
        fold (int, optional): Fold index. Defaults to 0.
        log_folder (str or None, optional): Folder to save predictions to. Defaults to None.
        use_full_size (bool, optional): Whether to use full resolution images. Defaults to True.
        global_threshold (float, optional): Threshold for probabilities. Defaults to None.
        use_tta (bool, optional): Whether to use tta. Defaults to False.
        save (bool, optional): Whether to save predictions. Defaults to False.
    """
    df_info = pd.read_csv(DATA_PATH + "HuBMAP-20-dataset_information.csv")

    if use_full_size:
        root = TIFF_PATH_TEST
        reduce_factor = config.reduce_factor
    else:
        root = DATA_PATH + f"test_{config.reduce_factor}/"
        reduce_factor = 1

    for img in images:
        print(f" - Image {img}")

        predict_dataset = InferenceDataset(
            f"{root}/{img}.tiff",
            rle=None,
            overlap_factor=config.overlap_factor,
            reduce_factor=reduce_factor,
            tile_size=config.tile_size,
            transforms=HE_preprocess_test(augment=False, visualize=False),
        )

        if use_full_size:
            global_pred = predict_entire_mask(
                predict_dataset, model, batch_size=config.val_bs, tta=use_tta
            )

        else:
            global_pred = predict_entire_mask_downscaled(
                predict_dataset, model, batch_size=config.val_bs, tta=use_tta
            )

        if save:
            np.save(
                log_folder + f"pred_{img}_{fold}.npy",
                global_pred.cpu().numpy()
            )

        if not use_full_size:
            try:
                shape = df_info[df_info.image_file == img + ".tiff"][
                    ["width_pixels", "height_pixels"]
                ].values.astype(int)[0]
            except IndexError:
                shape = EXTRA_IMGS_SHAPES[img]

            global_pred = threshold_resize_torch(
                global_pred, shape, threshold=global_threshold
            )
        else:
            global_pred = (global_pred > global_threshold).cpu().numpy()


def k_fold_inf_test(
    config,
    images,
    log_folder=None,
    use_full_size=True,
    global_threshold=None,
    use_tta=False,
    save=False,
):
    """
    Performs a k-fold inference on the test data.

    Args:
        config (Config): Parameters.
        images (list of strings): Image names.
        log_folder (None or str, optional): Folder to load the weights from. Defaults to None.
        use_full_size (bool, optional): Whether to use full resolution images. Defaults to True.
        global_threshold (float, optional): Threshold for probabilities. Defaults to None.
        use_tta (bool, optional): Whether to use tta. Defaults to False.
        save (bool, optional): Whether to save predictions. Defaults to False.
    """
    for fold in range(5):
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {5}  -------------\n")

            model = define_model(
                config.decoder,
                config.encoder,
                num_classes=config.num_classes,
                encoder_weights=config.encoder_weights,
            ).to(config.device)
            model.zero_grad()
            model.eval()

            load_model_weights(
                model, log_folder + f"{config.decoder}_{config.encoder}_{fold}.pt"
            )

            validate_inf_test(
                model,
                config,
                images,
                fold=fold,
                log_folder=log_folder,
                use_full_size=use_full_size,
                global_threshold=global_threshold,
                use_tta=use_tta,
                save=save,
            )
