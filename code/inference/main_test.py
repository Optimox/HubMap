import numpy as np
import pandas as pd

from training.predict import (
    predict_entire_mask_downscaled,
    predict_entire_mask,
    threshold_resize_torch,
)

from model_zoo.models import define_model

from data.dataset import InferenceDataset
from data.transforms import HE_preprocess
from utils.torch import load_model_weights
from params import TIFF_PATH_TEST, DATA_PATH


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
            transforms=HE_preprocess(augment=False, visualize=False),
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
            shape = df_info[df_info.image_file == img + ".tiff"][
                ["width_pixels", "height_pixels"]
            ].values.astype(int)[0]

            global_pred = threshold_resize_torch(
                global_pred, shape, threshold=global_threshold
            )
        else:
            global_pred = (global_pred > global_threshold).cpu().numpy()


def k_fold_inf_test(
    config,
    df,
    log_folder=None,
    use_full_size=True,
    global_threshold=None,
    use_tta=False,
    save=False,
):
    """
    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """
    images = df['id'].values
    for fold in range(5):
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {5}  -------------\n")

            model = define_model(
                config.decoder,
                config.encoder,
                num_classes=config.num_classes,
                encoder_weights=config.encoder_weights,
                double_model=config.double_model,
                input_size=config.tile_size,
                use_bot=config.use_bot,
                use_fpn=config.use_fpn,
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