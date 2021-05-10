import torch
import numpy as np
import pandas as pd

from training.predict import (
    predict_entire_mask_downscaled,
    predict_entire_mask,
    threshold_resize_torch,
    predict_entire_mask_downscaled_tta
)

from model_zoo.models import define_model

from data.dataset import InferenceDataset
from data.transforms import HE_preprocess_test

from utils.rle import enc2mask
from utils.torch import load_model_weights
from utils.metrics import dice_scores_img, tweak_threshold

from params import TIFF_PATH, DATA_PATH


def validate_inf(
    model,
    config,
    val_images,
    log_folder=None,
    use_full_size=True,
    global_threshold=None,
    use_tta=False,
    save=False,
    save_all_tta=False,
):
    """
    Performs inference with a model on a list of train images.

    Args:
        model (torch model): Segmentation model.
        config (Config): Parameters.
        val_images (list of strings): Image names.
        fold (int, optional): Fold index. Defaults to 0.
        log_folder (str or None, optional): Folder to save predictions to. Defaults to None.
        use_full_size (bool, optional): Whether to use full resolution images. Defaults to True.
        global_threshold (float, optional): Threshold for probabilities. Defaults to None.
        use_tta (bool, optional): Whether to use tta. Defaults to False.
        save (bool, optional): Whether to save predictions. Defaults to False.
        save_all_tta (bool, optional): Whether to save predictions for all tta. Defaults to False.
    """
    df_info = pd.read_csv(DATA_PATH + "HuBMAP-20-dataset_information.csv")

    if use_full_size:
        root = TIFF_PATH
        rle_path = DATA_PATH + "train.csv"
        reduce_factor = config.reduce_factor
    else:
        root = DATA_PATH + f"train_{config.reduce_factor}/"
        rle_path = DATA_PATH + f"train_{config.reduce_factor}.csv"
        reduce_factor = 1

    rles = pd.read_csv(rle_path)
    rles_full = pd.read_csv(DATA_PATH + "train.csv")

    print("\n    -> Validating \n")
    scores = []

    for img in val_images:

        predict_dataset = InferenceDataset(
            f"{root}/{img}.tiff",
            rle=rles[rles["id"] == img]["encoding"],
            overlap_factor=config.overlap_factor,
            reduce_factor=reduce_factor,
            tile_size=config.tile_size,
            transforms=HE_preprocess_test(augment=False, visualize=False),
        )

        if save_all_tta:
            global_pred = predict_entire_mask_downscaled_tta(
                predict_dataset, model, batch_size=config.val_bs
            )
            np.save(
                log_folder + f"pred_{img}.npy",
                global_pred.cpu().numpy()
            )

            global_pred = global_pred.mean(0)

        else:
            if use_full_size:
                global_pred = predict_entire_mask(
                    predict_dataset, model, batch_size=config.val_bs, tta=use_tta
                )
                threshold, score = 0.4, 0

            else:
                global_pred = predict_entire_mask_downscaled(
                    predict_dataset, model, batch_size=config.val_bs, tta=use_tta
                )

                threshold, score = tweak_threshold(
                    mask=torch.from_numpy(predict_dataset.mask).cuda(), pred=global_pred
                )
                print(
                    f" - Scored {score :.4f} for downscaled"
                    f"image {img} with threshold {threshold:.2f}"
                )

        shape = df_info[df_info.image_file == img + ".tiff"][
            ["width_pixels", "height_pixels"]
        ].values.astype(int)[0]
        mask_truth = enc2mask(rles_full[rles_full["id"] == img]["encoding"], shape)

        global_threshold = (
            global_threshold if global_threshold is not None else threshold
        )

        if save and not save_all_tta:
            np.save(
                log_folder + f"pred_{img}.npy",
                global_pred.cpu().numpy()
            )

        if not use_full_size:
            global_pred = threshold_resize_torch(
                global_pred, shape, threshold=global_threshold
            )
        else:
            global_pred = (global_pred > global_threshold).cpu().numpy()

        score = dice_scores_img(global_pred, mask_truth)
        scores.append(score)

        print(
            f" - Scored {score :.4f} for image {img} with threshold {global_threshold:.2f}\n"
        )

    return scores


def k_fold_inf(
    config,
    df,
    log_folder=None,
    use_full_size=True,
    global_threshold=None,
    use_tta=False,
    save=False,
    save_all_tta=False,
):
    """
    Performs a k-fold inference on the train data.

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Train metadata. Contains image names and rles.
        log_folder (None or str, optional): Folder to load the weights from. Defaults to None.
        use_full_size (bool, optional): Whether to use full resolution images. Defaults to True.
        global_threshold (float, optional): Threshold for probabilities. Defaults to None.
        use_tta (bool, optional): Whether to use tta. Defaults to False.
        save_all_tta (bool, optional): Whether to save predictions for all tta. Defaults to False.
    """
    folds = df[config.cv_column].unique()
    scores = []

    for i, fold in enumerate(folds):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {len(folds)}  -------------\n")
            df_val = df[df[config.cv_column] == fold].reset_index()

            val_images = df_val["tile_name"].apply(lambda x: x.split("_")[0]).unique()

            model = define_model(
                config.decoder,
                config.encoder,
                num_classes=config.num_classes,
                encoder_weights=config.encoder_weights,
            ).to(config.device)
            model.zero_grad()
            model.eval()

            load_model_weights(
                model, log_folder + f"{config.decoder}_{config.encoder}_{i}.pt"
            )

            scores += validate_inf(
                model,
                config,
                val_images,
                log_folder=log_folder,
                use_full_size=use_full_size,
                global_threshold=global_threshold,
                use_tta=use_tta,
                save=save,
                save_all_tta=save_all_tta,
            )

    return scores
