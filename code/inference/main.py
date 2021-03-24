import torch
import pandas as pd


from training.predict import (  # noqa
    predict_entire_mask_no_thresholding,
    threshold_resize,
    threshold_resize_torch
)

from model_zoo.models import define_model

from data.dataset import InferenceDataset
from data.transforms import HE_preprocess

from utils.rle import enc2mask
from utils.torch import load_model_weights
from utils.metrics import dice_scores_img, tweak_threshold
# from utils.plots import plot_thresh_scores
from params import TIFF_PATH, DATA_PATH, REDUCE_FACTOR, TIFF_PATH_4


def validate_inf(
    model,
    config,
    val_images,
    log_folder=None,
    use_full_size=True,
    global_threshold=None,
):
    df_info = pd.read_csv(DATA_PATH + "HuBMAP-20-dataset_information.csv")

    if use_full_size:
        root = TIFF_PATH
        rle_path = DATA_PATH + "train.csv"
        reduce_factor = REDUCE_FACTOR
        batch_size = config.val_bs // 4
    else:
        root = TIFF_PATH_4
        rle_path = DATA_PATH + "train_4.csv"
        reduce_factor = 1
        batch_size = config.val_bs // 4

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
            transforms=HE_preprocess(augment=False, visualize=False),
        )

        global_pred = predict_entire_mask_no_thresholding(
            predict_dataset, model, batch_size=batch_size, upscale=use_full_size
        )

        threshold, score = tweak_threshold(
            mask=torch.from_numpy(predict_dataset.mask).cuda(),
            pred=global_pred
        )

        print(f" - Scored {score :.4f} for image {img} with threshold {threshold:.2f}")

        shape = df_info[df_info.image_file == img + ".tiff"][
            ["width_pixels", "height_pixels"]
        ].values.astype(int)[0]
        mask_truth = enc2mask(rles_full[rles_full["id"] == img]["encoding"], shape)

        global_threshold = (
            global_threshold if global_threshold is not None else threshold
        )
        global_pred = threshold_resize(global_pred.cpu().numpy(), shape, threshold=global_threshold)

        score = dice_scores_img(global_pred, mask_truth)
        scores.append(score)

        print(
            f" - Scored {score :.4f} for image {img} "
            f"with threshold {global_threshold:.2f} after resizing\n"
        )

    return scores


def k_fold_inf(config, df, log_folder=None, use_full_size=True, global_threshold=None):
    """
    Performs a patient grouped k-fold cross validation.
    The following things are saved to the log folder : val predictions, val indices, histories

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
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
            )
            break

    return scores
