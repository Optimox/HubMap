import gc
import torch
import numpy as np
import pandas as pd

from training.train import fit
from data.dataset import TileDataset, InferenceDataset
from data.transforms import HE_preprocess
from model_zoo.models import define_model
from utils.save import save_as_jit
from utils.torch import seed_everything, count_parameters, save_model_weights

from params import SIZE, TIFF_PATH_4, DATA_PATH
from training.predict import predict_entire_mask_downscaled
from utils.metrics import tweak_threshold


def train(config, df_train, df_val, fold, log_folder=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        SegmentationMeter: Meter.
        pandas dataframe: Training history.
    """

    seed_everything(config.seed)

    model = define_model(
        config.decoder,
        config.encoder,
        num_classes=config.num_classes,
        encoder_weights=config.encoder_weights,
    ).to(config.device)
    model.zero_grad()

    train_dataset = TileDataset(
        df_train,
        img_dir=config.img_dir,
        mask_dir=config.mask_dir,
        transforms=HE_preprocess(),
    )

    val_dataset = TileDataset(
        df_val,
        img_dir=config.img_dir,
        mask_dir=config.mask_dir,
        transforms=HE_preprocess(augment=False),
    )

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    meter, history = fit(
        model,
        train_dataset,
        val_dataset,
        optimizer_name=config.optimizer,
        loss_name=config.loss,
        activation=config.activation,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        swa_first_epoch=config.swa_first_epoch,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
    )

    if config.save_weights and log_folder is not None:
        name = f"{config.decoder}_{config.encoder}_{fold}.pt"
        save_model_weights(
            model,
            name,
            cp_folder=log_folder,
        )
        if "efficientnet" not in name:
            save_as_jit(model, log_folder, name, train_img_size=SIZE)

    return meter, history, model


def validate(model, config, val_images):
    """
    Quick model validation on full images.
    Validation is performed on downscaled images.

    Args:
        model ([type]): Trained model.
        config ([type]): Model config.
        val_images (list]): Validation image ids.
    """
    rles = pd.read_csv(DATA_PATH + "train_4.csv")
    scores = []
    for img in val_images:

        predict_dataset = InferenceDataset(
            f"{TIFF_PATH_4}/{img}.tiff",
            rle=rles[rles["id"] == img]["encoding"],
            overlap_factor=config.overlap_factor,
            reduce_factor=1,
            transforms=HE_preprocess(augment=False, visualize=False),
        )

        global_pred = predict_entire_mask_downscaled(
            predict_dataset, model, batch_size=config.val_bs, tta=False
        )

        threshold, score = tweak_threshold(
            mask=torch.from_numpy(predict_dataset.mask).cuda(), pred=global_pred
        )

        scores.append(score)
        print(
            f" - Scored {score :.4f} for downscaled image {img} with threshold {threshold:.2f}"
        )

    return scores


def k_fold(config, df, log_folder=None):
    """
    Performs a patient grouped k-fold cross validation.
    The following things are saved to the log folder : val predictions, histories

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

            df_train = df[df[config.cv_column] != fold].reset_index()
            df_val = df[df[config.cv_column] == fold].reset_index()

            meter, history, model = train(
                config, df_train, df_val, i, log_folder=log_folder
            )

            print("\n    -> Validating \n")

            val_images = df_val["tile_name"].apply(lambda x: x.split("_")[0]).unique()[:1]
            scores += validate(model, config, val_images)

            if log_folder is not None:
                history.to_csv(log_folder + f"history_{i}.csv", index=False)

            if log_folder is None or len(config.selected_folds) == 1:
                return meter

            del meter, model
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n\n  ->  Dice CV : {np.mean(scores) :.3f}  +/- {np.std(scores) :.3f}")
