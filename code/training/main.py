import gc
import time
import torch
import numpy as np
import pandas as pd


from params import DATA_PATH
from training.train import fit
from model_zoo.models import define_model
from data.transforms import HE_preprocess
from utils.metrics import tweak_threshold
from training.predict import predict_entire_mask_downscaled
from data.dataset import InMemoryTrainDataset, InferenceDataset
from utils.torch import seed_everything, count_parameters, save_model_weights


def train(config, dataset, fold, log_folder=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        dataset (torch Dataset): whole dataset InMemory
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
        double_model=config.double_model,
        input_size=config.tile_size,
        use_bot=config.use_bot,
        use_fpn=config.use_fpn,
        use_mixstyle=config.use_mixstyle,
    ).to(config.device)
    model.zero_grad()

    n_parameters = count_parameters(model)

    print(f"    -> {n_parameters} trainable parameters")

    # switch dataset to the correct fold
    dataset.update_fold_nb(fold)
    print("    -> Validation images :", dataset.valid_set, "\n")

    meter, history = fit(
        model,
        dataset,
        optimizer_name=config.optimizer,
        loss_name=config.loss,
        activation=config.activation,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        swa_first_epoch=config.swa_first_epoch,
        mix_proba=config.mix_proba,
        mix_alpha=config.mix_alpha,
        loss_oof_weight=config.loss_oof_weight,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
        use_fp16=config.use_fp16,
    )

    if config.save_weights and log_folder is not None:
        name = f"{config.decoder}_{config.encoder}_{fold}.pt"
        save_model_weights(
            model,
            name,
            cp_folder=log_folder,
        )

    return meter, history, model


def validate(model, config, val_images):
    """
    Quick model validation on full images.
    Validation is performed on downscaled images.

    Args:
        model (torch model): Trained model.
        config (Config): Model config.
        val_images (list of strings): Validation image ids.
    """
    rles = pd.read_csv(DATA_PATH + f"train_{config.reduce_factor}.csv")
    scores = []
    for img in val_images:

        predict_dataset = InferenceDataset(
            f"{DATA_PATH}train_{config.reduce_factor}/{img}.tiff",
            rle=rles[rles["id"] == img]["encoding"],
            overlap_factor=config.overlap_factor,
            tile_size=config.tile_size,
            reduce_factor=1,
            transforms=HE_preprocess(augment=False, visualize=False, size=config.tile_size),
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


def k_fold(config, log_folder=None):
    """
    Performs a patient grouped k-fold cross validation.
    The following things are saved to the log folder : val predictions, histories

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """
    scores = []
    nb_folds = 5
    # Data preparation
    print("Creating in-memory dataset ...")

    start_time = time.time()
    df_rle = pd.read_csv(f"../input/train_{config.reduce_factor}_fc.csv")
    df_rle_test = pd.read_csv(config.pl_path)

    train_img_names = df_rle.id.unique()  # [::-1]

    in_mem_dataset = InMemoryTrainDataset(
        train_img_names,
        df_rle,
        train_tile_size=config.tile_size,
        reduce_factor=config.reduce_factor,
        train_transfo=HE_preprocess(size=config.tile_size),
        valid_transfo=HE_preprocess(augment=False, size=config.tile_size),
        train_path=f"../input/train_{config.reduce_factor}/",
        iter_per_epoch=config.iter_per_epoch,
        on_spot_sampling=config.on_spot_sampling,
        sampling_mode=config.sampling_mode,
        use_external=config.use_external,
        oof_folder=config.oof_folder,
        df_rle_test=df_rle_test,
        test_path="../input/test/",
        use_pl=config.use_pl,
    )
    print(f"Done in {time.time() - start_time :.0f} seconds.")

    for i in config.selected_folds:
        print(f"\n-------------   Fold {i + 1} / {nb_folds}  -------------\n")

        meter, history, model = train(config, in_mem_dataset, i, log_folder=log_folder)

        print("\n    -> Validating \n")

        val_images = in_mem_dataset.valid_set
        scores += validate(model, config, val_images)

        if log_folder is not None:
            history.to_csv(log_folder + f"history_{i}.csv", index=False)

        if log_folder is None or len(config.selected_folds) == 1:
            return meter

        del meter
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n\n  ->  Dice CV : {np.mean(scores) :.3f}  +/- {np.std(scores) :.3f}")
