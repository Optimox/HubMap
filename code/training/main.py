import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from training.train import fit
from data.dataset import TileDataset
from data.transforms import HE_preprocess
from model_zoo.models import define_model
from utils.metrics import compute_metrics, SeededGroupKFold
from utils.torch import seed_everything, count_parameters, save_model_weights


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
        np array: Validation predictions.
        pandas dataframe: Training history.
    """

    seed_everything(config.seed)

    model = define_model(
        config.selected_model,
        num_classes=config.num_classes,
        use_attention=config.use_attention,
        num_classes_aux=config.num_classes_aux,
        pretrained_weights=config.pretrained_weights,
    ).to(config.device)
    model.zero_grad()

    train_dataset = TileDataset(
        df_train,
        root_dir=config.img_folder,
        transforms=HE_preprocess(),
        target_name=config.target_name,
    )

    val_dataset = TileDataset(
        df_val,
        root_dir=config.img_folder,
        transforms=HE_preprocess(augment=False),
        target_name=config.target_name,
    )

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val, history = fit(
        model,
        train_dataset,
        val_dataset,
        samples_per_patient=config.samples_per_patient,
        optimizer_name=config.optimizer,
        loss_name=config.loss,
        class_weights=config.class_weights,
        activation=config.activation,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        swa_first_epoch=config.swa_first_epoch,
        mix=config.mix,
        mix_proba=config.mix_proba,
        mix_alpha=config.mix_alpha,
        num_classes=config.num_classes,
        aux_loss_weight=config.aux_loss_weight,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
    )

    if config.save_weights and log_folder is not None:
        save_model_weights(
            model,
            f"{config.selected_model}_{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )

    del model, train_dataset, val_dataset
    torch.cuda.empty_cache()

    return pred_val, history


def k_fold(config, df, log_folder=None):
    """
    Performs a patient grouped k-fold cross validation.
    The following things are saved to the log folder :
    oof predictions, val predictions, val indices, histories

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        df_extra (None pandas dataframe, optional): Additional training data. Defaults to None.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """


    pred_oof = np.zeros((len(df), config.num_classes))

    for fold, in range(np.max(df.fold) +1):
        train_idx = df[df.fold!=fold].index
        val_idx = df[df.fold==fold].index
        if fold in config.selected_folds:
            print(f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            pred_val, history = train(
                config, df_train, df_val, fold, log_folder=log_folder
            )
            pred_oof[val_idx] = pred_val

            if log_folder is not None:
                np.save(log_folder + f"val_idx_{fold}.npy", val_idx)
                np.save(log_folder + f"pred_val_{fold}.npy", pred_val)
                history.to_csv(log_folder + f"history_{fold}.csv", index=False)

    if log_folder is not None:
        np.save(log_folder + "pred_oof.npy", pred_oof)

    metrics = compute_metrics(
        pred_oof, df[config.target_name].values, num_classes=config.num_classes
    )

    if config.num_classes == 1:
        print(f"\n  -> CV auc : {metrics['auc'][0]:.3f}")
    else:
        print(f"\n  -> CV accuracy : {metrics['accuracy'][0]:.3f}")
        print(f"  -> CV balanced accuracy : {metrics['balanced_accuracy'][0]:.3f}")

    if log_folder is not None:
        metrics = pd.DataFrame.from_dict(metrics)
        metrics.to_csv(log_folder + "metrics.csv", index=False)

    return metrics
