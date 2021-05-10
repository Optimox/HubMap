import time
import torch
import numpy as np

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from params import NUM_WORKERS
from training.mix import cutmix_data
from utils.torch import worker_init_fn
from utils.logger import update_history
from training.meter import SegmentationMeter
from training.optim import define_loss, define_optimizer, prepare_for_loss


def fit(
    model,
    dataset,
    optimizer_name="Adam",
    loss_name="BCEWithLogitsLoss",
    activation="sigmoid",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    mix_proba=0,
    mix_alpha=0.4,
    verbose=1,
    first_epoch_eval=0,
    num_classes=1,
    device="cuda",
):
    """
    Usual torch fit function.

    Args:
        model (torch model): Model to train.
        dataset (InMemoryTrainDataset): Dataset.
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        loss_name (str, optional): Loss name. Defaults to 'BCEWithLogitsLoss'.
        activation (str, optional): Activation function. Defaults to 'sigmoid'.
        epochs (int, optional): Number of epochs. Defaults to 50.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        warmup_prop (float, optional): Warmup proportion. Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        mix_proba (float, optional): Probability to apply mixup with. Defaults to 0.
        mix_alpha (float, optional): Mixup alpha parameter. Defaults to 0.4.
        verbose (int, optional): Period (in epochs) to display logs at. Defaults to 1.
        first_epoch_eval (int, optional): Epoch to start evaluating at. Defaults to 0.
        num_classes (int, optional): Number of classes. Defaults to 1.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(val_dataset) x num_classes]: Last prediction on the validation data.
        pandas dataframe: Training history.
    """

    avg_val_loss = 0.0
    history = None

    scaler = torch.cuda.amp.GradScaler()

    optimizer = define_optimizer(optimizer_name, model.parameters(), lr=lr)

    loss_fct = define_loss(loss_name, device=device)
    w_fc = 0.2

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    meter = SegmentationMeter()

    num_warmup_steps = int(warmup_prop * epochs * len(data_loader))
    num_training_steps = int(epochs * len(data_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        dataset.train(True)
        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0

        for batch in data_loader:
            x = batch[0].to(device).float()
            y_batch = batch[1].float()
            w = batch[2].float().cuda()

            if np.random.random() > mix_proba:
                x, y_batch = cutmix_data(x, y_batch, alpha=mix_alpha, device=device)

            with torch.cuda.amp.autocast():
                y_pred = model(x)

                if num_classes == 2:
                    y_batch, y_batch_fc = y_batch[:, :, :, 0], y_batch[:, :, :, 1]
                    y_pred, y_pred_fc = y_pred[:, 0], y_pred[:, 1]

                    y_pred_fc, y_batch_fc = prepare_for_loss(
                        y_pred_fc, y_batch_fc, loss_name, device=device
                    )

                y_pred, y_batch = prepare_for_loss(y_pred, y_batch, loss_name, device=device)

                loss = loss_fct(y_pred, y_batch).mean()
                if num_classes == 2:
                    loss_fc = loss_fct(y_pred_fc, y_batch_fc).mean(-1).mean(-1) * w
                    loss_fc = loss_fc.sum() / (w.sum() + 1e-6)

                    loss = (loss + w_fc * loss_fc) / (1 + w_fc)

                scaler.scale(loss).backward()

                avg_loss += loss.item() / len(data_loader)

                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            for param in model.parameters():
                param.grad = None

        model.eval()
        dataset.train(False)
        avg_val_loss = 0.
        metrics = meter.reset()

        if epoch + 1 >= first_epoch_eval:
            with torch.no_grad():
                for batch in data_loader:
                    x = batch[0].to(device).float()
                    y_batch = batch[1].float()

                    y_pred = model(x)

                    if num_classes == 2:  # only non-fc
                        y_batch = y_batch[:, :, :, 0]
                        y_pred = y_pred[:, 0]

                    y_pred, y_batch = prepare_for_loss(
                        y_pred,
                        y_batch,
                        loss_name,
                        device=device,
                        train=False
                    )

                    loss = loss_fct(y_pred, y_batch).mean()

                    avg_val_loss += loss / len(data_loader)

                    if activation == "sigmoid":
                        y_pred = torch.sigmoid(y_pred)
                    elif activation == "softmax":
                        y_pred = torch.softmax(y_pred, 2)

                    meter.update(y_batch, y_pred)

            metrics = meter.compute()

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s\t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )
            if epoch + 1 >= first_epoch_eval:
                print(f"val_loss={avg_val_loss:.3f} \t dice={metrics['dice'][0]:.4f}")
            else:
                print("")
            history = update_history(
                history, metrics, epoch + 1, avg_loss, avg_val_loss, elapsed_time
            )

    del (data_loader, y_pred, loss, x, y_batch)
    torch.cuda.empty_cache()

    return meter, history
