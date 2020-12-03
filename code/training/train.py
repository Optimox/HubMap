import time
import torch

from torchcontrib.optim import SWA
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from params import NUM_WORKERS, SIZE
from utils.logger import update_history
from utils.metrics import SegmentationMetrics  ## noqa

# from training.freeze import freeze_backbone, unfreeze
from training.optim import (
    define_loss,
    define_optimizer,
    prepare_for_loss,
    average_loss
)


def fit(
    model,
    train_dataset,
    val_dataset,
    mode="mask",
    optimizer_name="Adam",
    loss_name="BCEWithLogitsLoss",
    activation="sigmoid",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    swa_first_epoch=50,
    num_classes=1,
    use_haussdorf=False,
    verbose=1,
    first_epoch_eval=0,
    device="cuda"
):
    """
    Usual torch fit function.
    Supports SWA.

    Args:
        model (torch model): Model to train.
        train_dataset (torch dataset): Dataset to train with.
        val_dataset (torch dataset): Dataset to validate with.
        mode (str, optional): Indicates what the model is predicting. Defaults to "mask".
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        loss_name (str, optional): Loss name. Defaults to 'BCEWithLogitsLoss'.
        activation (str, optional): Activation function. Defaults to 'sigmoid'.
        epochs (int, optional): Number of epochs. Defaults to 50.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        warmup_prop (float, optional): Warmup proportion. Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        swa_first_epoch (int, optional): Epoch to start applying SWA from. Defaults to 50.
        num_classes (int, optional): Number of classes. Defaults to 1.
        use_haussdorf (bool, optional): Whether to use the haussdorf metric. Defaults to False.
        verbose (int, optional): Period (in epochs) to display logs at. Defaults to 1.
        first_epoch_eval (int, optional): Epoch to start evaluating at. Defaults to 0.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(val_dataset) x num_classes]: Last prediction on the validation data.
        pandas dataframe: Training history.
    """

    avg_val_loss = 0.0
    history = None

    optimizer = define_optimizer(optimizer_name, model.parameters(), lr=lr)
    if swa_first_epoch <= epochs:
        optimizer = SWA(optimizer)

    loss_fct = define_loss(loss_name, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()

        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0

        if epoch + 1 > swa_first_epoch:
            optimizer.swap_swa_sgd()
            # print("Swap to SGD")

        for x, masks in train_loader:
            x = x.type(torch.FloatTensor).to(device)
            masks = masks.type(torch.FloatTensor).to(device)

            y_pred = model(x)
            y_pred, masks = prepare_for_loss(y_pred, masks, loss_name, mode=mode, device=device)

            loss = loss_fct(y_pred, masks)

            loss = average_loss(loss)

            loss.backward()

            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        if epoch + 1 >= swa_first_epoch:
            # print("update + swap to SWA")
            optimizer.update_swa()
            optimizer.swap_swa_sgd()

        model.eval()
        avg_val_loss = 0.
        meter = SegmentationMetrics()

        if epoch + 1 >= first_epoch_eval:
            with torch.no_grad():
                for x, masks in val_loader:
                    x = x.type(torch.FloatTensor).to(device)
                    masks = masks.type(torch.FloatTensor).to(device)

                    y_pred = model(x)

                    y_pred, y_batch = prepare_for_loss(
                        y_pred,
                        masks,
                        loss_name,
                        device=device,
                        train=False
                    )

                    loss = loss_fct(y_pred, y_batch)
                    loss = average_loss(loss)

                    avg_val_loss += loss / len(val_loader)

                    if activation == "sigmoid":
                        y_pred = torch.sigmoid(y_pred)
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
                print(
                    f"val_loss={avg_val_loss:.3f} \t dice={metrics['dice'][-1]:.4f}"
                )
            else:
                print("")
            history = update_history(
                history, metrics, epoch + 1, avg_loss,
                avg_val_loss.detach().cpu().numpy(), elapsed_time
            )

    del val_loader, train_loader, y_pred, loss, x, y_batch
    torch.cuda.empty_cache()

    return meter, history
