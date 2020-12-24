import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import sparse


def get_tile_weighting(size, sigma=1):
    half = size // 2
    w = np.ones((size, size), np.float32)

    x = np.concatenate([np.mgrid[-half:0], np.mgrid[1: half + 1]])[:, None]
    x = np.tile(x, (1, size))
    x = half + 1 - np.abs(x)
    y = x.T

    w = np.minimum(x, y)
    w = (w / w.max()) ** sigma
    w = np.minimum(w, 1)

    return w.astype(np.float16)


def predict_entire_mask_no_thresholding(dataset, model, batch_size=32, upscale=True):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    w = get_tile_weighting(dataset.tile_size, sigma=1)

    preds = []
    model.eval()
    for batch in loader:
        pred = model(batch.to("cuda"))
        if upscale:
            _, _, H, W = preds.shape
            pred = torch.nn.functional.interpolate(
                preds, (H * dataset.reduce_factor, W * dataset.reduce_factor)
            )

        pred = pred.sigmoid().detach().cpu().squeeze()
        preds.append(pred)

    preds = torch.cat(preds)

    global_pred = np.zeros(
        (dataset.orig_size[0], dataset.orig_size[1]), dtype=np.float16
    )
    global_counter = np.zeros(
        (dataset.orig_size[0], dataset.orig_size[1]), dtype=np.float16
    )

    for tile_idx, (pos_x, pos_y) in enumerate(dataset.positions):
        global_pred[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1]] += (
            preds[tile_idx, :, :].numpy().astype(np.float16) * w
        )
        global_counter[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1]] += w

    # divide by overlapping tiles
    global_pred = np.divide(global_pred, global_counter).astype(np.float16)

    return global_pred


def predict_entire_mask(predict_dataset, model, batch_size=32, upscale=True):
    reduce_fac = predict_dataset.reduce_factor
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)
    # Make all predictions by batch
    res = []
    model.eval()
    for batch in predict_loader:
        preds = model(batch.to("cuda"))
        b_size, C, H, W = preds.shape

        if upscale:
            preds = torch.nn.functional.interpolate(
                preds, (H * reduce_fac, W * reduce_fac)
            )

        preds = torch.sigmoid(preds) > 0.5
        preds = preds.detach().cpu().squeeze().type(torch.int8)
        res.append(preds)

    # reconstruct spatial positions
    preds_tensor = torch.cat(res).type(torch.int8)
    del res
    global_pred = np.zeros(
        (predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint8
    )
    global_counter = np.zeros(
        (predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint8
    )

    for tile_idx, (pos_x, pos_y) in enumerate(predict_dataset.positions):
        global_pred[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1]] += (
            preds_tensor[tile_idx, :, :].numpy().astype(np.uint8)
        )
        global_counter[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1]] += 1

    # divide by overlapping tiles
    global_pred = np.divide(global_pred, global_counter).astype(np.float16)

    return global_pred


def predict_entire_mask_efficient(
    predict_dataset, model, batch_size=32, threshold=None
):
    reduce_fac = predict_dataset.reduce_factor
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)
    # Make all predictions by batch
    res = []
    model.eval()
    with torch.no_grad():
        for batch in predict_loader:
            raw_preds = model(batch.to("cuda"))
            b_size, C, H, W = raw_preds.shape
            upscale_preds = torch.nn.functional.interpolate(
                raw_preds, (H * reduce_fac, W * reduce_fac)
            )
            preds = torch.sigmoid(upscale_preds) > 0.5
            preds = preds.detach().cpu().squeeze(dim=1).numpy().astype(np.uint8)
            for b in range(preds.shape[0]):
                res.append(sparse.csr_matrix(preds[b, :, :]))

    del predict_loader
    global_pred = np.zeros(
        (predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint8
    )
    global_counter = np.zeros(
        (predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint8
    )

    for tile_idx, (pos_x, pos_y) in enumerate(predict_dataset.positions):
        global_pred[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1]] += res[
            tile_idx
        ]  # .toarray()
        global_counter[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1]] += 1

    # divide by overlapping tiles
    if threshold is not None:
        bool_pred = np.zeros(
            (predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.bool
        )
        for row in range(predict_dataset.orig_size[0]):
            bool_pred[row, :] = (
                np.divide(global_pred[row, :], global_counter[row, :]) > threshold
            ).astype(np.bool)
        return bool_pred
    else:
        global_pred = np.divide(global_pred, global_counter).astype(np.float16)

    return global_pred


def predict_ensemble(
    predict_dataset, models_path, batch_size=32, threshold=None, device="cuda"
):
    reduce_fac = predict_dataset.reduce_factor
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    global_pred = np.zeros(
        (predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint16
    )
    global_counter = np.zeros(
        (predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint16
    )

    for model_path in models_path:
        model = torch.jit.load(model_path).to(device)
        # Make all predictions by batch
        res = []
        model.eval()
        for batch in predict_loader:
            raw_preds = model(batch.to(device))
            b_size, C, H, W = raw_preds.shape
            upscale_preds = torch.nn.functional.interpolate(
                raw_preds, (H * reduce_fac, W * reduce_fac)
            )
            preds = torch.sigmoid(upscale_preds) > 0.5
            preds = preds.detach().cpu().squeeze(dim=1).numpy().astype(np.uint8)
            for b in range(preds.shape[0]):
                res.append(sparse.csr_matrix(preds[b, :, :]))

        for tile_idx, (pos_x, pos_y) in enumerate(predict_dataset.positions):
            global_pred[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1]] += res[
                tile_idx
            ]  # .toarray()
            global_counter[pos_x[0]: pos_x[1], pos_y[0]: pos_y[1]] += 1
        del res
        del model
    del predict_loader

    # divide by overlapping tiles
    if threshold is not None:
        bool_pred = np.zeros(
            (predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.bool
        )
        for row in range(predict_dataset.orig_size[0]):
            bool_pred[row, :] = (
                np.divide(global_pred[row, :], global_counter[row, :]) > threshold
            ).astype(np.bool)
        return bool_pred
    else:
        global_pred = np.divide(global_pred, global_counter).astype(np.float16)

    return global_pred
