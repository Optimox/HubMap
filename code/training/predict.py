from tqdm.notebook import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader


def predict_entire_mask(predict_dataset, model, batch_size = 32):
    reduce_fac = predict_dataset.reduce_factor
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)
    # Make all predictions by batch
    res = []
    masks = []
    model.eval()
    for batch, batch_masks in tqdm(predict_loader):
        raw_preds = model(batch.to("cuda"))
        b_size, C, H, W = raw_preds.shape
        upscale_preds = torch.nn.functional.interpolate(raw_preds,
                                                        (H*reduce_fac, W*reduce_fac))
        preds = torch.sigmoid(upscale_preds) > 0.5
        preds = preds.detach().cpu()
        masks.append(batch_masks)
        res.append(preds.squeeze())
    
    # reconstruct spatial positions
    global_pred = np.zeros((predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint8)
    global_counter = np.zeros((predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint8)
    preds_tensor = torch.cat(res)
    for tile_idx, (pos_x, pos_y) in enumerate(predict_dataset.positions):
        global_pred[pos_x[0]:pos_x[1],
                         pos_y[0]:pos_y[1]] += preds_tensor[tile_idx,:,:].numpy()
        global_counter[pos_x[0]:pos_x[1],
                       pos_y[0]:pos_y[1]] += 1

    # divide by overlapping tiles
    global_pred = np.divide(global_pred, global_counter)
    
    return global_pred