from tqdm.notebook import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import sparse

def rle_encode_less_memory(mask):
    '''
    From https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    This simplified method requires first and last pixel to be zero
    '''
    pixels = mask.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

    
def predict_entire_mask(predict_dataset, model, batch_size = 32, threshold=None):
    reduce_fac = predict_dataset.reduce_factor
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)
    # Make all predictions by batch
    res = []
    model.eval()
    for batch in tqdm(predict_loader):
        raw_preds = model(batch.to("cuda"))
        b_size, C, H, W = raw_preds.shape
        upscale_preds = torch.nn.functional.interpolate(raw_preds,
                                                        (H*reduce_fac, W*reduce_fac))
        preds = torch.sigmoid(upscale_preds) > 0.5
        preds = preds.detach().cpu().squeeze(dim=1).numpy().astype(np.uint8)
        for b in range(preds.shape[0]):
            res.append(sparse.csr_matrix(preds[b,:,:]))

    del predict_loader
    global_pred = np.zeros((predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint8)
    global_counter = np.zeros((predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint8)
    

    for tile_idx, (pos_x, pos_y) in enumerate(predict_dataset.positions):
        global_pred[pos_x[0]:pos_x[1],
                         pos_y[0]:pos_y[1]] += res[tile_idx]#.toarray()
        global_counter[pos_x[0]:pos_x[1],
                       pos_y[0]:pos_y[1]] += 1

    # divide by overlapping tiles
    if threshold is not None:
        bool_pred = np.zeros((predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.bool)
        for row in tqdm(range(predict_dataset.orig_size[0])):
            bool_pred[row, :] = (np.divide(global_pred[row, :],
                                            global_counter[row, :]) > threshold).astype(np.bool)
        return bool_pred
    else:
        global_pred = np.divide(global_pred, global_counter).astype(np.float16)
    
    return global_pred

def predict_ensemble(predict_dataset, models_path, batch_size = 32, threshold=None, device='cuda'):
    reduce_fac = predict_dataset.reduce_factor
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    global_pred = np.zeros((predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint16)
    global_counter = np.zeros((predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.uint16)

    for model_path in models_path:
        model = torch.jit.load(model_path).to(device)
        # Make all predictions by batch
        res = []
        model.eval()
        for batch in tqdm(predict_loader):
            raw_preds = model(batch.to(device))
            b_size, C, H, W = raw_preds.shape
            upscale_preds = torch.nn.functional.interpolate(raw_preds,
                                                            (H*reduce_fac, W*reduce_fac))
            preds = torch.sigmoid(upscale_preds) > 0.5
            preds = preds.detach().cpu().squeeze(dim=1).numpy().astype(np.uint8)
            for b in range(preds.shape[0]):
                res.append(sparse.csr_matrix(preds[b,:,:]))

        for tile_idx, (pos_x, pos_y) in enumerate(predict_dataset.positions):
            global_pred[pos_x[0]:pos_x[1],
                            pos_y[0]:pos_y[1]] += res[tile_idx]#.toarray()
            global_counter[pos_x[0]:pos_x[1],
                        pos_y[0]:pos_y[1]] += 1
        del res
        del model
    del predict_loader

    # divide by overlapping tiles
    if threshold is not None:
        bool_pred = np.zeros((predict_dataset.orig_size[0], predict_dataset.orig_size[1]), dtype=np.bool)
        for row in tqdm(range(predict_dataset.orig_size[0])):
            bool_pred[row, :] = (np.divide(global_pred[row, :],
                                            global_counter[row, :]) > threshold).astype(np.bool)
        return bool_pred
    else:
        global_pred = np.divide(global_pred, global_counter).astype(np.float16)
    
    return global_pred