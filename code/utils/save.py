import torch
import os

def save_as_jit(model, saving_path, model_name,
                train_img_size=256, device='cuda'):
    """
    Save a model as jit.
    This enable calling this wihtout requiring
    the inital classes and dependencies.

    Parameters
    ----------
    - model : torch model
        Model to save
    - saving_path : str
        Path to the folder where to save the model
    - model_name : str
        Name use to save model
    - train_img_size : int
        Size of images used during training (train_img_size, train_img_size)
    """
    model.eval()
    for m in model.modules():
        m.requires_grad = False
    
    x = torch.ones(1, 3, train_img_size, train_img_size).to(device)
    traced_model = torch.jit.trace(model, x)
    traced_model.save(os.path.join(saving_path, f'{model_name}.jit'))
    print(f"Model {model_name} saved.")
    return