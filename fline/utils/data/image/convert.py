import torch
import numpy as np


def convert_tensor_image_to_numpy_batch(tensor_image: torch.Tensor):
    np_arr = tensor_image.detach().cpu().numpy()
    np_arr = np.moveaxis(np_arr, 1, -1)
    return np_arr


def convert_tensor_image_to_numpy(tensor_image: torch.Tensor):
    np_arr = tensor_image.detach().cpu().numpy()
    np_arr = np.moveaxis(np_arr, 0, -1)
    return np_arr
