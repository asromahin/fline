import cv2
import numpy as np


class ResizeMode:
    resize = 'resize'
    padding = 'padding'


def imread(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def imread_resize(im_path, shape):
    im = imread(im_path)
    im = cv2.resize(im, (shape[1], shape[0]))
    return im


def imread_padding(im_path, q, fill_value):
    im = imread(im_path)
    im = padding_q(im, q, fill_value=fill_value)
    return im


def padding(im, shape, fill_value):
    res = np.full(shape=shape, fill_value=fill_value)
    y_shift = (im.shape[0] - shape[0])
    x_shift = (im.shape[1] - shape[1])
    res[y_shift:y_shift+im.shape[0], x_shift:x_shift+im.shape[1]] = im
    return res


def padding_q(im, q, fill_value):
    res = padding_qxy(im, q, q, fill_value)
    return res


def padding_qxy(im, qy, qx, fill_value):
    q_x = im.shape[0] % qx
    q_y = im.shape[1] % qy
    shape_y = im.shape[0]
    shape_x = im.shape[1]
    if q_x != 0:
        shape_y = im.shape[0] - q_x + qx
    if q_y != 0:
        shape_x = im.shape[1] - q_y + qy
    if len(im.shape) == 2:
        res = np.full(shape=(shape_y, shape_x), fill_value=fill_value, dtype='float32')
    else:
        res = np.full(shape=(shape_y, shape_x, im.shape[2]), fill_value=fill_value, dtype='float32')
    y_shift = (shape_y - im.shape[0])//2
    x_shift = (shape_x - im.shape[1])//2
    res[y_shift:y_shift+im.shape[0], x_shift:x_shift+im.shape[1]] = im
    return res
