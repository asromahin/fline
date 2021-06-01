import torch
import torch.nn as nn
import numpy as np
import itertools
#from nltk.metrics.distance import edit_distance


def calc_confidence(pred, *args, **kwargs):
    pred = pred.permute(1, 0, 2)

    pred = nn.Softmax(dim=2)(pred)
    pred, _ = torch.Tensor.max(pred, dim=2)
    # pred = torch.prod(pred, dim=1)
    pred = pred.detach().cpu().numpy()

    return pred


def str_match(pred, true, *args, **kwargs):
    pred = pred.permute(1, 0, 2)
    pred = torch.Tensor.argmax(pred, dim=2).detach().cpu().numpy()

    true = true.detach().cpu().numpy()

    valid = np.zeros((pred.shape[0],))
    for j in range(pred.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in true[j] if k > 0]
        valid[j] = float(np.array_equal(p3, t))

    return valid


def str_match_2(pred, true, *args, **kwargs):
    pred = pred.permute(1, 0, 2)
    pred = torch.Tensor.argmax(pred, dim=2).detach().cpu().numpy()

    true = true.detach().cpu().numpy()
    valid_pos = []

    valid = 0
    for j in range(pred.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred[j])]
        p3 = [k for k in p3 if k > 0]
        valid += float(np.array_equal(p3, true[j]))
        valid_pos.append(valid)

    return valid / pred.shape[0], valid_pos


def statistic(pred, true, *args, **kwargs):
    pred = pred.permute(1, 0, 2)
    pred = torch.Tensor.argmax(pred, dim=2).detach().cpu().numpy()

    true = true.detach().cpu().numpy()

    errors = [0] * pred.shape[0]
    for j in range(pred.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in true[j] if k > 0]
        errors[j] = 1.0 - float(np.array_equal(p3, t))

    return errors


def mean_edit_distance(pred, true, *args, **kwargs):
    pred = pred.permute(1, 0, 2)
    pred = torch.Tensor.argmax(pred, dim=2).detach().cpu().numpy()

    true = true.detach().cpu().numpy()

    dist = 0
    for j in range(pred.shape[0]):
        p3 = [k for k, g in itertools.groupby(pred[j])]
        p3 = [k for k in p3 if k > 0]
        t = [k for k in true[j] if k > 0]

        s_pred = ''.join(list(map(lambda x: chr(x), p3)))
        s_true = ''.join(list(map(lambda x: chr(x), t)))

        dist += edit_distance(s_pred, s_true)

    return dist / pred.shape[0]
