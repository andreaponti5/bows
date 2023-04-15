import json
import numpy as np
import os
import random
import torch


def res_dict(keys, *args):
    return dict(zip(keys, args))


def save_res(filepath, res):
    if os.path.isfile(filepath) is False:
        json.dump(res, open(filepath, 'w'))
    else:
        old_res = json.load(open(filepath, 'r'))
        key_to_add = set(res.keys()) - set(old_res.keys())
        for key in key_to_add:
            old_res[key] = res[key]
        json.dump(old_res, open(filepath, 'w'))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
