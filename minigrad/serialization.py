''''
Functionality for saving/loading models.
'''
import numpy as np
from minigrad import Module

def save(model : Module, path):
    arrays = {f"p/{k}" : v for k,v in model.state_dict().items()}
    arrays.update({f"c/{k}" : v for k,v in model.config().items()})
    np.savez(path, allow_pickle=True, **arrays)

def load(path, cls):
    '''
    Load an instance of cls from path
    '''
    with np.load(path) as data:
        params = {}
        cfg = {}
        for arr in data.files:
            if arr[0] == "c":
                cfg[arr[2:]] = data[arr]
            else:
                params[arr[2:]] = data[arr]

    model = cls(**cfg)
    model.load_state_dict(params)
    return model
    