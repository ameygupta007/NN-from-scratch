import numpy as np

def layer_norm(x, weight=None, bias=None, eps=1e-5):
    # normalize over x's last axis (-1); weight and bias must broadcast to x.shape[-1:]
    mean = x.mean(axis=-1, keepdims=True)
    diff = x - mean
    var = (diff * diff).mean(axis=-1, keepdims=True)
    out = weight * diff / (var + eps).sqrt() + bias
    return out

def gelu(x):
    # approximate tanh GELU
    return 0.5 * x * (1 + (np.sqrt(2 /np.pi) * (x + 0.044715 * (x ** 3))).tanh())

