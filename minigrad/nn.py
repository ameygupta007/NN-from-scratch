from __future__ import annotations
import numpy as np
from minigrad import Tensor, embedding
import minigrad.functional as F

from typing import Any, Dict

class Parameter(Tensor):
    """
    Marker class - wrap Tensor to differentiate Module parameters from other variables
    """
    pass

class Module:
    """
    Generic class for a NN Module - enables us to recursively collect all parameters in a tree of submodules
    """
    # type annotations so static analysers know these attributes exist
    _params: Dict[str, Parameter]
    _modules: Dict[str, Module]
    
    def __init__(self):
        self.__dict__["_params"] = dict()
        self.__dict__["_modules"] = dict()
        self.training = False

    def parameters(self) -> list[Parameter]:
        params = list(self._params.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def modules(self) -> list[Module]:
        return list(self._modules.values())
    
    def __setattr__(self, name: str, value: Any) -> None:
        if "_params" not in self.__dict__: raise RuntimeError("Module.__init__() not called")

        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def register_module(self, name: str, module: Module):
        if not isinstance(module, Module):
            raise TypeError(f"{name} is not a Module")
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def register_parameter(self, name: str, p: Parameter):
        if not isinstance(p, Parameter):
            raise TypeError(f"{name} is not a Parameter")
        self._params[name] = p
        object.__setattr__(self, name, p)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        # set this and all submodules to train mode
        self.training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        # set this and all submodules to eval mode (training attribute False)
        self.training = False
        for m in self._modules.values():
            m.eval()
        
class Linear(Module):
    def __init__(self, nin, nout, bias=True, init_std=None):
        super().__init__()
        # Default He init to keep variance from blowing up
        std = np.sqrt(2.0 / nin) if init_std is None else init_std
        self.W = Parameter(np.random.randn(nin, nout) * std) 
        if bias:
            self.b = Parameter(np.zeros(nout))
        self.bias = bias

    def forward(self, X):
        if not isinstance(X, Tensor):
            X = Tensor(X)
        return X @ self.W + self.b if self.bias else X @ self.W
    
class MLP(Module):
    def __init__(self, nin, nouts, dropout_p=0.0, activation=F.tanh):
        super().__init__()
        # nin - number of inputs
        # nouts - list of number of neurons in each layer
        all_sizes = [nin] + nouts
        self.layers = [Linear(all_sizes[i], all_sizes[i+1]) for i in range(len(nouts))]
        # register params
        for i, l in enumerate(self.layers):
            self.register_module(f"Layer_{i}", l)

        self.dropout_p = dropout_p
        self.activation = activation

    def forward(self, x):
        # call each layer one after the other
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                # activation for all but last layer
                x = self.activation(x)

            if i < len(self.layers) - 1 and self.dropout_p > 0:
                x = dropout(x, self.dropout_p, training=self.training)
        return x

class Embedding(Module):
    def __init__(self, n_embeddings, embedding_dim):
        super().__init__()
        self.w = Parameter(
            np.random.randn(n_embeddings, embedding_dim)
        )

    def forward(self, indices):
        return embedding(indices, self.w)

class LayerNorm(Module):
    '''
    LayerNorm module with learnable weight and bias. Wraps F.layer_norm
    '''
    def __init__(self, dim):
        super().__init__()
        self.w = Parameter(np.ones(dim))
        self.b = Parameter(np.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, self.w, self.b, eps=1e-5)

class MultiHeadAttention(Module):
    def __init__(self, k, heads=4, mask=True, init_std=0.02, proj_std=None):
        super().__init__()
        assert k % heads == 0
        self.k, self.heads = k, heads
        self.mask = mask

        # compute queries, keys, values for all heads
        self.toqueries = Linear(k, k, bias=False, init_std=init_std)
        self.tokeys = Linear(k, k, bias=False, init_std=init_std)
        self.tovalues = Linear(k, k, bias=False, init_std=init_std)

        # apply after multi-head self attention
        self.unifyheads = Linear(k,k, init_std= proj_std or init_std)

    def forward(self, x):
        assert isinstance(x, Tensor)
        # b: number of batches
        # t: number of input vectors in batch
        # k: dimension of vectors
        b, t, k = x.shape
        h = self.heads

        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        # cut according to h, fold it into the batch dimension
        s = k // h
        qx = queries.reshape((b, t, h, s)).transpose(1, 2)
        kx = keys.reshape((b, t, h, s)).transpose(1,2)
        vx = values.reshape((b, t, h, s)).transpose(1,2)

        attention = scaled_dot_product_attention(qx, kx, vx, mask=self.mask)
        attention = attention.transpose(1,2).reshape((b,t,k)) # (b,h,t,s) -> (b,t,h,s) -> (b,t,k)
        return self.unifyheads(attention)

class TransformerBlock(Module):
    '''
    Decoder transformer block, uses pre-norm. 
    '''
    def __init__(self, k, heads=4, dropout_p=0.0):
        super().__init__()

        self.attn = MultiHeadAttention(k, heads, mask=True)
        self.ln1 = LayerNorm(k)
        self.ln2 = LayerNorm(k)

        self.expand = Linear(k, 4*k)
        self.contract = Linear(4*k, k)
        self.dropout_p = dropout_p

    def forward(self, x):
        x = x + dropout(self.attn(self.ln1(x)), p=self.dropout_p, training=self.training)
        h = self.contract(F.gelu(self.expand(self.ln2(x))))
        x = x + dropout(h, self.dropout_p, self.training)
        return x

def scaled_dot_product_attention(q, k, v, mask=False):
    d = q.shape[-1]
    w = q @ k.transpose(-1, -2) / np.sqrt(d)
    # mask
    if mask:
        causal = np.triu(np.full((w.shape[-2], w.shape[-1]), np.float64(-1e9)), k=1)
        w = w + causal
    w_softmax = w.softmax(axis=-1) # row-wise softmax

    out = w_softmax @ v
    return out

def dropout(x, p, training=True):
    '''
    x : Tensor
    p : probability with which we drop a neuron in the layer
    return: Tensor with dropout applied
    '''
    if not training or p == 0.0:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(x.data.dtype) / (1.0 - p)
    return x * mask
    