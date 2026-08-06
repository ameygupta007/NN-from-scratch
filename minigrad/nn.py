import numpy as np
from minigrad import Tensor

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
    _modules: Dict[str, "Module"]
    
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
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        # He init to keep variance from blowing up
        self.W = Parameter(np.random.randn(nin, nout) * np.sqrt(2.0 / nin)) 
        if bias:
            self.b = Parameter(np.zeros(nout))
        self.bias = bias

    def forward(self, X):
        if not isinstance(X, Tensor):
            X = Tensor(X)
        return X @ self.W + self.b if self.bias else X @ self.W
    
class MLP(Module):
    def __init__(self, nin, nouts, dropout_p=0.0):
        super().__init__()
        # nin - number of inputs
        # nouts - list of number of neurons in each layer
        all_sizes = [nin] + nouts
        self.layers = [Linear(all_sizes[i], all_sizes[i+1]) for i in range(len(nouts))]
        # register params
        for i, l in enumerate(self.layers):
            self.register_module(f"Layer_{i}", l)

        self.dropout_p = dropout_p

    def forward(self, x):
        # call each layer one after the other
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                # activation for all but last layer
                x = x.tanh()

            if i < len(self.layers) - 1 and self.dropout_p > 0:
                x = dropout(x, self.dropout_p, training=self.training)
        return x
    
def dropout(x, p, training=True):
    '''
    x : Tensor
    p : probability with which we drop a neuron in the layer
    return: Tensor with dropout applied
    '''
    if not training or p == 0.0:
        return x
    mask = (np.random.rand(*x.data.shape) > p).astype(x.data.dtype) / (1.0 - p)
    return x * mask
    