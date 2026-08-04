from minigrad import Tensor
import numpy as np

class Layer:
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout)) # each row represents weights of a neuron
        self.b = Tensor(np.random.rand(nout))
        self.activation = True

    def __call__(self, X):
        if not isinstance(X, Tensor):
            X = Tensor(X)
        outs = X @ self.W + self.b
        if self.activation:
            outs = outs.tanh()
        return outs
    
    def parameters(self):
        return [self.W, self.b]
    
class MLP:
    def __init__(self, nin, nouts, dropout_p=0.0):
        # nin - number of inputs
        # nouts - list of number of neurons in each layer
        all_sizes = [nin] + nouts
        self.layers = [Layer(all_sizes[i], all_sizes[i+1]) for i in range(len(nouts))]
        self.layers[-1].activation = False
        self.training = True
        self.dropout_p = dropout_p

    def __call__(self, x):
        # call each layer one after the other
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 and self.dropout_p > 0:
                x = dropout(x, self.dropout_p, training=self.training)
        return x
    
    def parameters(self):
        # return a list of all the Tensor objects representing paremeters in this MLP
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params
    
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
    