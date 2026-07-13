from minigrad import Tensor
import numpy as np

class Layer:
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.rand(nin, nout)) # each row represents weights of a neuron
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
    def __init__(self, nin, nouts):
        # nin - number of inputs
        # nouts - list of number of neurons in each layer
        all_sizes = [nin] + nouts
        self.layers = [Layer(all_sizes[i], all_sizes[i+1]) for i in range(len(nouts))]
        self.layers[-1].activation = False

    def __call__(self, x):
        # call each layer one after the other
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        # return a list of all the Value objects representing paremeters in this MLP
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params
    
