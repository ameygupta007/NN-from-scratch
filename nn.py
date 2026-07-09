import random
from typing import Any
from minigrad import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        val = sum(wi*xi for wi,xi in zip(self.w,x)) + self.b
        # activation function - could use relu as well
        return val.tanh()

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts):
        # nin - number of inputs
        # nouts - list of number of neurons in each layer
        all_sizes = [nin] + nouts
        self.layers = [Layer(all_sizes[i], all_sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        # call each layer one after the other
        for layer in self.layers:
            x = layer(x)
        return x
    
    def params(self):
        # return a list of all the Value objects representing paremeters in this MLP
        parameters = []
        for l in self.layers:
            for n in l.neurons:
                parameters.extend(n.w)
                parameters.append(n.b)

        return parameters
    
