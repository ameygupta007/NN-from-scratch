import math
import numpy as np

class Value:

    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self._prev = set(children)
        self._op = op
        self.grad = 0
        self._backward = lambda : None

        self.label = label


    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
     
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1.0
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def __pow__(self, other): # other should be a constant, not Value
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += other * self.data ** (other -1) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        out = Value(math.tanh(x), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        x = max(0, self.data)
        out = Value(x, (self,), 'ReLU')
        def _backward():
            self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1

        topo = []
        visited = set()
        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_sort(child)
                topo.append(v)
    
        topo_sort(self)
        for n in reversed(topo):
            n._backward()


class Tensor:
    # Value but for wrapping a numpy array, and supporting matrix/tensor operations

    def __init__(self, data, children=(), op='', label=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = set(children)
        self._op = op
        self._backward = lambda : None

        self.label = label


    def __repr__(self):
        return f"Array(data={self.data})"
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out
     
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1.0
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def __pow__(self, other): # other should be a constant, not Array
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += _unbroadcast(other * self.data ** (other -1) * out.grad, self.data.shape)
        out._backward = _backward

        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self, ), 'exp')
        def _backward():
            self.grad += _unbroadcast(out.data * out.grad, self.data.shape)
        
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        out = Tensor(np.tanh(x), (self,), 'tanh')
        def _backward():
            self.grad += _unbroadcast((1 - out.data**2) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def relu(self):
        x = np.maximum(0.0, self.data)
        out = Tensor(x, (self,), 'ReLU')
        def _backward():
            self.grad += _unbroadcast(out.grad * (out.data > 0), self.data.shape)
        out._backward = _backward
        return out

    def backward(self):
        self.grad = np.ones_like(self.data)

        topo = []
        visited = set()
        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_sort(child)
                topo.append(v)

        topo_sort(self)
        for n in reversed(topo):
            n._backward()


def _unbroadcast(grad, shape):
    # handle grads flowing backwards to Arrays that were broadcast in the initial operation
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def main():
    pass

if __name__ == "__main__":
    main()
