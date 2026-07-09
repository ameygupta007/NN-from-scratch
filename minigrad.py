import math

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


def main():
    pass

if __name__ == "__main__":
    main()
