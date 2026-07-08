class Value:

    def __init__(self, data, children=(), op='', label=''):
        self.data = data
        self._prev = set(children)
        self._op = op
        self.grad = 0
        self._backward = lambda : None

        self.label = label


    def __repr__(self):
        return f"Value:{{name: {self.label}, data: {self.data}}}"
    
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