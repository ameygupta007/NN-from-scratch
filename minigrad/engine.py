import numpy as np
from collections import deque

class Tensor:

    """
    Wraps data to enable operations on Tensor and enable backpropagation
    """

    def __init__(self, data, children=(), op='', label=''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._prev = set(children)
        self._op = op
        self._backward = lambda : None

        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward

        return out
     
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other
    
    def __matmul__(self, other):
        # also handles batch matmul
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            a, b, g = self.data, other.data, out.grad
            # promote a, b to 2D if they are 1D.
            a2 = a[None, :] if a.ndim == 1 else a # row vector, matching numpy behaviour
            b2 = b[:, None] if b.ndim == 1 else b # column vector, matching numpy behaviour
            g2 = g
            # undo what np strips in g if a or b were 1D
            if b.ndim == 1: g2 = np.expand_dims(g2, -1)
            if a.ndim == 1: g2 = np.expand_dims(g2, -2)
            grad_a = g2 @ b2.swapaxes(-1, -2)
            grad_b = a2.swapaxes(-1, -2) @ g2
            if a.ndim == 1: grad_a = grad_a.squeeze(-2)   # undo the row-vec promotion
            if b.ndim == 1: grad_b = grad_b.squeeze(-1)   # undo the col-vec promotion
            self.grad += _unbroadcast(grad_a, self.shape)
            other.grad += _unbroadcast(grad_b, other.shape)
        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other) 
        
        return other @ self
    
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
    
    def __pow__(self, other): # other should be a constant, not Tensor
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += _unbroadcast(other * self.data ** (other -1) * out.grad, self.shape)
        out._backward = _backward

        return out

    def sqrt(self):
        return self ** 0.5
    
    def __iter__(self):
        return iter(self.data)
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self, ), 'exp')
        def _backward():
            self.grad += _unbroadcast(out.data * out.grad, self.shape)
        
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        out = Tensor(np.tanh(x), (self,), 'tanh')
        def _backward():
            self.grad += _unbroadcast((1 - out.data**2) * out.grad, self.shape)
        out._backward = _backward
        return out

    def relu(self):
        x = np.maximum(0.0, self.data)
        out = Tensor(x, (self,), 'ReLU')
        def _backward():
            self.grad += _unbroadcast(out.grad * (out.data > 0), self.shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        N = self.data.size if axis is None else self.shape[axis]
        s = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(s / N, (self, ), 'mean')
        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad / N
        out._backward = _backward
        return out
    
    def softmax_cross_entropy(self, y):
        '''
        expects y: np array of probabilities for each logit
        '''
        assert isinstance(y, np.ndarray)
        z = self.data
        z_shift = z - z.max(axis=-1, keepdims=True)
        exps = np.exp(z_shift)
        sm = exps / exps.sum(axis=-1, keepdims=True)
        lse = np.log(exps.sum(axis=-1, keepdims=True))
        N = np.prod(z.shape[:-1])
        loss_val = (-(z_shift * y).sum(axis=-1, keepdims=True) + lse).sum() / N

        out = Tensor(loss_val, (self,), 'softmax_ce')
        def _backward():
            self.grad += (sm - y) / N * out.grad
        out._backward = _backward
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), (self, ), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out
    
    def transpose(self, axis1, axis2):
        # swap axis1 and axis2
        out = Tensor(self.data.swapaxes(axis1, axis2), (self, ), 'transpose')
        def _backward():
            self.grad += out.grad.swapaxes(axis1, axis2)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data
        z_shift = z - z.max(axis, keepdims=True)
        exps = np.exp(z_shift)
        sm = exps / exps.sum(axis, keepdims=True)
        out = Tensor(sm, (self,), 'softmax')
        def _backward():
            self.grad += out.data * (out.grad - np.sum(out.grad * out.data, axis, keepdims=True))
        out._backward = _backward
        return out

    def backward(self):
        self.grad = np.ones_like(self.data)

        # topo sort - iterative
        topo = []
        visited = set()
        stack = deque()
        stack.append((self, False))

        while stack:
            v, expanded = stack[-1]
            if expanded:
                stack.pop()
                topo.append(v)
            elif v in visited:
                stack.pop()
            else:
                visited.add(v)
                stack[-1] = (v, True)
                for child in v._prev:
                    if child not in visited:
                        stack.append((child, False))

        for n in reversed(topo):
            n._backward()

def concat(ts, axis=0):
    # concatenate Tensors ts along axis, return new Tensor
    ts = tuple(ts)
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), ts, 'concat')
    
    splits = np.cumsum([t.shape[axis] for t in ts])[:-1]
    def _backward():
        for t, g in zip(ts, np.split(out.grad, splits, axis=axis)):
            t.grad += g
    out._backward = _backward
    return out

def embedding(indices, e):
    # lookup embeddings in e by indices
    out = Tensor(e.data[indices], (e,), 'embedding')
    def _backward():
        np.add.at(e.grad, indices, out.grad)
    out._backward = _backward
    return out

def _unbroadcast(grad, shape):
    # handle grads flowing backwards to Tensors that were broadcast in the initial operation
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