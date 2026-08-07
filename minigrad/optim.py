import numpy as np

class Optimizer:
    '''
    Base class for optimizers.
    '''
    def __init__(self, params) -> None:
        self.params = params
        
    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0)

    def step(self):
        # subclass-specific
        raise NotImplementedError

class SGD(Optimizer):
    '''
    Stochastic Gradient Descent: step is learning rate * grad
    '''
    def __init__(self, params, lr=1.0):
        super().__init__(params)
        self.lr = lr

    def step(self):
        lr = self.lr
        for p in self.params:
            p.data -= p.grad * lr

### LR SCHEDULERS: each returns a function from t to LR

def step_decay(num_epochs=5000, lr=1.0, lr_decay_at=(0.5, 0.75), lr_decay_factor=0.1):
    decays = [int(num_epochs * fraction) for fraction in lr_decay_at]
    def f(t):
        l = lr
        for d in decays:
            if t >= d:
                l *= lr_decay_factor
            else:
                break
        return l

    return f