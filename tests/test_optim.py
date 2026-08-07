import numpy as np
import pytest
from minigrad import Tensor, Linear, SGD, step_decay
from minigrad.nn import Parameter
from minigrad.optim import Optimizer


# --- Optimizer base ---

def test_optimizer_base_step_not_implemented():
    opt = Optimizer([])
    with pytest.raises(NotImplementedError):
        opt.step()

def test_optimizer_zero_grad_clears_all_param_grads():
    p1 = Parameter(np.zeros(3)); p1.grad = np.array([1.0, 2.0, 3.0])
    p2 = Parameter(np.zeros((2, 2))); p2.grad = np.ones((2, 2))
    opt = SGD([p1, p2], lr=0.1)
    opt.zero_grad()
    assert np.all(p1.grad == 0)
    assert np.all(p2.grad == 0)


# --- SGD ---

def test_sgd_step_applies_update():
    p = Parameter(np.array([1.0, 2.0, 3.0]))
    p.grad = np.array([0.5, -1.0, 2.0])
    opt = SGD([p], lr=0.1)
    opt.step()
    assert np.allclose(p.data, [1.0 - 0.05, 2.0 + 0.1, 3.0 - 0.2])

def test_sgd_reduces_loss_on_toy_problem():
    # Fit W in a linear model to a linear target: mean-squared error should
    # drop by orders of magnitude across training.
    rng = np.random.default_rng(42)
    N = 32
    X_np = rng.standard_normal((N, 3))
    W_true = rng.standard_normal((3, 2))
    Y_target = X_np @ W_true

    lin = Linear(3, 2, bias=False)
    opt = SGD(lin.parameters(), lr=0.05)

    def mse(pred):
        diff = pred - Y_target
        return (diff * diff).sum() / N

    initial = mse(lin(X_np)).data.item()
    for _ in range(500):
        opt.zero_grad()
        loss = mse(lin(X_np))
        loss.backward()
        opt.step()
    final = mse(lin(X_np)).data.item()
    assert final < initial * 1e-3


# --- step_decay ---

def test_step_decay_schedule():
    f = step_decay(num_epochs=1000, lr=1.0, lr_decay_at=(0.5, 0.75), lr_decay_factor=0.1)
    assert f(0) == 1.0
    assert f(499) == 1.0
    assert np.isclose(f(500), 0.1)
    assert np.isclose(f(749), 0.1)
    assert np.isclose(f(750), 0.01)
    assert np.isclose(f(999), 0.01)

def test_step_decay_defaults_return_callable():
    f = step_decay()
    assert callable(f)
    assert f(0) == 1.0
