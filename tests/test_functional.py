import numpy as np
import torch
from minigrad import Tensor, layer_norm, gelu

def test_layer_norm():
    arr = np.random.randn(2, 10)
    weight = np.random.randn(10)
    bias = np.random.randn(10)

    t = Tensor(arr)
    ln = layer_norm(t, Tensor(weight), Tensor(bias))
    ln.backward()

    tt = torch.tensor(arr, requires_grad=True)
    ln_t = torch.nn.functional.layer_norm(tt, (10,), weight=torch.tensor(weight), bias=torch.tensor(bias))
    ln_t.backward(torch.ones_like(ln_t))

    assert np.allclose(ln.data, ln_t.data.numpy(), atol=1e-5)
    assert np.allclose(t.grad, tt.grad.numpy(), atol=1e-5) # type: ignore

def test_gelu():
    arr = np.random.randn(2, 10)

    t = Tensor(arr)
    ln = gelu(t)
    ln.backward()

    tt = torch.tensor(arr, requires_grad=True)
    ln_t = torch.nn.functional.gelu(tt, approximate='tanh')
    ln_t.backward(torch.ones_like(ln_t))

    assert np.allclose(ln.data, ln_t.data.numpy(), atol=1e-5)
    assert np.allclose(t.grad, tt.grad.numpy(), atol=1e-5) # type: ignore
