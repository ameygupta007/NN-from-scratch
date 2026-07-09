import torch
from minigrad import Value

def test_forward_pass():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    assert d.data == 4.0  # 2*-3 + 10

def test_backward_simple():
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b
    c.backward()
    assert a.grad == -3.0  # dc/da = b
    assert b.grad == 2.0   # dc/db = a

def test_against_pytorch():
    # minigrad
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a)
    d += 3 * d + (b - a)
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    # pytorch, same computation
    a = torch.Tensor([-4.0]).double(); a.requires_grad = True
    b = torch.Tensor([2.0]).double(); b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a)
    d = d + 3 * d + (b - a)
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    # forward pass matched
    assert abs(gmg.data - gpt.data.item()) < 1e-6
    # backward pass matched
    assert abs(amg.grad - apt.grad.item()) < 1e-6 # type: ignore
    assert abs(bmg.grad - bpt.grad.item()) < 1e-6 # type:ignore


