import numpy as np
import torch
from minigrad import Value, Tensor


# --- Value: scalar autograd ---

def test_value_forward_pass():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    assert d.data == 4.0  # 2*-3 + 10

def test_value_backward_simple():
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b
    c.backward()
    assert a.grad == -3.0  # dc/da = b
    assert b.grad == 2.0   # dc/db = a

def test_value_against_pytorch():
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

    assert abs(gmg.data - gpt.data.item()) < 1e-6
    assert abs(amg.grad - apt.grad.item()) < 1e-6  # type: ignore
    assert abs(bmg.grad - bpt.grad.item()) < 1e-6  # type: ignore


# --- Array: numpy-backed autograd, element-wise ---

def test_array_forward_pass():
    a = Tensor([2.0, 1.0, 3.0])
    b = Tensor([-3.0, 4.0, -1.0])
    c = Tensor([10.0, 5.0, 2.0])
    d = a * b + c
    expected = np.array([4.0, 9.0, -1.0])  # 2*-3+10, 1*4+5, 3*-1+2
    assert np.allclose(d.data, expected)

def test_array_backward_simple():
    a = Tensor([2.0, 1.0, 3.0])
    b = Tensor([-3.0, 4.0, -1.0])
    c = a * b
    c.backward()
    # gradient of sum(c) since backward broadcasts a scalar 1
    assert np.allclose(a.grad, b.data)  # dc/da = b element-wise
    assert np.allclose(b.grad, a.data)  # dc/db = a element-wise

def test_array_against_pytorch():
    a_init = [-4.0, 3.0, -2.0]
    b_init = [2.0, 1.5, -1.0]

    # array
    a = Tensor(a_init)
    b = Tensor(b_init)
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

    # pytorch, same computation (backward with ones matches minigrad's implicit
    # scalar-1 seed broadcast across the output vector)
    a = torch.tensor(a_init, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(b_init, dtype=torch.float64, requires_grad=True)
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
    g.backward(torch.ones_like(g))
    apt, bpt, gpt = a, b, g

    assert np.allclose(gmg.data, gpt.detach().numpy(), atol=1e-6)
    assert np.allclose(amg.grad, apt.grad.numpy(), atol=1e-6)  # type: ignore
    assert np.allclose(bmg.grad, bpt.grad.numpy(), atol=1e-6)  # type: ignore


def test_array_matmul_forward():
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])         # (2, 3)
    B = Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])    # (3, 2)
    C = A @ B
    expected = np.array([[58.0, 64.0], [139.0, 154.0]])
    assert C.data.shape == (2, 2)
    assert np.allclose(C.data, expected)

def test_array_matmul_backward():
    A_init = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]           # (2, 3)
    B_init = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]      # (3, 2)

    A = Tensor(A_init)
    B = Tensor(B_init)
    C = A @ B
    C.backward()

    At = torch.tensor(A_init, dtype=torch.float64, requires_grad=True)
    Bt = torch.tensor(B_init, dtype=torch.float64, requires_grad=True)
    Ct = At @ Bt
    Ct.backward(torch.ones_like(Ct))

    assert np.allclose(C.data, Ct.detach().numpy())
    assert np.allclose(A.grad, At.grad.numpy())  # type: ignore
    assert np.allclose(B.grad, Bt.grad.numpy())  # type: ignore

def test_array_matmul_chain():
    # (X @ W + b) style: composition of matmul with broadcast add and mul
    X_init = [[1.0, -2.0], [0.5, 3.0], [-1.5, 2.5]]       # (3, 2)
    W_init = [[0.3, -0.7, 1.1], [1.2, 0.4, -0.5]]         # (2, 3)
    b_init = [0.1, -0.2, 0.3]                             # (3,) broadcasts over rows

    X = Tensor(X_init)
    W = Tensor(W_init)
    b = Tensor(b_init)
    Y = (X @ W + b) * Tensor(2.0)
    Y.backward()

    Xt = torch.tensor(X_init, dtype=torch.float64, requires_grad=True)
    Wt = torch.tensor(W_init, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_init, dtype=torch.float64, requires_grad=True)
    Yt = (Xt @ Wt + bt) * 2.0
    Yt.backward(torch.ones_like(Yt))

    assert np.allclose(Y.data, Yt.detach().numpy())
    assert np.allclose(X.grad, Xt.grad.numpy())  # type: ignore
    assert np.allclose(W.grad, Wt.grad.numpy())  # type: ignore
    assert np.allclose(b.grad, bt.grad.numpy())  # type: ignore
