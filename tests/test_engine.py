import numpy as np
import pytest
import torch
from minigrad import Tensor


# --- Tensor: forward / backward basics ---

def test_tensor_forward_pass():
    a = Tensor([2.0, 1.0, 3.0])
    b = Tensor([-3.0, 4.0, -1.0])
    c = Tensor([10.0, 5.0, 2.0])
    d = a * b + c
    expected = np.array([4.0, 9.0, -1.0])  # 2*-3+10, 1*4+5, 3*-1+2
    assert np.allclose(d.data, expected)

def test_tensor_backward_simple():
    a = Tensor([2.0, 1.0, 3.0])
    b = Tensor([-3.0, 4.0, -1.0])
    c = a * b
    c.backward()
    # gradient of sum(c) since backward broadcasts a scalar 1
    assert np.allclose(a.grad, b.data)  # dc/da = b element-wise
    assert np.allclose(b.grad, a.data)  # dc/db = a element-wise

def test_tensor_against_pytorch():
    a_init = [-4.0, 3.0, -2.0]
    b_init = [2.0, 1.5, -1.0]

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


# --- Tensor: matmul across ndim combinations ---

def test_tensor_matmul_forward():
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])         # (2, 3)
    B = Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])    # (3, 2)
    C = A @ B
    expected = np.array([[58.0, 64.0], [139.0, 154.0]])
    assert C.data.shape == (2, 2)
    assert np.allclose(C.data, expected)

def test_tensor_matmul_backward():
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

def test_tensor_matmul_chain():
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

def test_tensor_matmul_1d_1d():
    # (k,) @ (k,) -> scalar (inner product)
    a_init = [1.0, -2.0, 3.0]
    b_init = [4.0, 0.5, -1.5]

    a = Tensor(a_init)
    b = Tensor(b_init)
    c = a @ b
    c.backward()

    at = torch.tensor(a_init, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b_init, dtype=torch.float64, requires_grad=True)
    ct = at @ bt
    ct.backward()

    assert c.data.shape == ()
    assert np.allclose(c.data, ct.detach().numpy())
    assert np.allclose(a.grad, at.grad.numpy())  # type: ignore
    assert np.allclose(b.grad, bt.grad.numpy())  # type: ignore

def test_tensor_matmul_1d_2d():
    # (k,) @ (k, m) -> (m,)
    x_init = [1.0, -2.0]
    W_init = [[0.3, -0.7, 1.1], [1.2, 0.4, -0.5]]

    x = Tensor(x_init)
    W = Tensor(W_init)
    y = x @ W
    y.backward()

    xt = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    Wt = torch.tensor(W_init, dtype=torch.float64, requires_grad=True)
    yt = xt @ Wt
    yt.backward(torch.ones_like(yt))

    assert y.data.shape == (3,)
    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())  # type: ignore
    assert np.allclose(W.grad, Wt.grad.numpy())  # type: ignore

def test_tensor_matmul_2d_1d():
    # (n, k) @ (k,) -> (n,)
    A_init = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x_init = [0.5, -1.0, 2.0]

    A = Tensor(A_init)
    x = Tensor(x_init)
    y = A @ x
    y.backward()

    At = torch.tensor(A_init, dtype=torch.float64, requires_grad=True)
    xt = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    yt = At @ xt
    yt.backward(torch.ones_like(yt))

    assert y.data.shape == (2,)
    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(A.grad, At.grad.numpy())  # type: ignore
    assert np.allclose(x.grad, xt.grad.numpy())  # type: ignore


# --- Tensor: reverse ops with non-Tensor left operand ---

def test_tensor_radd_rsub_rmul_rtruediv():
    a_init = [1.0, 2.0, 4.0]
    a = Tensor(a_init)
    out = 3.0 + a + 2.0 * a + 10.0 / a - a
    out.backward()

    at = torch.tensor(a_init, dtype=torch.float64, requires_grad=True)
    out_t = 3.0 + at + 2.0 * at + 10.0 / at - at
    out_t.backward(torch.ones_like(out_t))

    assert np.allclose(out.data, out_t.detach().numpy())
    assert np.allclose(a.grad, at.grad.numpy())  # type: ignore

def test_tensor_rmatmul_wraps_left_operand():
    # __rmatmul__ is invoked directly since numpy's @ short-circuits before
    # reflected operators fire; here we verify it wraps a bare list/ndarray.
    A_list = [[1.0, 2.0], [3.0, 4.0]]
    B_init = [[5.0, 6.0], [7.0, 8.0]]
    B = Tensor(B_init)
    C = B.__rmatmul__(A_list)  # equivalent to Tensor(A_list) @ B
    C.backward()

    At = torch.tensor(A_list, dtype=torch.float64)
    Bt = torch.tensor(B_init, dtype=torch.float64, requires_grad=True)
    Ct = At @ Bt
    Ct.backward(torch.ones_like(Ct))

    assert np.allclose(C.data, Ct.detach().numpy())
    assert np.allclose(B.grad, Bt.grad.numpy())  # type: ignore


# --- Tensor: unary activations & math ---

def test_tensor_exp_forward_backward():
    x_init = [-1.0, 0.0, 2.0]
    x = Tensor(x_init)
    y = x.exp()
    y.backward()

    xt = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    yt = xt.exp()
    yt.backward(torch.ones_like(yt))

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())  # type: ignore

def test_tensor_tanh_forward_backward():
    x_init = [[-2.0, -0.5], [0.5, 2.0]]
    x = Tensor(x_init)
    y = x.tanh()
    y.backward()

    xt = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    yt = xt.tanh()
    yt.backward(torch.ones_like(yt))

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())  # type: ignore

def test_tensor_relu_forward_backward():
    x_init = [-3.0, -0.5, 0.0, 1.5, 4.0]
    x = Tensor(x_init)
    y = x.relu()
    y.backward()

    xt = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    yt = xt.relu()
    yt.backward(torch.ones_like(yt))

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())  # type: ignore


# --- Tensor: sum with axis / keepdims ---

def test_tensor_sum_no_axis():
    x_init = [[1.0, 2.0], [3.0, 4.0]]
    x = Tensor(x_init)
    s = x.sum()
    s.backward()
    assert s.data.shape == ()
    assert np.isclose(s.data, 10.0)
    assert np.allclose(x.grad, np.ones_like(x.data))

def test_tensor_sum_axis_backward_matches_torch():
    x_init = [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]]
    x = Tensor(x_init)
    s = x.sum(axis=0)
    y = s * Tensor([2.0, -1.0, 0.5])  # non-uniform upstream grad
    y.backward()

    xt = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    st = xt.sum(dim=0)
    yt = st * torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    yt.backward(torch.ones_like(yt))

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())  # type: ignore

def test_tensor_sum_axis_keepdims():
    x_init = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    x = Tensor(x_init)
    s = x.sum(axis=1, keepdims=True)
    assert s.data.shape == (2, 1)
    upstream = Tensor([[2.0], [-1.0]])
    y = s * upstream
    y.backward()

    xt = torch.tensor(x_init, dtype=torch.float64, requires_grad=True)
    st = xt.sum(dim=1, keepdim=True)
    yt = st * torch.tensor([[2.0], [-1.0]], dtype=torch.float64)
    yt.backward(torch.ones_like(yt))

    assert np.allclose(y.data, yt.detach().numpy())
    assert np.allclose(x.grad, xt.grad.numpy())  # type: ignore


# --- Tensor: broadcasting through _unbroadcast ---

def test_tensor_broadcast_scalar_plus_vector():
    a = Tensor(3.0)                        # scalar
    b = Tensor([1.0, 2.0, 3.0])            # (3,)
    c = a + b
    c.backward()
    # scalar grad is sum of upstream 1s
    assert np.isclose(a.grad, 3.0)
    assert np.allclose(b.grad, np.ones(3))

def test_tensor_broadcast_row_plus_col():
    # (1, 3) + (2, 1) -> (2, 3), grads must unbroadcast in both directions
    row_init = [[1.0, 2.0, 3.0]]
    col_init = [[10.0], [20.0]]
    r = Tensor(row_init)
    c = Tensor(col_init)
    y = r + c
    y.backward()

    rt = torch.tensor(row_init, dtype=torch.float64, requires_grad=True)
    ct = torch.tensor(col_init, dtype=torch.float64, requires_grad=True)
    yt = rt + ct
    yt.backward(torch.ones_like(yt))

    assert y.data.shape == (2, 3)
    assert np.allclose(r.grad, rt.grad.numpy())  # type: ignore
    assert np.allclose(c.grad, ct.grad.numpy())  # type: ignore

def test_tensor_broadcast_mul_vector_and_matrix():
    v_init = [1.0, 2.0, 3.0]
    M_init = [[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]]
    v = Tensor(v_init)
    M = Tensor(M_init)
    y = v * M
    y.backward()

    vt = torch.tensor(v_init, dtype=torch.float64, requires_grad=True)
    Mt = torch.tensor(M_init, dtype=torch.float64, requires_grad=True)
    yt = vt * Mt
    yt.backward(torch.ones_like(yt))

    assert np.allclose(v.grad, vt.grad.numpy())  # type: ignore
    assert np.allclose(M.grad, Mt.grad.numpy())  # type: ignore


# --- Tensor: misc plumbing ---

def test_tensor_iter():
    a = Tensor([1.0, 2.0, 3.0])
    assert list(a) == [1.0, 2.0, 3.0]

def test_tensor_repr_contains_data():
    r = repr(Tensor([1.0, 2.0]))
    assert "Tensor" in r and "1" in r and "2" in r

def test_tensor_pow_asserts_on_tensor_exponent():
    a = Tensor([1.0, 2.0])
    with pytest.raises(AssertionError):
        _ = a ** Tensor([2.0, 2.0])  # type: ignore

def test_tensor_reuse_accumulates_grad():
    # x used twice in the graph — topo visits x once, chain-rule sums both paths
    x_init = [1.0, 2.0, 3.0]
    x = Tensor(x_init)
    y = x * x + x  # dy/dx = 2x + 1
    y.backward()
    assert np.allclose(x.grad, 2 * np.asarray(x_init) + 1)


# --- Tensor: softmax_cross_entropy ---

def test_softmax_cross_entropy_forward_backward_vs_torch():
    rng = np.random.default_rng(0)
    N, C = 5, 4
    logits_init = rng.standard_normal((N, C))
    labels = np.array([0, 2, 1, 3, 2])
    y_onehot = np.zeros((N, C))
    y_onehot[np.arange(N), labels] = 1.0

    z = Tensor(logits_init)
    loss = z.softmax_cross_entropy(y_onehot)
    loss.backward()

    zt = torch.tensor(logits_init, dtype=torch.float64, requires_grad=True)
    loss_t = torch.nn.functional.cross_entropy(zt, torch.tensor(labels))
    loss_t.backward()

    assert loss.data.shape == ()
    assert np.isclose(loss.data, loss_t.detach().numpy())
    assert np.allclose(z.grad, zt.grad.numpy())  # type: ignore
