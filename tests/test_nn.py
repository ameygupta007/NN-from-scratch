import numpy as np
import pytest
import torch
from minigrad import Tensor, Module, Linear, MLP, MultiHeadAttention, scaled_dot_product_attention, dropout
from minigrad.nn import Parameter


# --- Parameter marker ---

def test_parameter_is_tensor():
    p = Parameter(np.zeros(3))
    assert isinstance(p, Tensor)


# --- Module registration & lifecycle ---

def test_module_requires_super_init():
    class Bad(Module):
        def __init__(self):
            # deliberately skip super().__init__()
            self.p = Parameter(np.zeros(2))
    with pytest.raises(RuntimeError):
        Bad()

def test_module_setattr_registers_params_and_submodules():
    class M(Module):
        def __init__(self):
            super().__init__()
            self.p = Parameter(np.ones(2))
            self.sub = Linear(2, 3)
            self.plain = 42  # not registered

    m = M()
    assert "p" in m._params and m._params["p"] is m.p
    assert "sub" in m._modules and m._modules["sub"] is m.sub
    assert "plain" not in m._params and "plain" not in m._modules

def test_module_parameters_recursive():
    class Wrap(Module):
        def __init__(self):
            super().__init__()
            self.own = Parameter(np.zeros(4))
            self.lin = Linear(3, 5)  # W: (3,5), b: (5,)
    w = Wrap()
    params = w.parameters()
    shapes = sorted(p.shape for p in params)
    assert shapes == sorted([(4,), (3, 5), (5,)])

def test_module_modules_lists_direct_submodules_only():
    mlp = MLP(2, [3, 4])
    subs = mlp.modules()
    assert len(subs) == 2
    assert all(isinstance(s, Linear) for s in subs)

def test_register_module_and_parameter_type_checks():
    class M(Module):
        def __init__(self):
            super().__init__()
    m = M()
    with pytest.raises(TypeError):
        m.register_module("bad", "not a module")  # type: ignore
    with pytest.raises(TypeError):
        m.register_parameter("bad", np.zeros(2))  # type: ignore

def test_register_module_and_parameter_success():
    class M(Module):
        def __init__(self):
            super().__init__()
    m = M()
    p = Parameter(np.ones(3))
    lin = Linear(2, 2)
    m.register_parameter("weight", p)
    m.register_module("lin", lin)
    assert m.weight is p and m.lin is lin  # type: ignore
    assert m._params["weight"] is p
    assert m._modules["lin"] is lin

def test_module_forward_not_implemented():
    class M(Module):
        def __init__(self):
            super().__init__()
    with pytest.raises(NotImplementedError):
        M()(np.zeros(2))

def test_module_train_eval_recursive():
    mlp = MLP(2, [3, 3])
    mlp.train()
    assert mlp.training is True
    assert all(l.training for l in mlp.layers)
    mlp.eval()
    assert mlp.training is False
    assert all(l.training is False for l in mlp.layers)


# --- Linear ---

def test_linear_forward_shape_and_bias():
    lin = Linear(3, 4)
    X = np.random.randn(5, 3)
    y = lin(X)
    assert y.shape == (5, 4)
    # matches manual computation
    expected = X @ lin.W.data + lin.b.data
    assert np.allclose(y.data, expected)

def test_linear_no_bias_has_no_b_param():
    lin = Linear(3, 4, bias=False)
    assert "b" not in lin._params
    assert "W" in lin._params
    X = np.random.randn(2, 3)
    y = lin(X)
    assert np.allclose(y.data, X @ lin.W.data)

def test_linear_accepts_tensor_input():
    lin = Linear(2, 3)
    X = Tensor(np.random.randn(4, 2))
    y = lin(X)
    assert y.shape == (4, 3)

def test_linear_backward_populates_param_grads():
    lin = Linear(3, 2)
    X = Tensor(np.random.randn(4, 3))
    y = lin(X)
    y.sum().backward()
    assert lin.W.grad.shape == lin.W.shape
    assert lin.b.grad.shape == lin.b.shape
    assert np.any(lin.W.grad != 0.0)
    assert np.any(lin.b.grad != 0.0)


# --- MLP ---

def test_mlp_forward_shape_and_param_count():
    mlp = MLP(4, [8, 5, 3])
    X = np.random.randn(6, 4)
    y = mlp(X)
    assert y.shape == (6, 3)
    # params: (4,8)+(8,) + (8,5)+(5,) + (5,3)+(3,) = 6 tensors
    assert len(mlp.parameters()) == 6

def test_mlp_registers_layers_as_submodules():
    mlp = MLP(2, [3, 4])
    assert set(mlp._modules.keys()) == {"Layer_0", "Layer_1"}


# --- dropout ---

def test_dropout_identity_when_not_training():
    x = Tensor(np.ones((3, 4)))
    y = dropout(x, 0.5, training=False)
    assert y is x

def test_dropout_identity_when_p_zero():
    x = Tensor(np.ones((3, 4)))
    y = dropout(x, 0.0, training=True)
    assert y is x

def test_dropout_masks_some_units_and_scales():
    np.random.seed(0)
    x = Tensor(np.ones((100, 100)))
    p = 0.5
    y = dropout(x, p, training=True)
    # kept units are scaled by 1/(1-p) = 2.0; dropped are 0
    unique = np.unique(y.data)
    assert set(np.round(unique, 6).tolist()).issubset({0.0, 2.0})
    frac_kept = (y.data > 0).mean()
    assert 0.4 < frac_kept < 0.6  # roughly 1-p

def test_mlp_dropout_active_in_train_inactive_in_eval():
    np.random.seed(1)
    mlp = MLP(8, [16, 4], dropout_p=0.5)
    X = np.random.randn(10, 8)
    mlp.eval()
    y1 = mlp(X).data
    y2 = mlp(X).data
    assert np.allclose(y1, y2)  # deterministic in eval

    mlp.train()
    np.random.seed(2)
    y3 = mlp(X).data
    np.random.seed(3)
    y4 = mlp(X).data
    assert not np.allclose(y3, y4)  # stochastic in train


# --- scaled dot product attention ---

def _sdpa_ref(q, k, v, is_causal):
    # torch reference, returns (out, grad_q, grad_k, grad_v) for an upstream of ones
    qt = torch.tensor(q, requires_grad=True)
    kt = torch.tensor(k, requires_grad=True)
    vt = torch.tensor(v, requires_grad=True)
    out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=is_causal)
    out.sum().backward()
    return out.detach().numpy(), qt.grad.numpy(), kt.grad.numpy(), vt.grad.numpy()  # type: ignore

@pytest.mark.parametrize("shape", [
    (5, 4),        # unbatched: (t, d)
    (2, 5, 4),     # batched:   (b, t, d)
    (2, 3, 5, 4),  # per-head:  (b, h, t, d)
])
@pytest.mark.parametrize("mask", [False, True])
def test_sdpa_forward_backward_vs_torch(shape, mask):
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal(shape) for _ in range(3))

    tq, tk, tv = Tensor(q), Tensor(k), Tensor(v)
    out = scaled_dot_product_attention(tq, tk, tv, mask=mask)
    out.sum().backward()

    out_ref, gq, gk, gv = _sdpa_ref(q, k, v, is_causal=mask)

    assert out.shape == shape
    assert np.allclose(out.data, out_ref, atol=1e-5)
    assert np.allclose(tq.grad, gq, atol=1e-5)
    assert np.allclose(tk.grad, gk, atol=1e-5)
    assert np.allclose(tv.grad, gv, atol=1e-5)

def test_sdpa_handles_differing_key_and_value_dims():
    # t_q need not equal t_kv, and the value dim need not equal the key dim
    rng = np.random.default_rng(1)
    q = rng.standard_normal((2, 3, 8))
    k = rng.standard_normal((2, 5, 8))
    v = rng.standard_normal((2, 5, 6))

    out = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), mask=False)
    out_ref, _, _, _ = _sdpa_ref(q, k, v, is_causal=False)

    assert out.shape == (2, 3, 6)
    assert np.allclose(out.data, out_ref, atol=1e-5)

def test_sdpa_causal_mask_blocks_the_future():
    # position i must not see keys/values at j > i, so those get no gradient from row 0
    rng = np.random.default_rng(3)
    t, d = 5, 4
    q, k, v = (rng.standard_normal((t, d)) for _ in range(3))

    tq, tk, tv = Tensor(q), Tensor(k), Tensor(v)
    out = scaled_dot_product_attention(tq, tk, tv, mask=True)
    upstream = np.zeros((t, d))
    upstream[0] = 1.0  # differentiate output row 0 only
    (out * upstream).sum().backward()

    # row 0 attends only to position 0
    assert np.allclose(tv.grad[1:], 0.0)
    assert np.any(tv.grad[0] != 0.0)
    # attending to a single position makes the row's softmax flat -> no key/query gradient
    assert np.allclose(tk.grad, 0.0, atol=1e-9)

def test_sdpa_causal_first_row_copies_first_value():
    rng = np.random.default_rng(4)
    t, d = 6, 4
    q, k = rng.standard_normal((t, d)), rng.standard_normal((t, d))
    v = rng.standard_normal((t, d))

    out = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), mask=True).data
    assert np.allclose(out[0], v[0], atol=1e-9)

def test_sdpa_causal_output_ignores_later_positions():
    # perturbing v at the last position must leave every earlier output row unchanged
    rng = np.random.default_rng(5)
    t, d = 5, 4
    q, k, v = (rng.standard_normal((t, d)) for _ in range(3))

    out_a = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), mask=True).data
    v2 = v.copy()
    v2[-1] += 10.0
    out_b = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v2), mask=True).data

    assert np.allclose(out_a[:-1], out_b[:-1], atol=1e-9)
    assert not np.allclose(out_a[-1], out_b[-1])


# --- MultiHeadAttention ---

def _mha(k=8, heads=4, mask=True, seed=0):
    np.random.seed(seed)
    m = MultiHeadAttention(k, heads=heads, mask=mask)
    return m

def _project(m, x):
    # the per-head q/k/v for raw numpy input x, sliced out of the full projections
    return (x @ m.toqueries.W.data, x @ m.tokeys.W.data, x @ m.tovalues.W.data)

def test_mha_rejects_head_count_that_does_not_divide_k():
    MultiHeadAttention(12, heads=4)  # divides evenly
    with pytest.raises(AssertionError):
        MultiHeadAttention(10, heads=4)

def test_mha_forward_shape_and_params():
    m = _mha(k=8, heads=4)
    out = m(Tensor(np.random.randn(2, 5, 8)))
    assert out.shape == (2, 5, 8)
    # 3 projections (no bias) + unifyheads W and b
    assert len(m.parameters()) == 5

def test_mha_matches_explicit_per_head_reference():
    # slow, obviously-correct version: loop the heads, slice their columns, concat
    m = _mha(k=8, heads=4)
    b, t, k, h = 2, 5, 8, 4
    s = k // h
    x = np.random.randn(b, t, k)

    Q, K, V = _project(m, x)
    heads = []
    for i in range(h):
        sl = slice(i * s, (i + 1) * s)
        w = Q[..., sl] @ K[..., sl].swapaxes(-1, -2) / np.sqrt(s)
        w = w + np.triu(np.full((t, t), -1e9), k=1)
        w = np.exp(w - w.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        heads.append(w @ V[..., sl])
    expected = np.concatenate(heads, axis=-1) @ m.unifyheads.W.data + m.unifyheads.b.data

    assert np.allclose(m(Tensor(x)).data, expected, atol=1e-9)

def test_mha_heads_land_in_their_own_channel_block():
    # heads are independent streams; unifyheads is the only place they mix.
    # make it the identity and each channel block must be exactly that head's output.
    m = _mha(k=8, heads=4)
    s = 2
    m.unifyheads.W.data = np.eye(8)
    m.unifyheads.b.data = np.zeros(8)
    x = np.random.randn(1, 5, 8)
    out = m(Tensor(x)).data

    Q, K, V = _project(m, x)
    for i in range(4):
        sl = slice(i * s, (i + 1) * s)
        head_i = scaled_dot_product_attention(
            Tensor(Q[..., sl]), Tensor(K[..., sl]), Tensor(V[..., sl]), mask=True
        )
        assert np.allclose(out[..., sl], head_i.data, atol=1e-9), f"head {i} misplaced"

def test_mha_unmasked_is_permutation_equivariant():
    # with no positional encoding attention is a set operation - it cannot tell
    # what order the tokens arrived in, so permuting them permutes the outputs
    m = _mha(mask=False)
    x = np.random.randn(1, 6, 8)
    perm = np.random.permutation(6)

    out_then_permute = m(Tensor(x)).data[:, perm]
    permute_then_out = m(Tensor(x[:, perm])).data

    assert np.allclose(out_then_permute, permute_then_out, atol=1e-9)

def test_mha_causal_output_ignores_later_tokens():
    # defining property of a decoder: token i cannot see token j > i
    m = _mha(mask=True)
    x = np.random.randn(1, 6, 8)
    x_perturbed = x.copy()
    x_perturbed[:, -1] += 10.0

    out_a = m(Tensor(x)).data
    out_b = m(Tensor(x_perturbed)).data

    assert np.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-9)
    assert not np.allclose(out_a[:, -1], out_b[:, -1])

def test_mha_causal_gradient_does_not_reach_the_future():
    # same property as above, but exercising backward rather than forward
    m = _mha(mask=True)
    t = 6
    x = Tensor(np.random.randn(1, t, 8))
    out = m(x)

    i = 2  # differentiate output row i only
    upstream = np.zeros((1, t, 8))
    upstream[:, i] = 1.0
    (out * upstream).sum().backward()

    assert np.allclose(x.grad[:, i + 1:], 0.0, atol=1e-9)
    assert np.any(x.grad[:, : i + 1] != 0.0)

def test_mha_honours_the_mask_flag():
    m_masked = _mha(mask=True, seed=1)
    m_unmasked = _mha(mask=False, seed=1)
    x = Tensor(np.random.randn(1, 6, 8))
    # same seed -> same weights, so any difference is the mask itself
    assert np.allclose(m_masked.toqueries.W.data, m_unmasked.toqueries.W.data)
    assert not np.allclose(m_masked(x).data, m_unmasked(x).data)

def test_mha_backward_populates_all_parameter_grads():
    m = _mha()
    out = m(Tensor(np.random.randn(2, 5, 8)))
    out.sum().backward()
    for p in m.parameters():
        assert p.grad.shape == p.shape
    assert all(np.any(p.grad != 0.0) for p in m.parameters() if p is not m.unifyheads.b)
