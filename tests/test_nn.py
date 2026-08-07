import numpy as np
import pytest
from minigrad import Tensor, Module, Linear, MLP, dropout
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
    shapes = sorted(p.data.shape for p in params)
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
    assert y.data.shape == (5, 4)
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
    assert y.data.shape == (4, 3)

def test_linear_backward_populates_param_grads():
    lin = Linear(3, 2)
    X = Tensor(np.random.randn(4, 3))
    y = lin(X)
    y.sum().backward()
    assert lin.W.grad.shape == lin.W.data.shape
    assert lin.b.grad.shape == lin.b.data.shape
    assert np.any(lin.W.grad != 0.0)
    assert np.any(lin.b.grad != 0.0)


# --- MLP ---

def test_mlp_forward_shape_and_param_count():
    mlp = MLP(4, [8, 5, 3])
    X = np.random.randn(6, 4)
    y = mlp(X)
    assert y.data.shape == (6, 3)
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
