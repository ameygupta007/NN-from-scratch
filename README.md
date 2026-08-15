# NN-from-scratch

A small neural network library built from scratch using NumPy (no PyTorch/TensorFlow), used to train to **98+% accuracy on MNIST**. 

Includes an autograd engine (`minigrad/engine.py`) with gradients that numerically match PyTorch on autograd tests.

Demo hosted at https://whatdigit.vercel.app, where you can draw a digit and see the model's predictions.


[![Demo](assets/demo.gif)](https://whatdigit.vercel.app)

## Files

- **Autograd engine** (`minigrad/engine.py`) — reverse-mode autodiff on a `Tensor` type with broadcasting, matmul, and higher-level ops including softmax and cross-entropy.
- **MLP module** (`minigrad/nn.py`) — `Module`, `Linear` and `MLP` abstractions built on `Tensor` with parameter tracking.
- **Optimizers** (`minigrad/optim.py`) - `SGD` optimizer (more to come), LR scheduling functions e.g. `step_decay`
- **MNIST example** (`examples/mnist/`) — data loader (`data.py`), augmentation (`augment.py`), training loop (`train.py`), and the notebook (`train_mnist.ipynb`) that ties them together end-to-end.
- **Browser demo** (`web/`) — trained weights exported to a binary blob and run client-side in JS to classify digits.
- **Tests** (`tests/`) — gradients checked against PyTorch, and other modules also checked.
- `examples/mnist/models/momentum-98%.npz` — trained weights for the best model so far (**98% test accuracy**). Load with `np.load`.

## How the autograd works

Every operation on a `Tensor` builds a node in a computation graph, recording its inputs and a local backward function. Calling `.backward()` on the output does a topological sort of the graph and walks it in reverse, applying the chain rule at each node.

![Computation graph visualisation](assets/node_viz.png)

*(Rendered from an earlier scalar prototype of the engine; the graph today has the same shape but with `Tensor` objects.)*

## Results

Iterating on the MNIST classifier:

| Change                          | Test accuracy |
| ------------------------------- | ------------- |
| Baseline MLP                    | ~93%          |
| + softmax cross-entropy loss    | ~94%          |
| + higher learning rate + decay  | ~97%          |
| + dropout + data augmentation   | >97.5%        |
| + momentum in SGD               | **98%**       |

Final model is a 784 -> 100 -> 10 MLP, with tanh activations, dropout_p = 0.2.

## Run it

Train:

Requires the MNIST dataset in `examples/mnist/data/`, in the original IDX format from e.g. [Kaggle mirror](https://www.kaggle.com/datasets/hojjatk/mnist-dataset):

```
examples/mnist/data/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes `minigrad` importable from the notebook and scripts
jupyter notebook examples/mnist/train_mnist.ipynb
```

(For running the tests, use `requirements-dev.txt` instead — adds `torch` and `pytest`.)

Serve the web demo:
```bash
cd web && python -m http.server 8000
# open http://localhost:8000
```

## What a transformer needed that MNIST didn't
All that was needed for MNIST was a functioning autograd and MLP with a training loop. This could be done in a Jupyter notebook, without much abstraction into Optimizer and Module classes. For the transformer I decided to refactor first and add these abstractions to make my life easier, making sure the MNIST training still ran unchanged (apart from extracting the training loop).

Additionally:
- new operations: layer_norm, reshape, concat, softmax, embedding
- needed to change backprop algorithm to be iterative rather than recursive, to avoid hitting the recursion limit.



## Next steps (ideas)
**Currently working on implementing a transformer.**

- General:
  - Optimise training: implement different optimisers
    - could try Exponential Moving Average for SGD instead of LR scheduling
  - Better autograd: support more ops, broadcasting edge cases, `no_grad` context.
- MNIST:
  - Try a CNN
  - L2 weight decay, early stopping, label smoothing
  - Train on Fashion MNIST
  - Training ergonomics: better CLI flags and info
  - Confusion matrix, per-class accuracy
  - Multiple digits


## Resources

Started from Andrej Karpathy's [micrograd walkthrough](https://www.youtube.com/watch?v=VMj-3S1tku0), then extended to tensors, an MLP training loop, augmentation, and the browser demo.

- [MiniTorch](https://minitorch.github.io) - structure of an ML library
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - transformer architecture
- 3Blue1Brown, for gaining intuition
  - [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M&t=2s)
  - [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- Peter Bloem, [Transformers from scratch](https://peterbloem.nl/blog/transformers) - talks about transformers and attention

Optimizers:
- Momentum:
  - [Sutskever et al., "On the importance of initialization and momentum in deep learning"](https://proceedings.mlr.press/v28/sutskever13.html)
  - [Understanding SGD with momentum - Piiyush Kashyap](https://medium.com/@piyushkashyap045/understanding-sgd-with-momentum-in-deep-learning-a-beginner-friendly-guide-0252ede605b4)
- Adam:
  - [Kingma & Ba, "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980)
- LR warmup: [Xiong et al.](https://arxiv.org/abs/2002.04745)
- 