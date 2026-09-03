# notes from implementation

- added `mean` to Tensor class
- changed `topo_sort` in `backward()` to iterative rather than recursive
- had to modify Tensor's matmul to accomodate batched matrix multiplication, required handling the special cases with 1D vectors as well
- used prenorm instead of postnorm, to reduce our need for warmup
- put activation functions in `functional.py` so that they can be passed into the MLP
- tinyshakespeare dataset: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
- adapt cross_entropy to more than 2D logits
- save/load functionality needed for checkpointing on transformers; since training runs are much longer and more complex than MNIST