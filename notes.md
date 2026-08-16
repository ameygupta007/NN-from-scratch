# notes from implementation

- added `mean` to Tensor class
- changed `topo_sort` in `backward()` to iterative rather than recursive
- had to modify Tensor's matmul to accomodate batched matrix multiplication, required handling the special cases with 1D vectors as well