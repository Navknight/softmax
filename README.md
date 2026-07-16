# CUDA Softmax

Softmax for 10 million floats written from scratch in CUDA, benchmarked on a
Colab T4 against Triton, PyTorch, CuPy, JAX and TensorFlow.

- Subtract the max before exp so nothing overflows
- then 3 passes over the data - max, sum of exp, divide. Max and sum are reductions: each thread
folds a chunk into a register, warps merge their 32 values through shuffles,
blocks write partials to an array and a second kernel merges those. 
- Softmax is memory bound, so the only things that really matter are coalesced reads and
enough threads to keep the memory bus busy.
