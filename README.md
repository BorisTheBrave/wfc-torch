This is a prototype implementation of the WaveFunctionCollapse algorithm using pytorch.

* It uses the AC3 algorithm for propagation
* overlapped is handled by reduction to the adjacent case
* min-entropy not supported
* parallelism, low

It works on CUDA, but much slower.