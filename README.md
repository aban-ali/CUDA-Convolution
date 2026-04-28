# Performance Comparison: CPU vs. CUDA vs. 3 Optimized CUDA versions

## 1. Overview

* Benchmarking Convolution operation in CPU and CUDA to demonstrate the speedup of GPU acceleration and memory optimization (shared memory/tiling).

## 2. The Approach
Brief description of five Convolution implementations:

* **CPU Baseline**: Single-threaded C implementation serving as the overall baseline model.
* **Simple CUDA**: Naive global memory implementation, it serves as the base CUDA implementation.
* **Base Tile CUDA**: Naive tiled implementation. Depends on cache for halo elements. Implemented using contant and shared memory.
* **Half Optimized CUDA**: Improvement on Base Tile CUDA implementation. Each thread loads an input tile element. Thus, lot of threads does not involve in output calculation.
* **Optimized CUDA**: Most optimized version. Improvement on Half Optimized CUDA version. Each thread calculates an output element.

## 3. Optimization Techniques

- Shared memory tiling
- Memory coalescing
- Constant memory usage
- Loop unrolling

---

## 4. Results
Use a table for quick scannability:

| Implementation | Execution Time (ms) | Speedup (vs CPU) |
|---|---|---|
| CPU | 8.3125 ms | 1.0x |
| Simple CUDA | 0.052813 ms | 157.4x |
| Base Tile CUDA| 0.050234 ms | 165.5x  |
| Half Optimized CUDA| 0.036029 ms | 230.7x |
| Optimized CUDA | 0.044717ms | 185.9x |

## 5. Key Insights
Short bullets on why the optimized version won:

* Reduced global memory latency via Shared Memory.
* Improved Occupancy by tuning dim3 block sizes.
* Eliminated Bank Conflicts.

Would you like a specific table format or a list of hardware specs to include for more professional credibility?
#### 5. Analysis (this is the gold section)

Explain:

why naive is slow
how shared memory reduces global loads
how coalescing improves bandwidth
why optimized is faster

## 6. Setup & Usage
Provide a one-liner to build and run:

`nvcc conv_cuda_half_optim.cu -o benchmark && ./benchmark`

List hardware used (e.g., RTX 3080, Intel i7-12700K) so the numbers have context.

## 7. Future Work
- multi-channel convolution
- larger kernels
- tensor cores
- fusion
