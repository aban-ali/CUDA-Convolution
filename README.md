# Performance Comparison: CPU vs CUDA vs Optimized CUDA Kernels

## 1. Overview

This project benchmarks 2D convolution on CPU and GPU, demonstrating the impact of GPU parallelism and memory optimizations such as shared memory tiling and constant memory usage.

The goal is to understand not just speedup, but **how different memory access strategies affect performance**.

---

## 2. Approach

Five convolution implementations were evaluated:

* **CPU Baseline**
  A single-threaded C implementation used as the reference.

* **CUDA Base**
  A naive GPU implementation where each thread computes one output element using only global memory.

* **Optimized CUDA 1 (Cache-assisted Tiling)**
  Uses shared memory for tiles but does not explicitly load halo elements. Halo accesses fall back to global memory and rely on cache.

* **Optimized CUDA 2 (Full Tile Loading, Input-centric)**
  Each thread loads one input element into shared memory, including halo regions. Only inner threads compute outputs.

* **Optimized CUDA 3 (Full Tile Loading, Output-centric)**
  Each thread computes one output element. Some threads participate more in loading shared memory than others.

---

## 3. Optimization Techniques

The following optimizations were explored:

* Shared memory tiling to reduce global memory traffic
* Memory coalescing for efficient global memory access
* Constant memory for filter reuse
* Loop unrolling for small fixed-size kernels

---

## 4. Results

Configuration:

* Filter size: 3×3
* Image size: 2048×2048
* Tile size: 16×16

| Implementation   | Execution Time (ms) | Speedup (vs CPU) |
| ---------------- | ------------------- | ---------------- |
| CPU              | 132.4231            | 1.0×             |
| Base CUDA        | 0.7648              | 173.1×           |
| Optimized CUDA 1 | 0.7436              | 178.1×           |
| Optimized CUDA 2 | 0.4731              | 279.9×           |
| Optimized CUDA 3 | 0.6453              | 205.2×           |

<br>

The optimized implementations achieve up to **~280× speedup over CPU** and **~1.6× improvement over the naive CUDA version**.

---

## 5. Analysis

* **Base CUDA vs Optimized CUDA 1**
  Both show similar performance despite Optimized CUDA 1 using shared memory. This is because halo elements are not explicitly loaded into shared memory, leading to additional global memory accesses.
  These accesses are often uncoalesced and rely on cache, limiting the benefit of tiling.

* **Optimized CUDA 2 and 3**
  These implementations explicitly load halo regions into shared memory, eliminating the need for global memory access during computation.
  As a result:

  * Global memory traffic is significantly reduced
  * Memory access becomes more structured and predictable
  * Performance improves substantially

* **Why Optimized CUDA 2 is fastest**
  Although it wastes some threads (only inner threads compute), it benefits from:

  * simpler indexing
  * reduced control overhead
  * efficient shared memory usage

  This highlights an important trade-off:

  > reducing memory access cost can be more impactful than maximizing thread utilization.

* **General Observation**
  GPU performance is heavily influenced by memory behavior. Even small inefficiencies in memory access patterns can offset the benefits of parallelism.



<br>

#### Effect of Kernel Size on Performance

* The relative performance of optimized kernels changes with increasing filter size.

* For **small filters (e.g., 3×3)**, **Optimized CUDA 2** performs best.
  Although some threads remain idle, the overall execution is efficient due to:

  * lower control overhead
  * simpler indexing
  * efficient shared memory usage

* For **larger filters (e.g., 7×7)**, **Optimized CUDA 3 becomes comparable or faster** than Optimized CUDA 2.
  This is because:

  * In Optimized CUDA 2, the number of **active (useful) threads decreases significantly** as filter size increases
  * For example, with a 16×16 tile and a 7×7 filter, only about **10×10 = 100 threads out of 256 (~39%)** contribute to output computation
  * The remaining threads are idle during computation, reducing effective parallelism

* In contrast, Optimized CUDA 3 assigns **one output element per thread**, ensuring:

  * better thread utilization
  * more uniform workload distribution

* This leads to an important trade-off:

  > **Input-centric tiling (Kernel 2) is more efficient for small filters, while output-centric tiling (Kernel 3) scales better with larger filters.**


---

## 6. Setup & Usage

Compile and run:

```bash
nvcc <filename>.cu -o <executable>
./<executable>
```

Hardware used:

* GPU: GeForce GTX 1650
* NVIDIA Driver: 595.79
* CUDA Version: 13.2

---

## 7. Future Work

* Do a more thorough analysis through Nsight Compute
* Extend to multi-channel convolution
* Analyze performance scaling with image size
* Explore kernel fusion and operator-level optimizations
* Investigate tensor core acceleration
