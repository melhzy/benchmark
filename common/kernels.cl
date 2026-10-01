// OpenCL kernels shared by the C++, Python and R benchmarks (common/SPEC.md, "GPU").
//
// Every kernel uses the same calling convention, because the R OpenCL package requires it:
//   arg 0: __global float* out          (the output buffer)
//   arg 1: const unsigned int count      (number of elements in `out`; R's oclRun always passes this)
//   arg 2+: input buffers, then scalars (if any)
// No local memory and no fixed work-group size: the driver picks the local size,
// so every language launches exactly the same work.

#define PEAK_ITERS 1024  // loop rounds in fma_peak; FLOP per work-item = PEAK_ITERS * 16 multiply-adds * 2

// FP32 compute peak. One work-item per output element (count = work-items).
// 8 independent multiply-add chains, 2 rounds per iteration = 16 multiply-adds per iteration.
// Written as `a * b + c`, not fma(): Mesa's OpenCL (rusticl) emulates fma() in software, ~80x slower
// (measured: 6 vs 482 GFLOP/s), which made the full-size kernel exceed the GPU's ~2 s job timeout.
__kernel void fma_peak(__global float* out, const unsigned int n) {
    size_t i = get_global_id(0);
    if (i >= n) return;
    float a0 = (float)i * 1e-7f, a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
    float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
    const float b = 0.999f, c = 0.001f;
    for (int k = 0; k < PEAK_ITERS; k++) {
        a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
        a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
        a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
        a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
    }
    out[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// Device memory bandwidth: out = in, using float4 loads/stores.
// count = number of floats (a multiple of 4); launch count / 4 work-items.
__kernel void copy4(__global float* out, const unsigned int count, __global const float* in) {
    size_t i = get_global_id(0);
    if (i >= count / 4) return;
    vstore4(vload4(i, in), i, out);
}

// C = A * B for n x n row-major matrices; each work-item computes a 4x4 block of C.
// count = n * n (elements of C); n must be a multiple of 4; launch (n / 4) * (n / 4) work-items.
__kernel void sgemm4x4(__global float* C, const unsigned int count,
                       __global const float* A, __global const float* B, const unsigned int n) {
    size_t id = get_global_id(0);
    unsigned int nb = n / 4;
    if (id >= (size_t)nb * nb) return;
    unsigned int r = (unsigned int)(id / nb) * 4, c = (unsigned int)(id % nb) * 4;
    float4 acc0 = (float4)(0.0f), acc1 = (float4)(0.0f), acc2 = (float4)(0.0f), acc3 = (float4)(0.0f);
    for (unsigned int k = 0; k < n; k++) {
        float4 b = vload4(0, B + (size_t)k * n + c);
        acc0 += A[(size_t)(r + 0) * n + k] * b;
        acc1 += A[(size_t)(r + 1) * n + k] * b;
        acc2 += A[(size_t)(r + 2) * n + k] * b;
        acc3 += A[(size_t)(r + 3) * n + k] * b;
    }
    vstore4(acc0, 0, C + (size_t)(r + 0) * n + c);
    vstore4(acc1, 0, C + (size_t)(r + 1) * n + c);
    vstore4(acc2, 0, C + (size_t)(r + 2) * n + c);
    vstore4(acc3, 0, C + (size_t)(r + 3) * n + c);
}
