// WebGPU (WGSL) translation of common/kernels.cl, used by the JavaScript benchmark only.
// Same algorithms, same arithmetic in the same order, same sizes; see common/SPEC.md "GPU".
//
// Differences that WebGPU forces:
//  - a fixed work-group size must be declared (64 = one AMD wavefront); OpenCL lets the driver pick.
//  - at most 65535 work-groups per dimension, so large launches use a 2-D grid of work-groups;
//    every kernel converts its position back to a linear index `i` and ignores i >= its limit.
//  - sizes arrive in a small uniform buffer instead of kernel arguments.
// Each kernel uses its own bind group (0, 1, 2) so the three can live in one file.

const WG: u32 = 64u;
const PEAK_ITERS: i32 = 1024;   // as in kernels.cl: FLOP per work-item = 1024 * 16 multiply-adds * 2
// multiply-adds are written `a * b + c`, exactly as in kernels.cl (see the note there about fma()).

fn linear_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WG;
}

// ---------------------------------------------------------------- fma_peak (bind group 0)
struct PeakParams { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> peak_out: array<f32>;
@group(0) @binding(1) var<uniform> peak_p: PeakParams;

@compute @workgroup_size(64)
fn fma_peak(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_id(gid, nwg);
    if (i >= peak_p.count) { return; }
    var a0 = f32(i) * 1e-7;
    var a1 = a0 + 0.1; var a2 = a0 + 0.2; var a3 = a0 + 0.3;
    var a4 = a0 + 0.4; var a5 = a0 + 0.5; var a6 = a0 + 0.6; var a7 = a0 + 0.7;
    let b = 0.999;
    let c = 0.001;
    for (var k = 0; k < PEAK_ITERS; k++) {
        a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
        a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
        a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
        a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
    }
    peak_out[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

// ---------------------------------------------------------------- copy4 (bind group 1)
// count = number of floats (multiple of 4); count / 4 work-items, one vec4 each.
struct CopyParams { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(1) @binding(0) var<storage, read_write> copy_out: array<vec4<f32>>;
@group(1) @binding(1) var<uniform> copy_p: CopyParams;
@group(1) @binding(2) var<storage, read> copy_in: array<vec4<f32>>;

@compute @workgroup_size(64)
fn copy4(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_id(gid, nwg);
    if (i >= copy_p.count / 4u) { return; }
    copy_out[i] = copy_in[i];
}

// ---------------------------------------------------------------- sgemm4x4 (bind group 2)
// C = A * B, n x n row-major; each work-item computes a 4x4 block of C; (n/4)^2 work-items.
struct GemmParams { count: u32, n: u32, pad0: u32, pad1: u32 }
@group(2) @binding(0) var<storage, read_write> gemm_c: array<vec4<f32>>;
@group(2) @binding(1) var<uniform> gemm_p: GemmParams;
@group(2) @binding(2) var<storage, read> gemm_a: array<f32>;
@group(2) @binding(3) var<storage, read> gemm_b: array<vec4<f32>>;

@compute @workgroup_size(64)
fn sgemm4x4(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let id = linear_id(gid, nwg);
    let n = gemm_p.n;
    let nb = n / 4u;
    if (id >= nb * nb) { return; }
    let r = (id / nb) * 4u;
    let c = (id % nb) * 4u;
    var acc0 = vec4<f32>(0.0); var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0); var acc3 = vec4<f32>(0.0);
    for (var k = 0u; k < n; k++) {
        let b = gemm_b[(k * n + c) / 4u];
        acc0 += gemm_a[(r + 0u) * n + k] * b;
        acc1 += gemm_a[(r + 1u) * n + k] * b;
        acc2 += gemm_a[(r + 2u) * n + k] * b;
        acc3 += gemm_a[(r + 3u) * n + k] * b;
    }
    gemm_c[((r + 0u) * n + c) / 4u] = acc0;
    gemm_c[((r + 1u) * n + c) / 4u] = acc1;
    gemm_c[((r + 2u) * n + c) / 4u] = acc2;
    gemm_c[((r + 3u) * n + c) / 4u] = acc3;
}
