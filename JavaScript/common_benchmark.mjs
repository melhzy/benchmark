#!/usr/bin/env node
// common_benchmark.mjs -- the shared C++ / Python / R / JavaScript benchmark, JavaScript implementation.
//
// Implements common/SPEC.md: the same tests, sizes (common/tests.csv) and output format as the
// C++, Python and R versions, so all four can be compared in analysis.ipynb.
// GPU tests use WebGPU (npm package `webgpu`, Google's Dawn on Vulkan) with common/kernels.wgsl,
// a line-by-line translation of the OpenCL kernels the other languages use.
// matmul_blas calls the system OpenBLAS (the same library C++ and R use) through the npm package
// `koffi` (a foreign-function interface), since Node.js has no BLAS of its own.
//
// Setup (once):   cd ~/benchmark/JavaScript && npm install
// Usage
//   node --expose-gc common_benchmark.mjs                 full run
//   node --expose-gc common_benchmark.mjs --quick         smaller sizes, quick check
//   node --expose-gc common_benchmark.mjs --verify        tiny identical sizes, for checksums
//   node --expose-gc common_benchmark.mjs --only cpu_single,mem_copy
//   node --expose-gc common_benchmark.mjs --out DIR       results directory
//
// Categories: cpu_single, cpu_multi, ram, gpu (see common/SPEC.md for every test).
// Results: <out>/js_<timestamp>.csv (one row per timed repetition) and <out>/js_<timestamp>_meta.csv.

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, openSync, writeSync, closeSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker, isMainThread, parentPort } from 'node:worker_threads';
import os from 'node:os';

const SCRIPT = fileURLToPath(import.meta.url);
const ROOT = dirname(dirname(SCRIPT));
const LANG_KEY = 'js';
const LANGUAGE = 'JavaScript';
const CATEGORIES = ['cpu_single', 'cpu_multi', 'ram', 'gpu'];
const PHI = 0.6180339887498949;
const PHI2 = 0.7548776662466927;
const MIB = 2 ** 20, GIB = 2 ** 30;
const PEAK_ITERS = 1024;
const WG = 64;                 // work-group size declared in kernels.wgsl
const MAX_GROUPS = 65535;      // WebGPU limit per dispatch dimension
const RESULT_COLUMNS = ['run_id', 'batch', 'language', 'mode', 'category', 'test', 'style', 'threads',
  'size', 'rep', 'seconds', 'work', 'unit', 'rate', 'check'];
const META_KEYS = ['run_id', 'batch', 'language', 'mode', 'language_version', 'build', 'blas',
  'numpy_version', 'cpu', 'physical_cores', 'logical_cpus', 'ram_gib', 'power',
  'platform_profile', 'cpu_temp_start_c', 'cpu_temp_end_c', 'started', 'elapsed_s',
  'host', 'machine', 'gpu_platform', 'gpu_device', 'gpu_compute_units', 'gpu_max_clock_mhz',
  'gpu_driver', 'skipped', 'failed'];

// ---------------------------------------------------------------------------
// Shared workloads (also used by the worker threads)
// ---------------------------------------------------------------------------

const frac = (x) => x - Math.floor(x);
const w = (i) => frac(i * PHI);
const w2 = (i) => frac(i * PHI2);

// Mandelbrot over rows y0..y1-1 of a W x W grid, exactly as in SPEC.md; returns total iterations.
function mandelRows(y0, y1, W) {
  let total = 0;
  for (let y = y0; y < y1; y++) {
    const ci = -1.25 + (2.5 * y) / W;
    for (let x = 0; x < W; x++) {
      const cr = -2.0 + (2.5 * x) / W;
      let zr = 0.0, zi = 0.0, it = 0;
      while (it < 100) {
        const t = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = t;
        it += 1;
        if (zr * zr + zi * zi > 4.0) break;
      }
      total += it;
    }
  }
  return total;
}

// ---------------------------------------------------------------------------
// System information
// ---------------------------------------------------------------------------

function readText(path) {
  try { return readFileSync(path, 'utf8').trim(); } catch { return null; }
}

function cpuinfoField(key) {
  return (readText('/proc/cpuinfo') || '').split('\n')
    .filter((line) => line.split(':')[0].trim() === key)
    .map((line) => line.slice(line.indexOf(':') + 1).trim());
}

function physicalCores() {
  const ids = cpuinfoField('physical id'), cores = cpuinfoField('core id');
  const pairs = new Set(cores.map((c, i) => `${ids[i]}/${c}`));
  return pairs.size || os.cpus().length;
}

function logicalCpus() {
  return typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length;
}

function meminfoKib(key) {
  for (const line of (readText('/proc/meminfo') || '').split('\n')) {
    if (line.startsWith(key + ':')) return Number(line.split(/\s+/)[1]);
  }
  return null;
}

function powerSource() {
  const base = '/sys/class/power_supply';
  if (!existsSync(base)) return '';
  const online = readdirSync(base).some((name) => readText(`${base}/${name}/online`) === '1');
  return online ? 'AC' : 'battery';
}

// CPU temperature from the first hwmon sensor of a known CPU driver, in order of preference.
const CPU_SENSORS = ['k10temp', 'zenpower', 'coretemp', 'cpu_thermal'];

function cpuTempC() {
  const base = '/sys/class/hwmon';
  if (!existsSync(base)) return null;
  const hwmons = readdirSync(base).sort();
  for (const want of CPU_SENSORS) {
    for (const name of hwmons) {
      if (readText(`${base}/${name}/name`) === want) {
        const raw = readText(`${base}/${name}/temp1_input`);
        if (raw && /^-?\d+$/.test(raw)) return Number(raw) / 1000;
      }
    }
  }
  return null;
}

// lowercase; every run of characters outside [a-z0-9] becomes "-"; no "-" at either end.
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');

// Machine id (SPEC.md): BENCH_MACHINE, else the DMI vendor + product name, else the hostname.
function machineId() {
  if (process.env.BENCH_MACHINE) return process.env.BENCH_MACHINE;
  const dmi = slug(`${(readText('/sys/class/dmi/id/sys_vendor') || '').trim()} ` +
                   `${(readText('/sys/class/dmi/id/product_name') || '').trim()}`);
  return dmi || slug(os.hostname());
}

// Replaces the user's home directory with "~", so no personal paths end up in results.
function tilde(value) {
  const home = os.homedir();
  const text = String(value);
  return home && home !== '/' ? text.split(home).join('~') : text;
}

// ---------------------------------------------------------------------------
// CPU tests. Each setup prepares data (not timed) and returns {run, check, work}.
// ---------------------------------------------------------------------------

const ident = (x) => x;

function fib(k) {
  return k < 2 ? k : fib(k - 1) + fib(k - 2);
}

function fibCalls(n) {               // 2 * F(n+1) - 1
  let a = 0, b = 1;
  for (let i = 0; i <= n; i++) [a, b] = [b, a + b];
  return 2 * a - 1;
}

function setupScalarLoop(n) {
  const run = () => {
    let s = 0.0;
    for (let i = 0; i < n; i++) s += Math.sqrt(i);
    return s;
  };
  return { run, check: ident, work: n };
}

function setupMandelbrot(W) {
  return { run: () => mandelRows(0, W, W), check: ident, work: W * W };
}

function setupFib(n) {
  return { run: () => fib(n), check: ident, work: fibCalls(n) };
}

function setupVectorMath(n) {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = 100 * w(i);
  const run = () => {
    let s = 0.0;
    for (let i = 0; i < n; i++) {
      const v = x[i];
      s += Math.sqrt(v) * v + 1.0;
    }
    return s;
  };
  return { run, check: ident, work: n };
}

function setupSort(n) {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = w(i);
  return {
    run: () => x.slice().sort(),                 // typed-array sort is numeric
    check: (y) => y[0] + y[Math.floor(n / 2)] + y[n - 1],
    work: n,
  };
}

function setupHashmap(n) {
  const keys = new Float64Array(n);
  for (let i = 0; i < n; i++) keys[i] = Math.floor(w(i) * 2 ** 31);
  const run = () => {
    const map = new Map();
    for (let i = 0; i < n; i++) map.set(keys[i], i);
    let total = 0;
    for (let i = 0; i < n; i++) total += map.get(keys[i]);
    return total;
  };
  return { run, check: ident, work: 2 * n };
}

function setupStringOps(n) {
  const run = () => {
    const parts = new Array(n);
    for (let i = 0; i < n; i++) parts[i] = `item${i}`;
    const joined = parts.join(',');
    const pieces = joined.split(',');
    let total = 0;
    for (let i = 0; i < pieces.length; i++) total += Number.parseInt(pieces[i].slice(4), 10);
    return total;
  };
  return { run, check: ident, work: n };
}

// ---------------------------------------------------------------------------
// RAM tests
// ---------------------------------------------------------------------------

function setupMemCopy(sizeMib) {
  const n = Math.floor(sizeMib * MIB / 8);
  const src = new Float64Array(n);
  for (let i = 0; i < n; i++) src[i] = w(i);
  const dst = new Float64Array(n);
  return {
    run: () => { dst.set(src); return dst; },
    check: (d) => d[n - 1],
    work: 2 * n * 8,
  };
}

function setupMemTriad(sizeMib) {
  const n = Math.floor(sizeMib * MIB / 8);
  const a = new Float64Array(n), b = new Float64Array(n).fill(1.0), c = new Float64Array(n).fill(2.0);
  const q = 3.0;
  const run = () => {
    for (let i = 0; i < n; i++) a[i] = b[i] + q * c[i];
    return a;
  };
  return { run, check: (r) => r[Math.floor(n / 2)], work: 3 * n * 8 };
}

function setupMemGather(sizeMib) {
  const n = Math.floor(sizeMib * MIB / 8);
  const m = Math.floor(n / 4);
  const t = new Float64Array(n);
  for (let i = 0; i < n; i++) t[i] = w(i);
  const idx = new Uint32Array(m);
  for (let j = 0; j < m; j++) idx[j] = Math.floor(w2(j) * n);
  const run = () => {
    let s = 0.0;
    for (let j = 0; j < m; j++) s += t[idx[j]];
    return s;
  };
  return { run, check: ident, work: m };
}

// -- BLAS: the system OpenBLAS through koffi (FFI) ------------------------------------------
// BENCH_OPENBLAS (full path), else the first library found in the usual Debian/Ubuntu, Fedora and
// Arch locations (the same list as Cpp/Makefile).
const OPENBLAS_CANDIDATES = [
  '/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0',
  '/usr/lib/x86_64-linux-gnu/libopenblas.so.0',
  '/usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblas.so.0',
  '/usr/lib64/libopenblasp.so.0',
  '/usr/lib64/libopenblas.so.0',
  '/usr/lib/libopenblas.so.0',
  '/usr/lib/libopenblas.so',
];
const CblasRowMajor = 101, CblasNoTrans = 111;

class BlasMissing extends Error {}

function findOpenBlas() {
  if (process.env.BENCH_OPENBLAS) return process.env.BENCH_OPENBLAS;
  return OPENBLAS_CANDIDATES.find((path) => existsSync(path)) || null;
}

async function loadBlas() {
  const path = findOpenBlas();
  if (!path) throw new BlasMissing('OpenBLAS not found (install libopenblas, or set BENCH_OPENBLAS=/path/to/libopenblas.so.0)');
  const { default: koffi } = await import('koffi');
  const lib = koffi.load(path);
  const config = lib.func('const char *openblas_get_config()');
  const threads = lib.func('int openblas_get_num_threads()');
  const dgemm = lib.func('void cblas_dgemm(int order, int transA, int transB, int M, int N, int K, ' +
    'double alpha, const double *A, int lda, const double *B, int ldb, double beta, _Inout_ double *C, int ldc)');
  return { dgemm, threads: threads(), config: `${config()} (via koffi ${koffi.version})` };
}

function setupMatmulBlas(blas, n) {
  const nn = n * n;
  const A = new Float64Array(nn), B = new Float64Array(nn), C = new Float64Array(nn);
  for (let i = 0; i < nn; i++) { A[i] = w(i); B[i] = w(nn + i); }     // row-major, as in SPEC.md
  const run = () => {
    blas.dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);
    return C;
  };
  const check = (c) => { let s = 0.0; for (let i = 0; i < c.length; i++) s += c[i]; return s; };
  return { run, check, work: 2 * n ** 3 };
}

const CPU_SETUPS = {
  scalar_loop: setupScalarLoop, mandelbrot: setupMandelbrot, fib_recursive: setupFib,
  vector_math: setupVectorMath, sort: setupSort, hashmap: setupHashmap, string_ops: setupStringOps,
  mem_copy: setupMemCopy, mem_triad: setupMemTriad, mem_gather: setupMemGather,
};

// ---------------------------------------------------------------------------
// GPU (WebGPU via Dawn) with common/kernels.wgsl
// ---------------------------------------------------------------------------

class Gpu {
  static async create() {
    let mod;
    try {
      mod = await import('webgpu');
    } catch {
      throw new Error("npm package 'webgpu' is not installed (run npm install in JavaScript/)");
    }
    Object.assign(globalThis, mod.globals);
    const gpu = mod.create([]);
    // BENCH_GPU_POWER = "high-performance" or "low-power" picks between GPUs; default: WebGPU's choice.
    const power = process.env.BENCH_GPU_POWER;
    const adapter = await gpu.requestAdapter(power ? { powerPreference: power } : {});
    if (!adapter) throw new Error('no WebGPU adapter found');
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    const g = new Gpu();
    g.instance = gpu;   // keep the Dawn instance alive: if it is garbage-collected, Dawn aborts the process
    g.adapter = adapter;
    g.device = device;
    g.webgpuVersion = JSON.parse(readFileSync(join(dirname(SCRIPT), 'node_modules', 'webgpu', 'package.json'), 'utf8')).version;
    device.pushErrorScope('validation');
    const module = device.createShaderModule({ code: readFileSync(join(ROOT, 'common', 'kernels.wgsl'), 'utf8') });
    const messages = (await module.getCompilationInfo()).messages.filter((m) => m.type === 'error');
    if (messages.length) throw new Error('kernels.wgsl: ' + messages.map((m) => `${m.lineNum}: ${m.message}`).join('; '));
    // entry point -> bind group index used by that kernel in kernels.wgsl
    g.groups = { fma_peak: 0, copy4: 1, sgemm4x4: 2 };
    g.pipelines = {};
    for (const name of Object.keys(g.groups)) {
      // synchronous create: createComputePipelineAsync never resolves in dawn.node 0.6.1
      g.pipelines[name] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: name } });
    }
    const err = await device.popErrorScope();
    if (err) throw new Error(`WebGPU: ${err.message}`);
    return g;
  }

  info() {
    const i = this.adapter.info;
    return {
      gpu_platform: `WebGPU (npm webgpu ${this.webgpuVersion}, Dawn, Vulkan)`,
      gpu_device: i.device || i.description || '',
      gpu_driver: i.description || '',
      gpu_compute_units: '',       // WebGPU does not expose these
      gpu_max_clock_mhz: '',
    };
  }

  buffer(bytes, usage, data) {
    const buf = this.device.createBuffer({ size: bytes, usage });
    if (data) this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  storage(bytes, data) {
    return this.buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, data);
  }

  uniform(values) {
    return this.buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Uint32Array([...values, 0, 0, 0, 0].slice(0, 4)));
  }

  // A launch closure: bind the buffers once; each call encodes + submits + waits.
  kernel(name, buffers, invocations) {
    const pipeline = this.pipelines[name];
    const group = this.groups[name];
    const bindGroups = [];
    for (let gi = 0; gi < group; gi++) {   // unused lower groups need an (empty) bind group
      bindGroups.push(this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(gi), entries: [] }));
    }
    bindGroups.push(this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(group),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    const groups = Math.ceil(invocations / WG);
    const gx = Math.min(groups, MAX_GROUPS), gy = Math.ceil(groups / gx);
    return async () => {
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      bindGroups.forEach((bg, i) => pass.setBindGroup(i, bg));
      pass.dispatchWorkgroups(gx, gy);
      pass.end();
      this.device.queue.submit([enc.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    };
  }

  // Read `count` floats starting at float index `first` (not timed; used for checks).
  async read(buf, count, first = 0) {
    const bytes = count * 4;
    const rb = this.buffer(Math.ceil(bytes / 4) * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, first * 4, rb, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return out;
  }

  // Run fn with validation / out-of-memory errors turned into exceptions.
  async guarded(fn) {
    this.device.pushErrorScope('out-of-memory');
    this.device.pushErrorScope('validation');
    const result = await fn();
    const errors = [await this.device.popErrorScope(), await this.device.popErrorScope()].filter(Boolean);
    if (errors.length) throw new Error(`WebGPU: ${errors.map((e) => e.message).join('; ')}`);
    return result;
  }

  async setupFp32Peak(n) {
    return this.guarded(async () => {
      const out = this.storage(n * 4);
      const run = this.kernel('fma_peak', [out, this.uniform([n])], n);
      await run();
      return { run, check: async () => (await this.read(out, 1))[0], work: n * PEAK_ITERS * 32, buffers: [out] };
    });
  }

  async setupBandwidth(sizeMib) {
    return this.guarded(async () => {
      const bytes = Math.floor(sizeMib * MIB), nf = bytes / 4;
      const src = this.storage(bytes, new Float32Array(nf).fill(1.5));
      const out = this.storage(bytes);
      const run = this.kernel('copy4', [out, this.uniform([nf]), src], nf / 4);
      await run();
      return { run, check: async () => (await this.read(out, 1, nf - 1))[0], work: 2 * bytes, buffers: [src, out] };
    });
  }

  async setupSgemm(n) {
    return this.guarded(async () => {
      const nn = n * n;
      const A = new Float32Array(nn), B = new Float32Array(nn);
      for (let i = 0; i < nn; i++) { A[i] = w(i); B[i] = w(nn + i); }
      const a = this.storage(nn * 4, A), b = this.storage(nn * 4, B), c = this.storage(nn * 4);
      const run = this.kernel('sgemm4x4', [c, this.uniform([nn, n]), a, b], (n / 4) ** 2);
      await run();
      const check = async () => {
        const C = await this.read(c, nn);
        let s = 0.0;
        for (let i = 0; i < nn; i++) s += C[i];
        return s;
      };
      return { run, check, work: 2 * n ** 3, buffers: [a, b, c] };
    });
  }

  async setupUpload(sizeMib) {
    return this.guarded(async () => {
      const bytes = Math.floor(sizeMib * MIB), nf = bytes / 4;
      const host = new Float32Array(nf).fill(2.5);
      const buf = this.storage(bytes);
      const run = async () => {
        this.device.queue.writeBuffer(buf, 0, host);
        await this.device.queue.onSubmittedWorkDone();
      };
      await run();
      return { run, check: async () => nf, work: bytes, buffers: [buf] };
    });
  }

  async setupDownload(sizeMib) {
    return this.guarded(async () => {
      const bytes = Math.floor(sizeMib * MIB), nf = bytes / 4;
      const buf = this.storage(bytes, new Float32Array(nf).fill(2.5));
      await this.device.queue.onSubmittedWorkDone();
      const rb = this.buffer(bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const host = new Float32Array(nf);
      const run = async () => {
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(buf, 0, rb, 0, bytes);
        this.device.queue.submit([enc.finish()]);
        await rb.mapAsync(GPUMapMode.READ);
        host.set(new Float32Array(rb.getMappedRange()));
        rb.unmap();
        return host;
      };
      await run();
      return { run, check: async (h) => h[0], work: bytes, buffers: [buf, rb] };
    });
  }
}

const GPU_SETUPS = {
  gpu_fp32_peak: 'setupFp32Peak', gpu_bandwidth: 'setupBandwidth', gpu_sgemm: 'setupSgemm',
  gpu_upload: 'setupUpload', gpu_download: 'setupDownload',
};

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

function fmtRate(perSec, unit) {
  for (const [factor, prefix] of [[1e9, 'G'], [1e6, 'M'], [1e3, 'k']]) {
    if (perSec >= factor) return `${(perSec / factor).toFixed(2).padStart(8)} ${prefix}${unit}/s`;
  }
  return `${perSec.toFixed(2).padStart(8)} ${unit}/s`;
}

// Like Python's '%.17g' for floats; integers printed exactly.
function fmtCheck(v) {
  v = Number(v);
  if (Number.isSafeInteger(v)) return String(v);
  let s = v.toPrecision(17);
  if (s.includes('e')) {
    const [mant, exp] = s.split('e');
    return (mant.includes('.') ? mant.replace(/\.?0+$/, '') : mant) + 'e' + exp;
  }
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

function csvField(v) {
  const s = v === null || v === undefined ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function stamp(d) {
  const p = (x) => String(x).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function isoLocal(d) {
  const p = (x) => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const collectGarbage = () => { if (typeof globalThis.gc === 'function') globalThis.gc(); };

function loadTests() {
  const lines = readFileSync(join(ROOT, 'common', 'tests.csv'), 'utf8').split('\n').filter((l) => l.trim());
  const head = lines[0].split(',').map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(',');
    return Object.fromEntries(head.map((h, i) => [h, (cells[i] || '').trim()]));
  });
}

class Runner {
  constructor(opts) {
    this.mode = opts.mode;
    this.started = new Date();
    const st = stamp(this.started);
    this.runId = `${LANG_KEY}_${st}`;
    this.batch = process.env.BENCH_BATCH || st;
    this.machine = machineId();
    this.outDir = resolve(opts.out);
    mkdirSync(this.outDir, { recursive: true });
    this.csvPath = join(this.outDir, `${this.runId}.csv`);
    this.metaPath = join(this.outDir, `${this.runId}_meta.csv`);
    this.skipped = [];
    this.failed = [];
    this.gpu = null;
    this.gpuError = null;
    this.fd = openSync(this.csvPath, 'w');
    writeSync(this.fd, RESULT_COLUMNS.join(',') + '\n');
  }

  sizeFor(spec) {
    return spec[this.mode === 'verify' ? 'verify' : `${this.mode}_${LANG_KEY}`];
  }

  repeatsFor(spec) {
    const r = Number(spec.repeats);
    return this.mode === 'verify' ? 1 : this.mode === 'quick' ? Math.min(r, 3) : r;
  }

  record(spec, threads, size, rep, seconds, work, check) {
    const row = [this.runId, this.batch, LANGUAGE, this.mode, spec.category, spec.test, spec.style,
      threads, size, rep, seconds, work, spec.unit, work / seconds, fmtCheck(check)];
    writeSync(this.fd, row.map(csvField).join(',') + '\n');
  }

  // Warm-up (if repeats > 1 and not verify), then timed repetitions; one row each.
  async measure(spec, setup, threads, size, label = '') {
    const { run, check, work } = setup;
    const repeats = this.repeatsFor(spec);
    if (repeats > 1 && this.mode !== 'verify') {
      const r = run();
      if (r instanceof Promise) await r;
    }
    const times = [];
    for (let rep = 1; rep <= repeats; rep++) {
      collectGarbage();
      const t0 = performance.now();
      let result = run();
      if (result instanceof Promise) result = await result;
      const seconds = (performance.now() - t0) / 1000;
      times.push(seconds);
      this.record(spec, threads, size, rep, seconds, work, await check(result));
    }
    const med = median(times);
    const name = spec.test + (label ? ` ${label}` : '');
    console.log(`  ${spec.category.padEnd(11)} ${name.padEnd(26)} size=${String(size).padEnd(10)} ` +
      `${med.toFixed(4).padStart(9)} s  ${fmtRate(work / med, spec.unit)}`);
  }

  skip(spec, reason, size = '') {
    this.skipped.push(`${spec.test}${size !== '' ? ` ${size}` : ''}: ${reason}`);
    const name = spec.test;
    console.log(`  ${spec.category.padEnd(11)} ${name.padEnd(26)} ${size !== '' ? `size=${String(size).padEnd(10)} ` : ''}skipped (${reason})`);
  }

  // -- tests that need their own loop ---------------------------------------------------

  async runParallelMandelbrot(spec, W) {
    const tasks = Math.min(96, W);
    const chunks = Array.from({ length: tasks }, (_, k) => [Math.floor((k * W) / tasks), Math.floor(((k + 1) * W) / tasks)]);
    const logical = logicalCpus();
    const counts = [...new Set([1, 2, 4, physicalCores(), 8, logical].filter((c) => c >= 1 && c <= logical))].sort((a, b) => a - b);
    for (const workers of counts) {
      const pool = await createPool(workers);
      try {
        const run = () => dispatch(pool, chunks, W);
        await this.measure(spec, { run, check: ident, work: W * W }, workers, W, `x${workers}`);
      } finally {
        await Promise.all(pool.map((p) => p.worker.terminate()));
      }
    }
  }

  async runMemAlloc(spec, sizeText) {
    const avail = (meminfoKib('MemAvailable') || 0) * 1024;
    for (const gText of String(sizeText).split(';')) {
      const g = Number(gText);
      const bytes = Math.floor(g * GIB);
      if (bytes > 0.4 * avail) {
        this.skip(spec, `larger than 40% of MemAvailable (${(avail / GIB).toFixed(1)} GiB)`, g);
        continue;
      }
      const n = Math.floor(bytes / 8);
      const run = () => {
        let x = new Float64Array(n);
        x.fill(1.0);
        let s = 0.0;
        for (let i = 0; i < n; i++) s += x[i];
        x = null;
        return s;
      };
      try {
        await this.measure(spec, { run, check: ident, work: 2 * bytes }, 1, g, `${g} GiB`);
      } catch (e) {
        if (e instanceof RangeError) this.skip(spec, `cannot allocate: ${e.message}`, g);
        else throw e;
      }
      collectGarbage();
    }
  }

  async initBlas() {
    if (this.blas || this.blasError) return;
    try {
      this.blas = await loadBlas();
    } catch (e) {
      this.blasError = (e instanceof BlasMissing ? e.message
        : `no BLAS: ${e.message} (run npm install in JavaScript/)`).replace(/\n/g, ' ');
    }
  }

  async initGpu() {
    if (this.gpu || this.gpuError) return;
    try {
      this.gpu = await Gpu.create();
    } catch (e) {
      this.gpuError = `${e.name}: ${e.message}`.replace(/\n/g, ' ');
    }
  }

  async runTest(spec) {
    const test = spec.test, sizeText = this.sizeFor(spec);
    if (test === 'parallel_mandelbrot') return this.runParallelMandelbrot(spec, Number(sizeText));
    if (test === 'mem_alloc') return this.runMemAlloc(spec, sizeText);
    if (test === 'matmul_blas') {
      await this.initBlas();
      if (!this.blas) return this.skip(spec, this.blasError);
      return this.measure(spec, setupMatmulBlas(this.blas, Number(sizeText)), this.blas.threads, Number(sizeText));
    }
    const size = Number(sizeText);
    if (test in GPU_SETUPS) {
      await this.initGpu();
      if (!this.gpu) return this.skip(spec, this.gpuError);
      const setup = await this.gpu[GPU_SETUPS[test]](size);
      try {
        return await this.measure(spec, setup, 0, size);
      } finally {
        for (const b of setup.buffers || []) b.destroy();
      }
    }
    if (!(test in CPU_SETUPS)) throw new Error(`no JavaScript implementation for ${test}`);
    return this.measure(spec, CPU_SETUPS[test](size), 1, size);
  }

  writeMeta(elapsed) {
    const memKib = meminfoKib('MemTotal');
    const meta = {
      run_id: this.runId, batch: this.batch, language: LANGUAGE, mode: this.mode,
      language_version: `Node.js ${process.version} (V8 ${process.versions.v8})`,
      build: [process.execPath, ...process.execArgv].join(' '), blas: this.blas ? this.blas.config : 'none', numpy_version: '',
      cpu: cpuinfoField('model name')[0] || os.cpus()[0]?.model || '',
      physical_cores: physicalCores(), logical_cpus: logicalCpus(),
      ram_gib: memKib ? (memKib / 2 ** 20).toFixed(1) : '',
      power: powerSource(), platform_profile: readText('/sys/firmware/acpi/platform_profile') || '',
      cpu_temp_start_c: this.tempStart ?? '', cpu_temp_end_c: cpuTempC() ?? '',
      started: isoLocal(this.started), elapsed_s: elapsed.toFixed(3), host: this.machine, machine: this.machine,
      skipped: this.skipped.join('; '), failed: this.failed.join('; '),
      ...(this.gpu ? this.gpu.info() : {}),
    };
    const lines = ['key,value', ...META_KEYS.map((k) => `${k},${csvField(tilde(meta[k] ?? ''))}`)];
    writeFileSync(this.metaPath, lines.join('\n') + '\n');
  }

  async run(specs) {
    this.tempStart = cpuTempC();
    if (specs.some((s) => s.category === 'gpu')) await this.initGpu();
    if (specs.some((s) => s.test === 'matmul_blas')) await this.initBlas();
    const gpuText = this.gpu ? `${this.gpu.info().gpu_device} (${this.gpu.info().gpu_driver})`
      : this.gpuError ? `none (${this.gpuError})` : '-';
    console.log('Common benchmark - JavaScript');
    console.log('='.repeat(72));
    for (const [key, value] of [
      ['Node.js', `${process.version}, V8 ${process.versions.v8} (${tilde(process.execPath)})`],
      ['Machine', this.machine],
      ['CPU', `${cpuinfoField('model name')[0] || '?'}, ${physicalCores()} physical / ${logicalCpus()} logical`],
      ['BLAS', this.blas ? `${this.blas.config}, ${this.blas.threads} threads` : `none (${this.blasError || 'not needed'})`],
      ['GPU', gpuText],
      ['Power', `${powerSource()}, profile ${readText('/sys/firmware/acpi/platform_profile')}`],
      ['CPU temp', this.tempStart !== null ? `${this.tempStart.toFixed(0)} C` : '?'],
      ['Mode', this.mode],
      ['Output', tilde(this.csvPath)],
    ]) console.log(`  ${key.padEnd(9)} ${value}`);
    if (typeof globalThis.gc !== 'function') console.log('  note: run with node --expose-gc so the garbage collector runs between repetitions');
    console.log();
    const start = performance.now();
    try {
      for (const spec of specs) {
        try {
          await this.runTest(spec);
        } catch (e) {
          const msg = `${e.name}: ${e.message}`.replace(/\n/g, ' ');
          this.failed.push(`${spec.test}: ${msg}`);
          console.log(`  ${spec.category.padEnd(11)} ${spec.test.padEnd(26)} FAILED: ${msg}`);
        }
      }
    } finally {
      closeSync(this.fd);
      const elapsed = (performance.now() - start) / 1000;
      this.writeMeta(elapsed);
      const tempEnd = cpuTempC();
      const m = Math.floor(elapsed / 60), s = Math.floor(elapsed % 60);
      console.log(`\nDone in ${m}m ${String(s).padStart(2, '0')}s` +
        (tempEnd !== null && this.tempStart !== null ? `; CPU temp ${this.tempStart.toFixed(0)} C -> ${tempEnd.toFixed(0)} C` : ''));
      console.log(`Results: ${tilde(this.csvPath)}\nMeta:    ${tilde(this.metaPath)}`);
      if (this.gpu) this.gpu.device.destroy();
    }
  }
}

// -- worker pool for parallel_mandelbrot ------------------------------------------------

async function createPool(workers) {
  const pool = [];
  for (let i = 0; i < workers; i++) {
    const worker = new Worker(SCRIPT);
    const entry = { worker, onResult: null };
    worker.on('message', (total) => entry.onResult(total));
    worker.on('error', (e) => entry.onError?.(e));
    pool.push(entry);
  }
  // Warm-up, not timed: every worker starts, loads this module and runs a tiny Mandelbrot once.
  await Promise.all(pool.map((p) => new Promise((res, rej) => {
    p.onResult = res; p.onError = rej;
    p.worker.postMessage({ y0: 0, y1: 8, W: 8 });
  })));
  return pool;
}

// Hand chunks to whichever worker is free; resolve with the total iteration count.
function dispatch(pool, chunks, W) {
  return new Promise((resolve, reject) => {
    let next = 0, done = 0, total = 0;
    const give = (p) => {
      if (next < chunks.length) {
        const [y0, y1] = chunks[next++];
        p.worker.postMessage({ y0, y1, W });
      }
    };
    for (const p of pool) {
      p.onError = reject;
      p.onResult = (t) => {
        total += t;
        done += 1;
        if (done === chunks.length) resolve(total);
        else give(p);
      };
    }
    pool.forEach(give);
  });
}

// ---------------------------------------------------------------------------

function usage(code) {
  console.log(`usage: node --expose-gc common_benchmark.mjs [--quick | --verify] [--only LIST] [--out DIR]

Common C++/Python/R/JavaScript benchmark - JavaScript implementation (common/SPEC.md).
  --quick     smaller sizes, quick check
  --verify    tiny sizes identical in all languages, for comparing checksums
  --only      comma-separated categories (${CATEGORIES.join(', ')}) and/or test names
  --out DIR   results directory (default: <benchmark root>/results/<machine>)
environment: BENCH_MACHINE (machine id), BENCH_GPU_POWER (high-performance | low-power),
             BENCH_OPENBLAS (path to libopenblas), BENCH_BATCH`);
  process.exit(code);
}

async function main() {
  const args = process.argv.slice(2);
  const opts = { mode: 'full', only: null, out: join(ROOT, 'results', machineId()) };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--quick' || a === '--verify') {
      const m = a.slice(2);
      if (opts.mode !== 'full' && opts.mode !== m) { console.error('error: --quick and --verify are mutually exclusive'); process.exit(2); }
      opts.mode = m;
    } else if (a === '--only') opts.only = args[++i];
    else if (a.startsWith('--only=')) opts.only = a.slice(7);
    else if (a === '--out') opts.out = args[++i];
    else if (a.startsWith('--out=')) opts.out = a.slice(6);
    else if (a === '-h' || a === '--help') usage(0);
    else { console.error(`error: unknown option ${a}`); usage(2); }
  }
  let specs = loadTests();
  if (opts.only) {
    const wanted = opts.only.split(',').map((s) => s.trim()).filter(Boolean);
    const known = new Set([...CATEGORIES, ...specs.map((s) => s.test)]);
    const unknown = wanted.filter((x) => !known.has(x));
    if (unknown.length) { console.error(`error: unknown test or category: ${unknown.join(', ')}`); process.exit(2); }
    specs = specs.filter((s) => wanted.includes(s.category) || wanted.includes(s.test));
  }
  process.on('SIGINT', () => { console.log('\nInterrupted.'); process.exit(130); });
  await new Runner(opts).run(specs);
  process.exit(0);   // Dawn may keep the event loop alive
}

// ---------------------------------------------------------------------------
// Entry point (last, so every const/class above is initialised)

if (!isMainThread) {
  // Worker thread for parallel_mandelbrot: compute the rows it is sent, reply with the total.
  parentPort.on('message', ({ y0, y1, W }) => parentPort.postMessage(mandelRows(y0, y1, W)));
} else {
  await main();
}
