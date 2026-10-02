# Common benchmark specification (C++, Python, R, JavaScript)

One set of tests, implemented the same way in four languages, so results can be compared
directly and analysed together in `analysis.ipynb`. This file is the contract: if an
implementation disagrees with it, the implementation is wrong.

Files:

| Path | Purpose |
|---|---|
| `common/tests.csv` | test list, sizes per language and mode, repeats |
| `common/kernels.cl` | OpenCL kernels used by the C++, Python and R GPU sections |
| `common/kernels.wgsl` | the same kernels translated to WebGPU (WGSL), for JavaScript |
| `Cpp/common_benchmark.cpp` (+ `Cpp/Makefile`) | C++ implementation |
| `Python/common_benchmark.py` | Python implementation (NumPy + pyopencl) |
| `R/common_benchmark.R` | R implementation |
| `JavaScript/common_benchmark.mjs` (+ `package.json`) | JavaScript implementation (Node.js) |
| `run_all.sh` | runs all four on Linux |
| `run_all.ps1` (+ `run_all.cmd`), `setup_windows.ps1`, `common/windows_tools.ps1` | the same on Windows, and its setup |
| `results/` | output of every run (shared by all languages) |

## 1. Command line (identical for all four)

```
<program> [--quick | --verify] [--only LIST] [--out DIR]
```

- default mode is **full**; `--quick` uses the `quick_*` sizes; `--verify` uses the `verify` size
  (identical in all languages) so checksums can be compared across languages.
- `--only` takes a comma-separated list of categories and/or test names
  (e.g. `--only cpu_single,mem_copy`).
- `--out` sets the results directory (default: `<benchmark root>/results/<machine>`, see below).
- `common/tests.csv` and `common/kernels.cl` are found relative to the program's own location
  (`<root>/common/`), so the programs work from any current directory.
- Environment variable `BENCH_BATCH` (set by `run_all.sh`) is copied into the output; if unset,
  the batch is the run's own timestamp.

### Machines and environment variables

Results are kept per machine, in `results/<machine>/`. The **machine id** is computed the same way
by all four programs and the runners: `BENCH_MACHINE` if set, otherwise the firmware (DMI / SMBIOS)
vendor and product name, otherwise the hostname, made into a *slug*: lower-cased, with every run of
characters other than `[a-z0-9]` replaced by `-` and leading/trailing `-` removed. Vendor and product
come from `/sys/class/dmi/id/sys_vendor` and `/sys/class/dmi/id/product_name` on Linux, and from the
registry key `HKLM\HARDWARE\DESCRIPTION\System\BIOS` (`SystemManufacturer`, `SystemProductName`) on
Windows. The id is slug(vendor + " " + product), except that when slug(product) equals slug(vendor) or
starts with slug(vendor) + "-", it is slug(product) alone. Examples: `dell-inc-inspiron-14-7425-2-in-1`
("Dell Inc." + "Inspiron 14 7425 2-in-1") and `alienware-m16-r1` ("Alienware" + "Alienware m16 R1").

| variable | effect |
|---|---|
| `BENCH_MACHINE` | machine id / results folder (`run_all.sh --machine NAME` sets it) |
| `BENCH_BATCH` | batch id shared by one `run_all.sh` invocation |
| `BENCH_PYTHON` | Python interpreter the runner uses (default: `python3` on `PATH`; Windows: `python`, else the `py` launcher) |
| `BENCH_RSCRIPT` | Windows: `Rscript.exe` to use (default: on `PATH`, else the newest R in the registry / Program Files) |
| `BENCH_GPU` | OpenCL (C++, Python, R): use the first GPU whose name contains this text (case-insensitive); none matches → GPU tests skipped, listing the devices |
| `BENCH_GPU_POWER` | JavaScript/WebGPU: `high-performance` or `low-power` adapter preference |
| `BENCH_OPENBLAS` | full path of the OpenBLAS shared library (C++ build and JavaScript), if not found automatically |
| `BENCH_OPENCL_SDK` | Windows: OpenCL SDK folder (`include\CL\cl.h`, `lib\OpenCL.lib`) for the C++ build; default `deps\opencl-sdk` |
| `RUSTICL_ENABLE` | Mesa OpenCL drivers to expose; default `radeonsi,iris` (AMD and Intel GPUs) |

OpenBLAS is searched (C++ at build time, JavaScript at run time) in this order:
`/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0`, `/usr/lib/x86_64-linux-gnu/libopenblas.so.0`,
`/usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblas.so.0`, `/usr/lib64/libopenblasp.so.0`,
`/usr/lib64/libopenblas.so.0`, `/usr/lib/libopenblas.so.0`, `/usr/lib/libopenblas.so`; on Windows:
`deps\openblas\bin\libopenblas.dll` (installed by `setup_windows.ps1`), `C:\OpenBLAS\bin\libopenblas.dll`,
`C:\msys64\ucrt64\bin\libopenblas.dll`, `C:\msys64\mingw64\bin\libopenblas.dll`. If none exists,
`matmul_blas` is skipped with the reason.

**No personal data in results** (they are committed to a public repository): the meta value `host`
is the machine id, never the real hostname, and the user's home directory is written as `~` in
every value (Windows: `%USERPROFILE%`, written with `\` or `/`).

## 2. Sizes, repeats and timing

- Size for a test = column `full_<lang>`, `quick_<lang>` or `verify` of `tests.csv`
  (`<lang>` is `cpp`, `python`, `r` or `js`). Loop-style tests use smaller sizes for Python and R
  because they are slower; results are always reported as **rates** (work per second), which do
  not depend on size.
- Repeats: full = `repeats`, quick = min(`repeats`, 3), verify = 1.
- If repeats > 1 and the mode is not verify, run **one untimed warm-up** first.
- Before each timed repetition: run the garbage collector (Python `gc.collect()`, R `gc()`),
  not timed. (JavaScript: `globalThis.gc()`, available because Node runs with `--expose-gc`.)
- Use the highest-resolution wall clock available (C++ `std::chrono::steady_clock`,
  Python `time.perf_counter()`, JavaScript `performance.now()`, R `Sys.time()` differences — not `system.time()`, which has
  1 ms resolution).
- Data setup (generating inputs) is **not** timed unless the test says so.
- A test that fails must not stop the run: print the error, record its name in the meta key
  `failed`, continue.

### Shared input data (Weyl sequence)

Inputs must be identical in every language, so they are not generated with each language's
random number generator. Instead:

```
w(i)  = frac(i * 0.6180339887498949)        for i = 0, 1, 2, ...
w2(i) = frac(i * 0.7548776662466927)
frac(x) = x - floor(x)
```

computed in IEEE double precision (vectorised in R/NumPy is fine; the result is bit-identical).

## 3. Output files

Each run writes two CSV files to the results directory, with
`run_id = <lang>_<YYYYmmdd-HHMMSS>` (`lang` = `cpp`, `python`, `r`, `js`; local time at start):

**`<run_id>.csv`** — one row per timed repetition, columns exactly:

```
run_id,batch,language,mode,category,test,style,threads,size,rep,seconds,work,unit,rate,check
```

| column | meaning |
|---|---|
| language | `C++`, `Python`, `R` or `JavaScript` |
| mode | `full`, `quick` or `verify` |
| category, test, style, unit | copied from `tests.csv` |
| threads | 1 for single-thread tests; worker count for `parallel_mandelbrot`; thread count for `mem_triad` rows; BLAS thread count for `matmul_blas`; 0 for GPU tests |
| size | the size value used (N, W, n, MiB or GiB, as defined per test) |
| rep | 1..repeats (the warm-up is not recorded) |
| seconds | wall-clock time of this repetition |
| work | units of work done (formula per test below) |
| rate | work / seconds |
| check | checksum of the result (formula per test), printed with 17 significant digits |

**`<run_id>_meta.csv`** — columns `key,value`, with at least these keys (empty value if unknown):

```
run_id, batch, language, mode, language_version, build, blas, numpy_version,
cpu, physical_cores, logical_cpus, ram_gib, power, platform_profile,
cpu_temp_start_c, cpu_temp_end_c, started, elapsed_s, host, machine,
gpu_platform, gpu_device, gpu_compute_units, gpu_max_clock_mhz, gpu_driver,
skipped, failed
```

- `build`: C++ compiler + flags; Python executable path; R `R.home()`.
- `blas`: BLAS library in use (C++: `openblas_get_config()`; Python: from `numpy.show_config`;
  R: `sessionInfo()$BLAS`; JavaScript: `none`).
- `cpu` from `/proc/cpuinfo` "model name"; `physical_cores` = number of distinct
  ("physical id", "core id") pairs in `/proc/cpuinfo` (do **not** use R's
  `detectCores(logical = FALSE)`, which is wrong on Linux); `logical_cpus` = online CPUs.
  Windows: `cpu` = registry `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString`;
  `physical_cores` = processor cores from `GetLogicalProcessorInformationEx` (R: `detectCores(logical = FALSE)`,
  which is right on Windows; JavaScript: the sum of `NumberOfCores` of `Win32_Processor`).
- `power`: `AC` or `battery` (any `/sys/class/power_supply/*/online` == 1 → AC; Windows: the AC line
  status from `GetSystemPowerStatus`, or AC when there is no battery).
- `platform_profile`: `/sys/firmware/acpi/platform_profile`; Windows: the power mode for the current
  power source (registry `...\Control\Power\User\PowerSchemes`, `ActiveOverlayAcPowerScheme` /
  `ActiveOverlayDcPowerScheme`), written as `best power efficiency`, `balanced` or `best performance`.
- CPU temperature: `temp1_input` / 1000 of the first hwmon whose `name` is, in this order of
  preference, `k10temp`, `zenpower`, `coretemp`, `cpu_thermal`. Windows offers no CPU temperature
  to programs without administrator rights: the values are empty there.
- `MemAvailable` (used by `mem_alloc`) and `ram_gib`: `/proc/meminfo`; Windows: `GlobalMemoryStatusEx`
  (`ullAvailPhys`, `ullTotalPhys`), or the equivalent `Win32_OperatingSystem` values (R) and
  `os.freemem()` / `os.totalmem()` (JavaScript).
- `skipped`: `;`-separated `test: reason` items (e.g. GPU missing, size too big for RAM).

**Files written by the runner** (`run_all.sh`, or `run_all.ps1` on Windows; not by the programs), in
the same folder:

- `system_<batch>.csv` (`key,value`): machine id, suite version (`suite_commit` = `git describe`,
  `suite_tests_sha` = first 12 hex digits of the SHA-256 of `tests.csv`, `kernels.cl` and `kernels.wgsl`
  concatenated, with LF line endings — compare machines only when this matches), vendor/product, OS,
  kernel, CPU model, max clock, cores, SMT, instruction-set flags, L3 size, RAM size/type/speed/modules
  (from `udevadm`, no root needed), GPUs (`lspci`), OpenCL devices, power source and profile, governor,
  transparent huge pages, tool paths. Windows fills the same keys from CIM and the registry (RAM type and
  speed from `Win32_PhysicalMemory`, the power mode as above, `governor` = the power plan) and adds
  `cpu_base_mhz`. Both runners then add what **`Cpp/hw_probe`** (built with the C++ program) reports:
  `cpu_flags` (Windows), on hybrid CPUs `performance_cores` / `efficiency_cores` and `cpu_e_max_mhz`,
  `cpu_max_mhz` where the OS does not report it (Windows; Linux without cpufreq) with
  `cpu_max_mhz_source`, and for an NVIDIA GPU `gpu_cuda_name`, `gpu_sm_count`, `gpu_boost_clock_mhz`,
  `gpu_memory_clock_mhz`, `gpu_memory_bus_bits` and `gpu_memory_gbs` (= 2 × memory clock × bus width / 8,
  from the CUDA driver that comes with NVIDIA's driver). hw_probe *measures* a core's clock: a chain of
  dependent register-to-register additions (one cycle each) on a thread pinned to one core of each type,
  best of 25 samples after 0.3 s of load. After the run, `gpu_clock_max_logged_mhz` = the highest GPU
  clock in the sensor log is appended. The notebook and the results page compute each machine's
  theoretical peaks from this file, taking the highest measured clocks over all of a machine's snapshots
  (a measured boost clock depends on the load at that moment), and for the GPU the higher of the driver's
  clock and the highest logged one.
- `sensors_<batch>.csv`: one row per second — phase (language running), CPU temperature and clocks,
  GPU temperature/load/clock (amdgpu sysfs, or `nvidia-smi` on NVIDIA), RAM in use. On Windows the
  clocks are the effective clock (base clock × the `% Processor Performance` counter, as Task Manager
  shows it) and the temperature column is empty.

**Optional, written by hand: `hardware.csv`** (`key,value,source`) — corrections that override
`system_<batch>.csv` when peaks are computed (any of its keys, e.g. `gpu_memory_gbs` for a discrete
non-NVIDIA GPU, whose memory bus nothing reports). `source` says where each value comes from.

Console output: a short header (language, version, CPU, BLAS, GPU device, mode, output path),
then one line per test with size, median time and median rate, e.g.

```
  cpu_single  scalar_loop          size=20000000     0.812 s     24.6 M iter/s
```

## 4. Tests

`style` tells how each language must implement the test:

- **loop**: explicit element-by-element code written in the language itself
  (C++: plain loops; Python: pure Python, no NumPy; R: plain R inside a function, which R
  byte-compiles).
- **vectorized**: whole-array operations (C++: a plain loop over an array — the compiler may
  vectorise it; Python: NumPy; R: base R vector operations).
- **builtin**: the language's standard library facility named in the test.
- **parallel**: the language's standard multi-process / multi-thread facility.
- **blas**: matrix multiply through BLAS.
- **gpu**: OpenCL, using `common/kernels.cl` unchanged (JavaScript: WebGPU with `common/kernels.wgsl`).

JavaScript has no array-math library or BLAS in its standard library, so: **vectorized** tests
are plain loops over typed arrays (`Float64Array`), like C++; **blas** (`matmul_blas`) calls
`cblas_dgemm` in the same system OpenBLAS as C++ and R, through the npm package `koffi` (a
foreign-function interface; no copying of the arrays); **parallel** uses `worker_threads`.

Indices below are 0-based. R implementations translate to 1-based indexing as needed.
JavaScript per test: `Math.sqrt` loop; mandelbrot as written; `vector_math` one loop over a
`Float64Array`; `sort` = `x.slice().sort()` on a `Float64Array` (numeric, no comparator);
`hashmap` = `Map` with number keys; `string_ops` = template literals, `Array.join`,
`String.split`, `Number.parseInt`; `mem_copy` = `dst.set(src)`; `mem_triad` = loop into the
existing `a` (threads = 1 only); `mem_gather` = loop with a `Uint32Array` of indices;
`mem_alloc` = `new Float64Array(n)` + `fill(1)` + summing loop, then drop the reference.

### cpu_single — one CPU core

**scalar_loop** (size N, style loop) — timed:
```
s = 0.0
for i in 0..N-1: s += sqrt(double(i))
```
work = N (iter), check = s.
Python: `for i in range(N)` with `sqrt` bound to a local name. R: `for (i in 0:(N - 1))`.

**mandelbrot** (size W, style loop) — timed, W×W pixels:
```
total = 0
for y in 0..W-1:
  ci = -1.25 + (2.5 * y) / W
  for x in 0..W-1:
    cr = -2.0 + (2.5 * x) / W
    zr = 0.0; zi = 0.0; it = 0
    while it < 100:
      t  = zr*zr - zi*zi + cr
      zi = 2.0*zr*zi + ci
      zr = t
      it += 1
      if zr*zr + zi*zi > 4.0: break
    total += it
```
Use separate real/imaginary doubles in every language (not a complex type), evaluated in exactly
this order. work = W*W (px), check = total.

**fib_recursive** (size n, style loop) — timed: `fib(n)` with
`fib(k) = k if k < 2 else fib(k-1) + fib(k-2)`, written as a plain recursive function.
work = 2*F(n+1) - 1 calls (F = Fibonacci numbers), check = fib(n).
(The C++ compiler may restructure the recursion; the rate is then "effective calls/s".)

**vector_math** (size N, style vectorized) — setup: `x[i] = 100 * w(i)`. Timed:
`s = sum(sqrt(x) * x + 1.0)` (C++: one loop accumulating into a double; NumPy:
`np.sum(np.sqrt(x) * x + 1.0)`; R: `sum(sqrt(x) * x + 1)`). work = N (elem), check = s.

**sort** (size N, style builtin) — setup: `x[i] = w(i)`. Timed: produce a sorted copy
(C++: copy into a vector + `std::sort`; NumPy: `np.sort(x)`; R: `sort(x)`).
work = N (item), check = y[0] + y[N/2 rounded down] + y[N-1].

**hashmap** (size N, style builtin) — setup: integer keys `k[i] = floor(w(i) * 2^31)`.
Timed: insert `map[k[i]] = i` for i in order (later writes overwrite), then
`total = sum over i of map[k[i]]`. No pre-sizing of the table.
C++ `std::unordered_map<int64_t, int64_t>`; Python `dict`; R an environment
(`new.env(hash = TRUE)`, `e[[key]] <- value`, `e[[key]]`) — R environments only accept string
keys, so R converts keys with `as.character` during setup (not timed).
work = 2N (op), check = total.

**string_ops** (size N, style builtin) — timed, all steps:
1. build N strings `"item" + decimal(i)` for i in 0..N-1
2. join them with `","` into one string
3. split that string on `","`
4. parse the number after `"item"` in each piece and sum them

C++: `std::to_string`, `std::string` operations; Python: f-strings, `str.join`, `str.split`,
`int()` (pure Python); R: `paste0`, `paste(collapse = ",")`, `strsplit(fixed = TRUE)`,
`substring`, `as.integer`. work = N (item), check = the sum (= N(N-1)/2).

### cpu_multi — all CPU cores

**parallel_mandelbrot** (size W, style parallel) — the `mandelbrot` computation over W×W,
with the rows split into T = min(96, W) contiguous chunks of nearly equal size, distributed to
workers. Run once (repeats = 1, no warm-up) for each worker count in
{1, 2, 4, P, 8, L} ∩ [1, L], sorted, without duplicates (P = physical cores, L = logical CPUs).
C++: `std::thread` workers pulling chunk indices from an atomic counter; Python:
`multiprocessing.Pool(workers)` running the pure-Python chunk function (pool start-up is
**not** timed: create the pool, run one trivial task per worker, then time `pool.map`);
JavaScript: a pool of `worker_threads` created and warmed up before timing (like Python), chunk
indices handed to whichever worker is free;
R: `parallel::mclapply(mc.cores = workers)` (R forks inside the call, so its start-up **is**
included — that is how R works). Windows cannot fork: there Python's pool starts its workers with
*spawn* (still not timed), and R uses a socket cluster (`makeCluster(workers)`, `mandel_rows`
exported, one call to every worker) created before timing, then times
`parLapplyLB(chunk.size = 1)`. threads = workers, work = W*W (px),
check = total iterations (same as `mandelbrot` with the same W).

**matmul_blas** (size N, style blas) — setup (double precision, row-major definition):
`A[i][j] = w(i*N + j)`, `B[i][j] = w(N*N + i*N + j)` (R: build with `byrow = TRUE`).
Timed: `C = A × B` (C++: `cblas_dgemm` from the system OpenBLAS; NumPy: `A @ B`; R: `A %*% B`).
BLAS uses its default thread count (all logical CPUs); threads = that count.
work = 2·N³ (FLOP), check = sum of all elements of C.

### ram — memory

Sizes in MiB / GiB of float64 data; n = number of doubles.

**mem_copy** (size S MiB, style builtin) — setup: `src[i] = w(i)`, `dst` allocated and filled
with 0. Timed: copy src into the existing dst (C++: `std::memcpy`; NumPy:
`np.copyto(dst, src)`; R: `dst[] <- src`). work = 2·S·2^20 bytes (read + write),
check = dst[n-1].

**mem_triad** (size S MiB per array, style vectorized) — STREAM triad. setup: three arrays of
S MiB, `b` filled with 1.0, `c` with 2.0, scalar `q = 3.0`. Timed: `a = b + q * c`
(C++: loop writing into existing `a`; NumPy: `a = b + q * c`; R: `a <- b + q * c` — the last
two allocate temporaries, which is part of their cost).
C++ reports threads = 1 **and** additional rows with threads = P and threads = L (the loop split
across `std::thread`s); Python and R report threads = 1 only.
work = 3·S·2^20 bytes (STREAM convention, regardless of temporaries), check = a[n/2] (= 7.0).

**mem_gather** (size S MiB table, style vectorized) — setup: table `t[i] = w(i)` (n elements),
m = n / 4 indices `idx[j] = floor(w2(j) * n)` (R: +1). Timed: `s = sum(t[idx])`
(C++: loop; NumPy: `t[idx].sum()`; R: `sum(t[idx])`). work = m (access), check = s.

**mem_alloc** (sizes: `;`-separated GiB list, style builtin) — for each size G, timed: allocate
an array of G GiB of doubles, fill it with 1.0, sum it, release it
(C++: `new double[n]` (no zeroing) + `std::fill` + `std::accumulate`, then `delete[]`;
NumPy: `np.empty(n)`, `x.fill(1.0)`, `x.sum()`, `del x`; R: `x <- numeric(n)`, `x[] <- 1`,
`sum(x)`, `rm(x)` (R's `gc()` after timing)). Skip sizes larger than 40 % of `MemAvailable`
(from `/proc/meminfo`) at the start of the test, and list them in `skipped`.
repeats = 1, no warm-up; size = G, work = 2·G·2^30 bytes, check = the sum (= n).

### gpu — the GPU through OpenCL (JavaScript: WebGPU)

Device: the first OpenCL device of type GPU on any platform, or the first whose name contains
`BENCH_GPU`. If there is none (or OpenCL is not available for the language), skip all GPU tests
and say why in `skipped`. The programs set `RUSTICL_ENABLE=radeonsi,iris` if it is unset, so Mesa's
rusticl exposes AMD and Intel GPUs (NVIDIA's driver brings its own OpenCL).

Not timed: context, queue, program build, buffer creation and input upload (except in
`gpu_upload`). Each timed repetition = enqueue + wait for completion (`clFinish` or a blocking
call). Float32 throughout; the kernels' calling convention is described in `kernels.cl`
(argument 2 is always the output element count, because R's `oclRun` passes it automatically).
R's `oclRun` cannot wait for a kernel, so R ends each timed launch with a 1-element blocking read,
and it allocates a fresh output buffer on every launch (included in R's times).

**GPU warm-up** (added with Windows support): before the first GPU test, except in verify mode, run
`fma_peak` with 2^20 work-items back to back for 2 seconds, untimed, and print one `warm-up` line. A
discrete laptop GPU (NVIDIA) idles at a few hundred MHz and needs a sustained load to reach its working
clock; without this, a test's single warm-up launch is too short and the measured rate depends on the
GPU's power state rather than on the language. (Integrated GPUs are unaffected: the Linux results of
the AMD iGPU, measured before this rule, have identical times in every repetition.)

**JavaScript (WebGPU)**: Node.js has no maintained OpenCL binding, so it uses the npm package
`webgpu` (Google's Dawn: Vulkan backend on Linux, Direct3D 12 on Windows) and `common/kernels.wgsl`, a line-by-line
translation of `kernels.cl` (differences listed at the top of that file). Timed kernel launch =
encode + `queue.submit` + `await queue.onSubmittedWorkDone()`. Upload = `queue.writeBuffer` +
`await onSubmittedWorkDone()`. Download = `copyBufferToBuffer` into a `MAP_READ` buffer +
`mapAsync` + copying the mapped range into a host `Float32Array`. Request the adapter's
maximum buffer limits so the full sizes fit. GPU meta values come from `adapter.info`.

**gpu_fp32_peak** (size n work-items) — kernel `fma_peak`, global size n, output n floats.
work = n·1024·32 FLOP (1024 loop rounds × 16 multiply-adds × 2), check = out[0] read after timing.
(The kernel uses `a * b + c`, not `fma()`, which Mesa's OpenCL emulates in software.)
**Every GPU kernel launch must finish in well under 2 s**: the amdgpu driver resets the compute
queue after ~2 s ("ring comp_1.x.x timeout"), so full sizes are chosen to stay far below that.

**gpu_bandwidth** (size S MiB) — input buffer of S MiB floats filled with 1.5, output buffer of
the same size; kernel `copy4` with argument n = number of floats and global size n/4.
work = 2·S·2^20 bytes, check = out[n-1] (= 1.5).

**gpu_sgemm** (size N) — A and B as in `matmul_blas` but float32, uploaded before timing;
kernel `sgemm4x4` with arguments (C, count = N², A, B, n = N) and global size (N/4)². work = 2·N³ FLOP,
check = sum of C (computed on the host after timing, in double).

**gpu_upload** (size S MiB) — timed: copy a host array of S MiB float32 (value 2.5) into an
existing device buffer, blocking. R has no float32 type, so R's conversion from double is part
of its upload time. work = S·2^20 bytes, check = n.

**gpu_download** (size S MiB) — device buffer of S MiB floats (value 2.5) prepared before timing.
Timed: read it into a host array, blocking (R: including conversion to double).
work = S·2^20 bytes, check = first element read (= 2.5).

## 5. Build and runtime rules

- **C++**: `g++ -O3 -march=native -ffp-contract=off -std=c++20 -pthread`, no `-ffast-math`
  (so results match R and Python bit-for-bit where the spec says so). BLAS: the system OpenBLAS
  found as described in §1 (also used by JavaScript; R uses whatever BLAS R is linked to). OpenCL: C API, linked
  against `libOpenCL.so.1`. If OpenCL headers are missing at build time, build without the GPU
  section (it then reports GPU as skipped).
- **Python**: CPython 3 with the packages in `requirements.txt` (`run_all.sh` honours `BENCH_PYTHON`); NumPy for vectorized / blas / ram tests; pure
  Python for loop, hashmap and string_ops; `multiprocessing` for parallel; `pyopencl` for GPU.
- **R**: base R (+ `parallel`); the CRAN package `OpenCL` for GPU. Loop tests run inside
  functions so R byte-compiles them.
- **Platform**: Linux (the programs read `/proc` and `/sys`) and Windows 10/11 x64 (registry and Win32
  API; `run_all.ps1`). Windows C++ build: MinGW-w64 g++ (e.g. from Rtools) with the same flags plus
  `-Wa,-muse-unaligned-vector-move` (GCC cannot align the Windows stack to 32 bytes, so AVX spills must
  use unaligned moves), linked to the OpenBLAS DLL (copied next to the program) and to the OpenCL SDK's
  import library; the GPU driver's `OpenCL.dll` is used at run time. R for Windows uses its bundled
  reference BLAS (`Rblas.dll`, one thread) unless it has been replaced; `blas` names that file.
- **JavaScript**: Node.js (e.g. installed with nvm, or the Windows installer), run as `node --expose-gc`; ES module; standard
  library only, except two npm packages (in `JavaScript/package.json`): `webgpu` for the GPU and
  `koffi` to call OpenBLAS. If one is missing (or no GPU adapter is found), skip the affected tests
  with the reason.
