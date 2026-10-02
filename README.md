# C++ vs Python vs R vs JavaScript: a hardware benchmark

An 18-test suite measuring how C++, Python, R and JavaScript (Node.js) use a computer's **CPU, RAM and GPU**.
It covers explicit loops, built-in operations, parallel workers, BLAS matrix multiplication and GPU kernels.
The aim is to identify where time is spent: in language execution, libraries, memory movement or GPU work.

All implementations follow a [shared specification](common/SPEC.md), with deterministic inputs and explicit
timing boundaries. Performance runs use smaller inputs for some slower implementations; `--verify` uses
identical small inputs and checks results with test-specific numerical tolerances. This compares the
implementations and software configurations recorded here, rather than defining a universal language ranking.

Runs on Linux, macOS and Windows. Licensed under the [MIT License](LICENSE).

Author: Ziyuan Huang, University of Massachusetts Chan Medical School, Microbiology and Microbiome Dynamics AI Hub.

## Results

**Results page: <https://melhzy.github.io/benchmark/>** (interactive charts; source in [`docs/`](docs/)).
The full analysis is in [`analysis.ipynb`](analysis.ipynb) (GitHub displays it with all charts). Raw data:
[`results/`](results/), one folder per machine.

The tables below summarize the newest full run of each language in the committed **1–2 October 2026** dataset,
which holds three full batches per machine. **1× means the same throughput as C++; 0.5× means half the
throughput.** Category scores are geometric means of test ratios; Overall gives each of the four categories
equal weight. These scores do not predict the elapsed time of a complete application.
See [How to interpret the scores](#how-to-interpret-the-scores).

Each machine is scored against its own C++ build, so the four tables below answer "how do these languages
compare *on this computer*". They do not compare the computers with each other — for that, see
[How the machines compare](#how-the-machines-compare).

**Dell Inspiron 14 7425 2-in-1** (Linux) — AMD Ryzen 5 5625U (6 cores / 12 threads, 15 W), 64 GB DDR4-3200,
Radeon Vega 7 integrated GPU, Ubuntu 26.04:

| | Overall | CPU, one core | CPU, all cores | RAM | GPU |
|---|---|---|---|---|---|
| C++ | 1× | 1× | 1× | 1× | 1× |
| JavaScript (Node.js 24) | 0.75× | 0.59× | 0.93× | 0.87× | 0.67× |
| Python 3.14 + NumPy | 0.42× | 0.23× | 0.12× | 1.2× | 0.97× |
| R 4.5 | 0.17× | 0.06× | 0.09× | 0.34× | 0.38× |

**Alienware m16 R1** (Windows) — Intel Core i9-13900HX (24 cores: 8 performance + 16 efficiency / 32 threads),
64 GB DDR5-5200, NVIDIA GeForce RTX 4090 Laptop GPU (16 GB GDDR6), Windows 11 Home 25H2:

| | Overall | CPU, one core | CPU, all cores | RAM | GPU |
|---|---|---|---|---|---|
| C++ | 1× | 1× | 1× | 1× | 1× |
| JavaScript (Node.js 22) | 0.76× | 0.59× | 0.97× | 0.97× | 0.60× |
| Python 3.13 + NumPy | 0.34× | 0.19× | 0.11× | 0.65× | 1.0× |
| R 4.5 | 0.11× | 0.08× | 0.02× | 0.47× | 0.22× |

**Apple MacBook Pro (14-inch, M5 Pro)** (macOS) — Apple M5 Pro (15 cores: 5 Super + 10 Performance, no SMT),
24 GB LPDDR5X-9600 unified memory (307 GB/s), 16-core GPU, macOS 26.5:

| | Overall | CPU, one core | CPU, all cores | RAM | GPU |
|---|---|---|---|---|---|
| C++ | 1× | 1× | 1× | 1× | 1× |
| JavaScript (Node.js 25) | 0.68× | 0.55× | 0.98× | 0.58× | 0.70×\* |
| Python 3.14 + NumPy | 0.34× | 0.15× | 0.13× | 0.72× | 0.92× |
| R 4.5 | 0.08× | 0.06× | 0.02× | 0.20× | 0.19× |

\* The Mac's JavaScript score includes `gpu_fp32_peak`, whose nominal operation count is unreliable under
Metal's relaxed math compilation (see [GPU interpretation](#implementation-details-and-limitations)).
Excluding that test from **every language's GPU category on this machine** changes JavaScript's GPU score
from **0.70× to 0.54×** and Overall from **0.68× to 0.64×**. The measured rows remain available in the analysis.

**Custom workstation (i9-10900X)** (Linux) — Intel Core i9-10900X (10 cores / 20 threads, AVX-512), 128 GB
quad-channel DDR4-2666 (85.3 GB/s), NVIDIA GeForce RTX 3090 (24 GB GDDR6X; one of four installed, the
benchmark uses one), Ubuntu 24.04:

| | Overall | CPU, one core | CPU, all cores | RAM | GPU |
|---|---|---|---|---|---|
| C++ | 1× | 1× | 1× | 1× | 1× |
| JavaScript (Node.js 22) | 0.65× | 0.52× | 0.82× | 0.67× | 0.62×† |
| Python 3.12 + NumPy | 0.31× | 0.18× | 0.12× | 0.47× | 0.98× |
| R 4.6 | 0.15× | 0.056× | 0.092× | 0.31× | 0.32× |

† The same `gpu_fp32_peak` caveat as on the Mac, from a different GPU stack: under Dawn's Vulkan backend
JavaScript reports 52.9 TFLOP/s against this GPU's estimated 41.6 TFLOP/s ceiling, so its nominal operation
count cannot describe the work actually executed. Across the three batches this test alone varied by 31% for
JavaScript (52.9 to 69.4 TFLOP/s) while C++, Python and R each stayed within 1% — that instability is itself a
sign that the shader compiler, not the hardware, sets the result. Excluding the test from **every language's
GPU category on this machine** changes JavaScript's GPU score from **0.62× to 0.50×** and Overall from
**0.65× to 0.61×**.

### How the machines compare

The same C++ implementation on each machine, relative to the Dell. This separates hardware and toolchain from
the language comparison above: the tables above ask "which language is fastest *here*", this one asks "how much
does the computer change the answer".

| C++ throughput vs the Dell | Overall | CPU, one core | CPU, all cores | RAM | GPU |
|---|---|---|---|---|---|
| Dell Inspiron 14 7425 2-in-1 (Linux) | 1× | 1× | 1× | 1× | 1× |
| Alienware m16 R1 (Windows) | 2.2× | 1.2× | 1.5× | 1.3× | 9.3× |
| Apple MacBook Pro 14-inch, M5 Pro (macOS) | 4.1× | 2.7× | 2.4× | 5.1× | 8.6× |
| Custom workstation, i9-10900X (Linux) | 2.1× | 0.98× | 2.0× | 1.1× | 8.8× |

The Overall column is the least useful one in this table. The Alienware and the workstation sit 0.07 apart on
it while differing by 1.2× against 0.98× per core and by 1.3× against 1.1× on RAM; the Mac leads Overall at
4.1× largely on the strength of unified memory (5.1× the Dell's RAM throughput), even though its GPU summary
is the lowest of the three faster machines. Equal weighting of four categories is a reporting convention,
not a claim that applications are distributed that way.

### Main findings

- **Performance depends on the operation, not on a language's reputation.** JavaScript runs `scalar_loop`
  within a few percent of C++ on three of the four machines (1×, 0.98×, 0.99×) and at 1.2× on the Alienware,
  and `mandelbrot` between 0.93× and 1.1× everywhere — yet recursive Fibonacci collapses to 0.13×–0.28×.
  Python's NumPy sort *beats* C++ on three machines (7.4× Dell, 6.3× Alienware, 5.6× workstation) and loses
  on the fourth (0.58× Mac). Any single per-language number averages these apart.
- **The installed BLAS matters more than the language.** R's FP64 matrix multiply reaches 231 GFLOP/s with
  OpenBLAS on the Dell and 568 GFLOP/s with OpenBLAS on the workstation, against 3.9 and 8.4 GFLOP/s with the
  reference BLAS on Windows and macOS — a spread of more than two orders of magnitude attributable to the
  library rather than to R. The workstation shows how incidental that choice can be: installing
  `libopenblas-dev` for the C++ build also repointed Ubuntu's `libblas.so.3` alternative, so R switched BLAS
  as a side effect, with nobody selecting it. These compare installed stacks, not languages or CPUs.
- **Resident GPU work and host transfers are separate measurements.** On the Dell, R reaches 0.96× C++ across
  the three resident-buffer tests but 0.098× across upload and download; on the workstation the same split is
  0.66× against 0.11×. The workstation makes the cause concrete: its discrete GPU has the widest resident
  bandwidth measured here — 840 GB/s, 90% of its 936 GB/s ceiling — while every byte reaches it across PCIe at
  about 7.8 GB/s, a hundredfold difference folded into one combined GPU score. Whether an application keeps
  data resident on the device therefore matters more than that score suggests.
- **A desktop is not just a faster laptop.** The workstation separates by category more sharply than anything
  else here: 0.98× the Dell per core — a 2019 HEDT core at 4.6 GHz against a 2022 mobile Zen 3 — but 2.0×
  across all cores and 8.8× on GPU. Its Overall figure of 2.1× averages a core no quicker than a
  thin-and-light's with a GPU nearly nine times quicker. The Alienware and the Mac trade places the same way:
  the Alienware leads on C++ FP32 peak compute (41.8 vs 6.2 TFLOP/s) and GPU matrix multiply (7.3 vs
  1.7 TFLOP/s), the Mac leads the measured host–GPU transfers, and their five-test GPU summaries end up
  close regardless (9.3× and 8.6×).
- **How much a difference means depends on which machine produced it.** Across three full batches each, the
  median coefficient of variation for repeated language/test pairs is 0.8% on the workstation, 1.2% on the
  Dell and 2.8% on the Mac, but 16.2% on the Alienware. A 10% gap is a clear effect on the desktop and is
  indistinguishable from noise on the gaming laptop. Fixed cooling is the most plausible reason the desktop
  is the steadiest, and run-to-run ranges are reported per test on the results page for this reason.

### How to interpret the scores

1. Within each selected run, take the **median rate for each test, size and worker count**. Most tests have
   five measured repetitions after a warm-up; parallel Mandelbrot and allocation have one per configuration.
2. Use the **highest of those medians** as the test's headline rate. This selects the best measured worker
   count or allocation size, not a fixed configuration shared by all languages. C++'s RAM score can include
   its multithreaded triad; the other implementations use one thread for that test.
3. Divide by C++'s selected rate on the same machine. Average ratios geometrically within categories, then
   average the category scores geometrically. Each category contributes 25% when all four are present.

Missing tests are omitted, so incomplete runs can have different score coverage. Smaller inputs, cache
behavior, startup costs and different allocation sizes can affect rates even after normalizing by work.
Between-machine ratios also include differences in OS, compiler, runtime, BLAS and GPU API. The recorded
test hashes match for these four machines, but that does not make this a controlled OS or hardware experiment.
Run ranges describe observed variation; they are not confidence intervals or evidence that small gaps are
significant, and how wide they are differs by machine (see the last of the [main findings](#main-findings)).

## What is measured

| Category | Tests |
|---|---|
| CPU, one core | plain loop, Mandelbrot, recursive calls, vector math, sort, hash map, string processing |
| CPU, all cores | parallel Mandelbrot with 1 → all workers; FP64 matrix multiply through the installed BLAS |
| RAM | copy, STREAM triad, random access, allocating 1 → 16 GiB |
| GPU | FP32 compute peak, memory bandwidth, matrix multiply, upload, download |

Exact definitions, sizes and rules are in [`common/SPEC.md`](common/SPEC.md) (sizes in
[`common/tests.csv`](common/tests.csv)). GPU tests run the same OpenCL kernels
([`common/kernels.cl`](common/kernels.cl)) from C++, Python and R; JavaScript runs a line-by-line WebGPU
translation ([`common/kernels.wgsl`](common/kernels.wgsl)).

## Setup on a new machine

### Linux

Ubuntu/Debian package names are shown; other distributions have equivalents (`openblas`, `opencl-headers`,
`ocl-icd`, Mesa's OpenCL "rusticl").

```bash
# C++ and the BLAS library (also used by JavaScript)
sudo apt install g++ make libopenblas0-pthread
# GPU through OpenCL (AMD and Intel GPUs via Mesa; NVIDIA's driver ships its own OpenCL)
sudo apt install mesa-opencl-icd ocl-icd-opencl-dev opencl-headers clinfo
# Python 3 packages (benchmark + notebook), into the Python you will use
python3 -m pip install -r requirements.txt
# R, plus its OpenCL package
sudo apt install r-base
Rscript -e 'install.packages("OpenCL", repos = "https://cloud.r-project.org")'
# JavaScript: Node.js 20 or newer (for example through nvm), then the two npm packages
cd JavaScript && npm install && cd ..

./run_all.sh --check      # shows what is installed and what is missing, per language
```

### macOS (Apple Silicon or Intel)

Install the Xcode Command Line Tools (`xcode-select --install`), [Homebrew](https://brew.sh) and
[R for macOS](https://cran.r-project.org/bin/macosx/). Then, in the repository folder:

```bash
./setup_macos.sh
./run_all.sh --check
```

`setup_macos.sh` installs OpenBLAS, clinfo, Python and (if missing or older than 20) Node.js with Homebrew, creates
a virtual environment `.venv` (ignored by git; `run_all.sh` uses it) with the Python packages, runs `npm install`,
and installs R's OpenCL package. The GPU tests use the OpenCL framework that macOS still ships (C++, Python, R)
and Metal through WebGPU (JavaScript).

### Windows 10 / 11 (x64)

Install with their normal installers: [R](https://cran.r-project.org/bin/windows/base/),
[Rtools](https://cran.r-project.org/bin/windows/Rtools/) (it provides g++ and make, for the C++ program and for
R's OpenCL package), [Python 3](https://www.python.org/downloads/windows/), [Node.js](https://nodejs.org/) 20 or
newer, and a GPU driver (NVIDIA, AMD and Intel drivers include OpenCL). Then, in the repository folder:

```bat
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
.\run_all.cmd --check
```

`setup_windows.ps1` downloads OpenBLAS and the Khronos OpenCL SDK into `deps\` (ignored by git), installs the
Python packages (`requirements.txt`) and the npm packages, and builds R's OpenCL package from source (CRAN has no
Windows binary of it). No administrator rights are needed.

A language whose tools are missing is skipped (or just its GPU / BLAS test, with the reason recorded), so a
partial setup still produces useful results.

## Running

```bash
./run_all.sh --verify          # tiny sizes; checks agreement under each test's tolerance
./run_all.sh                   # full run of all four languages (about 7 minutes on a 6-core laptop)
./run_all.sh --langs r         # one language at a time: cpp, python, r, js
./run_all.sh --only gpu        # only some categories / tests: cpu_single, cpu_multi, ram, gpu, or test names
./run_all.sh --help            # all options
```

On macOS the commands are the same as on Linux. On Windows use `.\run_all.cmd` with the same options (it runs
`run_all.ps1` in Windows PowerShell).

`run_all.sh` builds the C++ program, waits up to five minutes for an available CPU sensor to fall below
60 °C before each language, logs
temperatures, clocks, GPU load and RAM use once per second, and records the machine's hardware. Everything goes
to `results/<machine>/`, where `<machine>` comes from the computer's model (e.g.
`dell-inc-inspiron-14-7425-2-in-1`, `alienware-m16-r1`, `apple-inc-macbook-pro-14-inch-m5-pro`); `--machine NAME`
chooses another name. The `workstation-i9-10900x-rtx-3090` folder was named that way for a concrete reason: a
self-built machine usually leaves its DMI strings at the board defaults, which would otherwise derive the folder
`system-manufacturer-system-product-name`. Such a machine can also set `vendor` and `product` in its
`hardware.csv` so the page and the notebook show a readable name instead of the placeholders.
Results contain no hostname, serial number or home-directory paths. On macOS the
temperature is the hottest SoC die sensor, read without administrator rights by `Cpp/hw_probe`, which also
logs the GPU load and the RAM in use; the runner does not collect privileged CPU or GPU clock counters, so those
columns of the sensor log are empty. `run_all.ps1` does the same on Windows, except that its unprivileged
sensor log has no CPU temperature: it pauses 30 s before each language instead
(`--pause SECONDS`), and its sensor log has CPU clocks, RAM and (NVIDIA) GPU data but no CPU temperature.

Environment variables: `BENCH_PYTHON` (which Python; default `.venv/bin/python` if it exists, else `python3`), `BENCH_GPU` (pick an OpenCL GPU by part of its name,
e.g. `BENCH_GPU=nvidia`), `BENCH_GPU_POWER` (JavaScript: `high-performance` or `low-power`), `BENCH_OPENBLAS`
(path to `libopenblas` if it isn't found automatically); on Windows also `BENCH_RSCRIPT` and `BENCH_OPENCL_SDK`.
Details in `common/SPEC.md`.

For comparable numbers: plug in the charger, keep the same power profile, close heavy programs, and don't
change `common/tests.csv` (the notebook warns when machines ran different test versions). On a Mac whose
Desktop and Documents folders are in iCloud Drive, a repository there is uploaded as it changes: after
`setup_macos.sh` (tens of thousands of files in `.venv` and `node_modules`) wait until iCloud has finished
syncing (Activity Monitor: `fileproviderd` and `bird` idle) before running, or keep the repository outside
iCloud.

## Adding a machine

1. Clone the repository on the new machine, set it up as above, and run `./run_all.sh` (Windows:
   `.\run_all.cmd`) a few times, and once with `--verify`, so the results page can report numerical agreement.
2. The runner records the hardware itself, including what the operating system does not report:
   `Cpp/hw_probe` measures each core type's clock (Windows and macOS have no boost-clock value), reads an NVIDIA
   GPU's memory bus from the CUDA driver, and on a Mac the GPU's core count and highest clock. If something
   still comes out wrong or is missing (e.g. a discrete AMD GPU's memory bandwidth, or a Mac's memory
   bandwidth, which macOS does not report), add a hand-written `results/<machine>/hardware.csv`
   (`key,value,source`) with the corrected keys of `system_<batch>.csv`.
3. Commit the new `results/<machine>/` folder and push it (or open a pull request).
4. Rebuild the results page (below).

## Analysing

```bash
jupyter lab analysis.ipynb          # or VS Code, with a Python kernel that has requirements.txt installed
```

Run all cells. Sections 1–11 analyse one machine (`MACHINE` in the first cell; default: the machine with the
newest run): scores relative to C++, every test, parallel scaling, BLAS, memory bandwidth, the GPU, share of
the theoretical peak, within-run variation, temperature and clock speed during the run, and generated key
findings, including GPU kernel/transfer summaries and a sensitivity analysis for flagged GPU compute results.
Section 12 compares all machines; section 13 states the conclusions and the experiments needed to test likely
explanations. Theoretical peaks are estimated from each machine's
`system_<batch>.csv`; if a guess is wrong, correct it with `PEAK_OVERRIDES` in the first cell.

## Results page

`docs/index.html` is the page GitHub Pages serves at <https://melhzy.github.io/benchmark/>. It compares the
machines, and a switcher shows each machine in detail. It is rebuilt from `results/` with:

```bash
python3 docs/build_page.py                     # every machine with a full run
python3 docs/build_page.py --machine NAME      # only some machines (repeat the option)
python3 docs/build_page.py --list              # machines that have full runs
```

Then commit `docs/index.html`. The build computes chart values, theoretical peaks, sensitivity summaries and
machine-specific findings from the results. A verification badge reports agreement under the checks in
`common/verify.py`; it covers check values on small inputs, rather than every output element or executed
instruction. The machine with the oldest full run is the comparison reference. Edit explanatory text and
layout in `docs/page_template.html`, and generated analysis and wording in `docs/build_page.py`, then rebuild.

## Repository layout

```
run_all.sh                     runs the benchmark (all languages), writes results/<machine>/
run_all.ps1, run_all.cmd       the same on Windows (start it with run_all.cmd)
setup_macos.sh                 installs the macOS dependencies (Homebrew packages, .venv, npm, R's OpenCL)
setup_windows.ps1              installs the Windows dependencies (deps\: OpenBLAS, OpenCL SDK; packages)
analysis.ipynb                 charts and findings
requirements.txt               Python packages
common/
  SPEC.md                      what every test does, output format, rules (the contract)
  tests.csv                    test list, sizes per language and mode
  kernels.cl, kernels.wgsl     GPU kernels: OpenCL (C++, Python, R) and WebGPU (JavaScript)
  verify.py                    checks that all languages computed the same answers
  windows_tools.ps1            finds the tools on Windows (used by run_all.ps1 and setup_windows.ps1)
Cpp/                           C++ version (common_benchmark.cpp, Makefile); hw_probe.cpp: hardware facts for the runners;
                               macos_sys.h: macOS system facts (sysctl, IOKit) for both
Python/                        Python version (common_benchmark.py)
R/                             R version (common_benchmark.R)
JavaScript/                    JavaScript version (common_benchmark.mjs, package.json)
results/<machine>/             <lang>_<time>.csv + _meta.csv, system_<batch>.csv, sensors_<batch>.csv
                               (+ optional hand-written hardware.csv: corrections)
docs/                          results page: index.html (GitHub Pages), build_page.py, page_template.html
legacy/                        the earlier single-language R and Python benchmarks
```

## Implementation details and limitations

- **The implementation is part of the result.** The loop, library and vectorized groups contain different
  tasks. Their score difference is not the speedup from rewriting one program. C++ is a reference
  implementation, not an optimal implementation of every operation; built-in sorting and hash maps use
  language-specific libraries and data structures.
- **Matrix multiply goes through a BLAS library in every language** (OpenBLAS on Linux; JavaScript through the
  `koffi` package), so it measures the BLAS library more than the language. On macOS, C++ and JavaScript use
  Homebrew's OpenBLAS, the recorded NumPy build uses Apple's Accelerate, and the recorded R installation
  uses its single-threaded reference BLAS (`libRblas.0.dylib`). The Windows R runs use `Rblas.dll`.
  The metadata identifies the library; these measurements do not identify which CPU instructions it used.
- **Allocation includes first use of memory.** `mem_alloc` times allocation, filling, summation and release.
  NumPy's Linux huge-page support is a plausible contributor to its Dell advantage, but the stored runs do
  not isolate it. A controlled rerun with huge-page advice disabled would test that explanation.
  [NumPy documents the setting](https://numpy.org/doc/stable/reference/global_state.html#madvise-hugepage-on-linux).
- **Peak percentages are model-dependent.** CPU SIMD width, core types, boost clocks and memory bus width
  are estimated from recorded hardware and overrides. Values above 100% warrant checking both the model
  and the operation count. The charts do not measure power draw, energy efficiency or battery life.
- **WebGPU on older Linux**: the `webgpu` package's Linux build needs a recent C++ runtime (GLIBCXX 3.4.32, e.g.
  Ubuntu 24.04); on Ubuntu 22.04 JavaScript's GPU tests are skipped with that reason. Node.js must be 20 or
  newer; `run_all.sh` prefers an nvm-installed Node when the one on `PATH` is older.
- **Discrete GPUs idle at a low clock**: the runner's programs run the GPU for 2 s before the first GPU test, so
  a laptop's NVIDIA GPU is at its working clock when timing starts.
- **Windows C++ build**: MinGW-w64 g++ (Rtools or MSYS2) with `-Wa,-muse-unaligned-vector-move`, because GCC
  cannot align the Windows stack for AVX registers; R's parallel test uses a socket cluster (Windows cannot
  fork), started before timing, like Python's process pool.
- **macOS**: the C++ program is built with Apple clang (`g++` on a Mac is clang too). Python's process pool starts
  its workers with *spawn* (not timed), R's `mclapply` forks as on Linux. Apple's OpenCL is version 1.2 and
  reports a fixed 1000 MHz clock, so the GPU's real clock comes from its DVFS table (`hw_probe`). Apple Silicon
  has two core types (e.g. 5 "Super" and 10 "Performance" cores in the recorded M5 Pro) and no SMT.
- **GPU interpretation:** JavaScript's `gpu_fp32_peak` exceeds the estimated hardware peak on two machines:
  12.1 against 6.6 TFLOP/s on the Mac, and 52.9 against 41.6 TFLOP/s on the i9-10900X workstation. Relaxed
  math compilation is the explanation in both cases — Metal on the Mac, Dawn's Vulkan backend on the
  workstation — so it is a property of shader compilation rather than of one platform: reassociation can
  reduce the arithmetic while the reported FLOP count stays fixed. The runner flags any machine whose
  measured rate exceeds 1.05× its estimated ceiling, and the analysis then excludes that test from every
  language for a sensitivity check. Treat this as a comparability caveat, not a hardware speedup. Kernel tests exclude initial input upload; transfers are measured separately. Their
  geometric mean is not the elapsed time of an upload–compute–download pipeline, and CPU FP64 BLAS is not
  directly comparable to the custom FP32 GPU matrix kernel.
- **AMD integrated GPUs**: the `amdgpu` driver resets the compute queue if one GPU job runs longer than about
  2 seconds, so GPU sizes are kept small; Mesa's OpenCL runs `fma()` in slow software, so the kernels use
  `a * b + c`.
