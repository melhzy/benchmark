# C++ vs Python vs R vs JavaScript: a hardware benchmark

The same set of tests, written four times (C++, Python, R, JavaScript on Node.js), to see how each language
uses a computer's **CPU (one core and all cores), RAM and GPU**, and how close each gets to the hardware's
limits. Every language runs the same algorithms on the same inputs, and a verify mode checks that all four
compute identical answers. Results from different machines live side by side in `results/<machine>/`, and
`analysis.ipynb` charts one machine in detail and compares machines.

Linux only. Licensed under the [MIT License](LICENSE).

## Results

Charts and findings: [`analysis.ipynb`](analysis.ipynb) (GitHub displays it with all charts). Raw data:
[`results/`](results/), one folder per machine.

First machine: Dell Inspiron 14 7425 2-in-1 — AMD Ryzen 5 5625U (6 cores / 12 threads, 15 W), 64 GB DDR4-3200,
Radeon Vega 7 integrated GPU, Ubuntu 26.04. Speed relative to C++ (1× = as fast as C++), geometric mean of
the tests in each category:

| | Overall | CPU, one core | CPU, all cores | RAM | GPU |
|---|---|---|---|---|---|
| C++ | 1× | 1× | 1× | 1× | 1× |
| JavaScript (Node.js 24) | 0.75× | 0.59× | 0.93× | 0.87× | 0.67× |
| Python 3.14 + NumPy | 0.42× | 0.23× | 0.12× | 1.2× | 0.97× |
| R 4.5 | 0.17× | 0.06× | 0.09× | 0.34× | 0.38× |

## What is measured

| Category | Tests |
|---|---|
| CPU, one core | plain loop, Mandelbrot, recursive calls, vector math, sort, hash map, string processing |
| CPU, all cores | parallel Mandelbrot with 1 → all workers; FP64 matrix multiply through OpenBLAS |
| RAM | copy, STREAM triad, random access, allocating 1 → 16 GiB |
| GPU | FP32 compute peak, memory bandwidth, matrix multiply, upload, download |

Exact definitions, sizes and rules are in [`common/SPEC.md`](common/SPEC.md) (sizes in
[`common/tests.csv`](common/tests.csv)). GPU tests run the same OpenCL kernels
([`common/kernels.cl`](common/kernels.cl)) from C++, Python and R; JavaScript runs a line-by-line WebGPU
translation ([`common/kernels.wgsl`](common/kernels.wgsl)).

## Setup on a new machine

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

A language whose tools are missing is skipped (or just its GPU / BLAS test, with the reason recorded), so a
partial setup still produces useful results.

## Running

```bash
./run_all.sh --verify          # about a minute: tiny sizes, then checks all four languages agree
./run_all.sh                   # full run of all four languages (about 7 minutes on a 6-core laptop)
./run_all.sh --langs r         # one language at a time: cpp, python, r, js
./run_all.sh --only gpu        # only some categories / tests: cpu_single, cpu_multi, ram, gpu, or test names
./run_all.sh --help            # all options
```

`run_all.sh` builds the C++ program, waits for the CPU to cool below 60 °C before each language, logs
temperatures, clocks, GPU load and RAM use once per second, and records the machine's hardware. Everything goes
to `results/<machine>/`, where `<machine>` comes from the computer's model (e.g.
`dell-inc-inspiron-14-7425-2-in-1`); `--machine NAME` chooses another name. Results contain no hostname and no
home-directory paths.

Environment variables: `BENCH_PYTHON` (which Python), `BENCH_GPU` (pick an OpenCL GPU by part of its name,
e.g. `BENCH_GPU=nvidia`), `BENCH_GPU_POWER` (JavaScript: `high-performance` or `low-power`), `BENCH_OPENBLAS`
(path to `libopenblas` if it isn't found automatically). Details in `common/SPEC.md`.

For comparable numbers: plug in the charger, keep the same power profile, close heavy programs, and don't
change `common/tests.csv` (the notebook warns when machines ran different test versions).

## Adding a machine

1. Clone the repository on the new machine, set it up as above, and run `./run_all.sh`.
2. Commit the new `results/<machine>/` folder and push it (or open a pull request).

## Analysing

```bash
jupyter lab analysis.ipynb          # or VS Code, with a Python kernel that has requirements.txt installed
```

Run all cells. Sections 1–11 analyse one machine (`MACHINE` in the first cell; default: the machine with the
newest run): scores relative to C++, every test, parallel scaling, BLAS, memory bandwidth, the GPU, share of
the theoretical peak, run-to-run variation, temperature and clock speed during the run, and generated key
findings. Section 12 compares all machines. Theoretical peaks are estimated from each machine's
`system_<batch>.csv`; if a guess is wrong, correct it with `PEAK_OVERRIDES` in the first cell.

## Repository layout

```
run_all.sh                     runs the benchmark (all languages), writes results/<machine>/
analysis.ipynb                 charts and findings
requirements.txt               Python packages
common/
  SPEC.md                      what every test does, output format, rules (the contract)
  tests.csv                    test list, sizes per language and mode
  kernels.cl, kernels.wgsl     GPU kernels: OpenCL (C++, Python, R) and WebGPU (JavaScript)
  verify.py                    checks that all languages computed the same answers
Cpp/                           C++ version (common_benchmark.cpp, Makefile)
Python/                        Python version (common_benchmark.py)
R/                             R version (common_benchmark.R)
JavaScript/                    JavaScript version (common_benchmark.mjs, package.json)
results/<machine>/             <lang>_<time>.csv + _meta.csv, system_<batch>.csv, sensors_<batch>.csv
legacy/                        the earlier single-language R and Python benchmarks
```

## Things worth knowing

- **Python and R are fast only when work goes to compiled libraries** (NumPy, R's built-in functions, BLAS);
  their plain loops run at a few percent of C++ speed. JavaScript's JIT compiler runs plain loops at about half
  of C++ speed.
- **Matrix multiply goes through OpenBLAS in every language** (JavaScript through the `koffi` package), so it
  measures the BLAS library more than the language.
- **NumPy allocates large arrays faster** than the others because it asks Linux for 2 MB huge pages.
- **AMD integrated GPUs**: the `amdgpu` driver resets the compute queue if one GPU job runs longer than about
  2 seconds, so GPU sizes are kept small; Mesa's OpenCL runs `fma()` in slow software, so the kernels use
  `a * b + c`.
