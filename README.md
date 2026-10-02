# C++ vs Python vs R vs JavaScript: a hardware benchmark

The same set of tests, written four times (C++, Python, R, JavaScript on Node.js), to see how each language
uses a computer's **CPU (one core and all cores), RAM and GPU**, and how close each gets to the hardware's
limits. Every language runs the same algorithms on the same inputs, and a verify mode checks that all four
compute identical answers. Results from different machines live side by side in `results/<machine>/`, and
`analysis.ipynb` charts one machine in detail and compares machines.

Runs on Linux and Windows. Licensed under the [MIT License](LICENSE).

Author: Ziyuan Huang, University of Massachusetts Chan Medical School, Microbiology and Microbiome Dynamics AI Hub.

## Results

**Results page: <https://melhzy.github.io/benchmark/>** (interactive charts; source in [`docs/`](docs/)).
The full analysis is in [`analysis.ipynb`](analysis.ipynb) (GitHub displays it with all charts). Raw data:
[`results/`](results/), one folder per machine.

Speed relative to C++ on the same machine (1× = as fast as C++), geometric mean of the tests in each category.

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

Machine against machine, the Alienware runs the suite 2.2× as fast as the Dell in C++ and JavaScript, 1.8× in
Python and 1.4× in R (its GPU 9.3×). R's all-cores score on Windows is low because R for Windows ships a
single-threaded reference BLAS for matrix multiply (R on Ubuntu uses OpenBLAS); see the results page.

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
./run_all.sh --verify          # about a minute: tiny sizes, then checks all four languages agree
./run_all.sh                   # full run of all four languages (about 7 minutes on a 6-core laptop)
./run_all.sh --langs r         # one language at a time: cpp, python, r, js
./run_all.sh --only gpu        # only some categories / tests: cpu_single, cpu_multi, ram, gpu, or test names
./run_all.sh --help            # all options
```

On Windows use `.\run_all.cmd` with the same options (it runs `run_all.ps1` in Windows PowerShell).

`run_all.sh` builds the C++ program, waits for the CPU to cool below 60 °C before each language, logs
temperatures, clocks, GPU load and RAM use once per second, and records the machine's hardware. Everything goes
to `results/<machine>/`, where `<machine>` comes from the computer's model (e.g.
`dell-inc-inspiron-14-7425-2-in-1`, `alienware-m16-r1`); `--machine NAME` chooses another name. Results contain
no hostname and no home-directory paths. `run_all.ps1` does the same on Windows, except that Windows lets no
program read the CPU temperature without administrator rights: it pauses 30 s before each language instead
(`--pause SECONDS`), and its sensor log has CPU clocks, RAM and (NVIDIA) GPU data but no CPU temperature.

Environment variables: `BENCH_PYTHON` (which Python), `BENCH_GPU` (pick an OpenCL GPU by part of its name,
e.g. `BENCH_GPU=nvidia`), `BENCH_GPU_POWER` (JavaScript: `high-performance` or `low-power`), `BENCH_OPENBLAS`
(path to `libopenblas` if it isn't found automatically); on Windows also `BENCH_RSCRIPT` and `BENCH_OPENCL_SDK`.
Details in `common/SPEC.md`.

For comparable numbers: plug in the charger, keep the same power profile, close heavy programs, and don't
change `common/tests.csv` (the notebook warns when machines ran different test versions).

## Adding a machine

1. Clone the repository on the new machine, set it up as above, and run `./run_all.sh` (Windows:
   `.\run_all.cmd`) a few times, and once with `--verify`, so the results page can show that all languages agree.
2. The runner records the hardware itself, including what the operating system does not report:
   `Cpp/hw_probe` measures each core type's clock (Windows has no boost-clock value) and reads an NVIDIA
   GPU's memory bus from the CUDA driver. If something still comes out wrong (e.g. a discrete AMD GPU's
   memory bandwidth), add a hand-written `results/<machine>/hardware.csv` (`key,value,source`) with the
   corrected keys of `system_<batch>.csv`.
3. Commit the new `results/<machine>/` folder and push it (or open a pull request).
4. Rebuild the results page (below).

## Analysing

```bash
jupyter lab analysis.ipynb          # or VS Code, with a Python kernel that has requirements.txt installed
```

Run all cells. Sections 1–11 analyse one machine (`MACHINE` in the first cell; default: the machine with the
newest run): scores relative to C++, every test, parallel scaling, BLAS, memory bandwidth, the GPU, share of
the theoretical peak, run-to-run variation, temperature and clock speed during the run, and generated key
findings. Section 12 compares all machines. Theoretical peaks are estimated from each machine's
`system_<batch>.csv`; if a guess is wrong, correct it with `PEAK_OVERRIDES` in the first cell.

## Results page

`docs/index.html` is the page GitHub Pages serves at <https://melhzy.github.io/benchmark/>. It compares the
machines, and a switcher shows each machine in detail. It is rebuilt from `results/` with:

```bash
python3 docs/build_page.py                     # every machine with a full run
python3 docs/build_page.py --machine NAME      # only some machines (repeat the option)
python3 docs/build_page.py --list              # machines that have full runs
```

Then commit `docs/index.html`. All text, chart ranges, theoretical peaks and findings are computed from the
results; a machine's "identical answers" badge appears when its folder contains a passing `--verify` run. The
machine with the oldest full run is the reference of the comparison. Layout and wording live in
`docs/page_template.html`.

## Repository layout

```
run_all.sh                     runs the benchmark (all languages), writes results/<machine>/
run_all.ps1, run_all.cmd       the same on Windows (start it with run_all.cmd)
setup_windows.ps1              installs the Windows dependencies (deps\: OpenBLAS, OpenCL SDK; packages)
analysis.ipynb                 charts and findings
requirements.txt               Python packages
common/
  SPEC.md                      what every test does, output format, rules (the contract)
  tests.csv                    test list, sizes per language and mode
  kernels.cl, kernels.wgsl     GPU kernels: OpenCL (C++, Python, R) and WebGPU (JavaScript)
  verify.py                    checks that all languages computed the same answers
  windows_tools.ps1            finds the tools on Windows (used by run_all.ps1 and setup_windows.ps1)
Cpp/                           C++ version (common_benchmark.cpp, Makefile); hw_probe.cpp: hardware facts for the runners
Python/                        Python version (common_benchmark.py)
R/                             R version (common_benchmark.R)
JavaScript/                    JavaScript version (common_benchmark.mjs, package.json)
results/<machine>/             <lang>_<time>.csv + _meta.csv, system_<batch>.csv, sensors_<batch>.csv
                               (+ optional hand-written hardware.csv: corrections)
docs/                          results page: index.html (GitHub Pages), build_page.py, page_template.html
legacy/                        the earlier single-language R and Python benchmarks
```

## Things worth knowing

- **Python and R are fast only when work goes to compiled libraries** (NumPy, R's built-in functions, BLAS);
  their plain loops run at a few percent of C++ speed. JavaScript's JIT compiler runs plain loops at about half
  of C++ speed.
- **Matrix multiply goes through OpenBLAS in every language** (JavaScript through the `koffi` package), so it
  measures the BLAS library more than the language.
- **NumPy allocates large arrays faster on Linux** than the others because it asks Linux for 2 MB huge pages;
  Windows gives every language normal 4 KB pages.
- **R for Windows ships a single-threaded reference BLAS** (`Rblas.dll`), so R's matrix multiply on Windows is
  far slower than on Linux, where R uses the system OpenBLAS. Replacing `Rblas.dll` with OpenBLAS (needs
  administrator rights) would change that; the benchmark measures R as installed.
- **WebGPU on older Linux**: the `webgpu` package's Linux build needs a recent C++ runtime (GLIBCXX 3.4.32, e.g.
  Ubuntu 24.04); on Ubuntu 22.04 JavaScript's GPU tests are skipped with that reason. Node.js must be 20 or
  newer; `run_all.sh` prefers an nvm-installed Node when the one on `PATH` is older.
- **Discrete GPUs idle at a low clock**: the runner's programs run the GPU for 2 s before the first GPU test, so
  a laptop's NVIDIA GPU is at its working clock when timing starts.
- **Windows C++ build**: MinGW-w64 g++ (Rtools or MSYS2) with `-Wa,-muse-unaligned-vector-move`, because GCC
  cannot align the Windows stack for AVX registers; R's parallel test uses a socket cluster (Windows cannot
  fork), started before timing, like Python's process pool.
- **AMD integrated GPUs**: the `amdgpu` driver resets the compute queue if one GPU job runs longer than about
  2 seconds, so GPU sizes are kept small; Mesa's OpenCL runs `fma()` in slow software, so the kernels use
  `a * b + c`.
