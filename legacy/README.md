# Legacy single-language benchmarks

The first, separate benchmarks for R and Python, written before the common four-language suite
(see the main [README](../README.md)). They still work but are not part of `run_all.sh` or
`analysis.ipynb`, and their results use a different format.

## R

```bash
cd legacy/R
Rscript r_benchmark.R              # everything (about 2-3 minutes)
Rscript r_benchmark.R r25          # R-benchmark-25 only: matrix math + programming (under a minute)
Rscript r_benchmark.R practical    # everyday work: data frames, merge, lm/glm, strings, 8 GB memory test
Rscript r_benchmark.R parallel     # multi-core scaling with mclapply
```

- Each run is saved to `results/r_<date>_<time>.csv`. To tag a run, use
  `RBENCH_LABEL=performance-profile Rscript r_benchmark.R`.
- The header shows which BLAS (matrix library) R is using. Matrix results depend almost entirely on it.
- To compare two runs (above 1 = the second run is faster):

  ```bash
  Rscript -e 'a <- read.csv("results/FIRST.csv"); b <- read.csv("results/SECOND.csv")
  m <- merge(a, b, by = c("section", "test")); m$speedup <- round(m$seconds.x / m$seconds.y, 2)
  print(m[c("test", "seconds.x", "seconds.y", "speedup")], row.names = FALSE)'
  ```

## Python

```bash
cd legacy/Python
python3 python_benchmark.py --save results/py_$(date +%F_%H%M).json   # full run (1-2 minutes)
python3 python_benchmark.py --quick                                     # smaller, quick check
python3 python_benchmark.py --only single,multi                         # chosen sections only
python3 python_benchmark.py --compare results/OLDER.json                # compare with a saved run
python3 python_benchmark.py --help                                      # all options
```

- Sections: `single`, `multi`, `memory`, `disk`, `numpy`.
- The disk test writes a temporary 1 GB file in the current folder and deletes it afterwards.
- The `numpy` section is skipped if NumPy isn't installed.

## Baseline: R on a Dell Inspiron 14 7425 (Ryzen 5 5625U), 2026-10-01, AC power, balanced profile

| File in `legacy/R/results/` | Setup |
|---|---|
| `r_2026-10-01_reference-blas.csv` | before OpenBLAS: R's slow built-in matrix library |
| `r_2026-10-01_openblas-test.csv` | OpenBLAS 0.3.32 loaded for a test run (the same version is now installed) |

| Test | Reference BLAS | OpenBLAS |
|---|---|---|
| R-benchmark-25 overall | 0.554 s | 0.178 s |
| Matrix multiply 2800×2800 | 8.27 s | 0.12 s |
| Cholesky 3000×3000 | 3.16 s | 0.10 s |
| mclapply, 1 → 6 workers | 4.05 → 0.84 s (4.8×) | no gain beyond 6 workers |
