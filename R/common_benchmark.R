#!/usr/bin/env Rscript
# common_benchmark.R -- R implementation of the shared C++/Python/R benchmark.
# The tests, sizes and output format are defined in ../common/SPEC.md and ../common/tests.csv.
#
# Usage
#   Rscript common_benchmark.R                    full run
#   Rscript common_benchmark.R --quick            smaller sizes
#   Rscript common_benchmark.R --verify           tiny sizes, identical in all languages (compare checks)
#   Rscript common_benchmark.R --only cpu_single,mem_copy   chosen categories and/or tests
#   Rscript common_benchmark.R --out DIR          results directory (default: <root>/results)
#
# CPU and RAM tests use base R only; the GPU tests need the CRAN package "OpenCL".

suppressWarnings(suppressMessages(library(parallel)))

# ---------------------------------------------------------------- command line and paths
args <- commandArgs(TRUE)
mode <- "full"; only <- NULL; out_dir <- NULL
i <- 1
while (i <= length(args)) {
  a <- args[i]
  if (a == "--quick") mode <- "quick"
  else if (a == "--verify") mode <- "verify"
  else if (a == "--only" && i < length(args)) { only <- strsplit(args[i + 1], ",")[[1]]; i <- i + 1 }
  else if (startsWith(a, "--only=")) only <- strsplit(sub("^--only=", "", a), ",")[[1]]
  else if (a == "--out" && i < length(args)) { out_dir <- args[i + 1]; i <- i + 1 }
  else if (startsWith(a, "--out=")) out_dir <- sub("^--out=", "", a)
  else if (a %in% c("-h", "--help")) {
    cat("Usage: Rscript common_benchmark.R [--quick | --verify] [--only LIST] [--out DIR]\n"); quit(status = 0)
  } else stop("unknown argument: ", a)
  i <- i + 1
}
file_arg   <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else getwd()
root       <- dirname(script_dir)
# Machine id (SPEC.md): BENCH_MACHINE, else the DMI vendor + product name, else the hostname.
# slug: lowercase, every run of characters outside [a-z0-9] becomes "-", no "-" at either end.
slug <- function(x) gsub("^-+|-+$", "", gsub("[^a-z0-9]+", "-", tolower(x)))
read_dmi <- function(f) {
  v <- tryCatch(readLines(file.path("/sys/class/dmi/id", f), n = 1, warn = FALSE), error = function(e) "")
  if (length(v)) trimws(v) else ""
}
machine <- Sys.getenv("BENCH_MACHINE")
if (!nzchar(machine)) machine <- slug(paste(read_dmi("sys_vendor"), read_dmi("product_name")))
if (!nzchar(machine)) machine <- slug(Sys.info()[["nodename"]])
# Replaces the user's home directory with "~", so no personal paths end up in results.
tilde <- function(x) {
  home <- path.expand("~")
  if (nzchar(home) && home != "/") gsub(home, "~", x, fixed = TRUE) else x
}
if (is.null(out_dir)) out_dir <- file.path(root, "results", machine)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

tests <- read.csv(file.path(root, "common", "tests.csv"), colClasses = "character")
if (!is.null(only)) {
  bad <- setdiff(only, c(tests$category, tests$test))
  if (length(bad)) stop("unknown --only item(s): ", paste(bad, collapse = ", "))
  tests <- tests[tests$category %in% only | tests$test %in% only, ]
}
kernels_path <- file.path(root, "common", "kernels.cl")

# ---------------------------------------------------------------- system information
read1 <- function(path) tryCatch(readLines(path, n = 1, warn = FALSE), error = function(e) NA_character_)
cpu_temp <- function() {   # first hwmon sensor of a known CPU driver, in order of preference
  hwmons <- Sys.glob("/sys/class/hwmon/hwmon*")
  for (want in c("k10temp", "zenpower", "coretemp", "cpu_thermal"))
    for (h in hwmons)
      if (identical(read1(file.path(h, "name")), want))
        return(as.numeric(read1(file.path(h, "temp1_input"))) / 1000)
  NA_real_
}
cpuinfo <- readLines("/proc/cpuinfo")
field <- function(key) trimws(sub("^[^:]*:", "", grep(paste0("^", key, "\\s*:"), cpuinfo, value = TRUE)))
phys_cores <- length(unique(paste(field("physical id"), field("core id"))))
logical_cpus <- detectCores()
meminfo_kb <- function(key) {
  line <- grep(paste0("^", key, ":"), readLines("/proc/meminfo"), value = TRUE)
  as.numeric(gsub("[^0-9]", "", line))
}
on_ac <- any(vapply(Sys.glob("/sys/class/power_supply/*/online"), read1, "") == "1")
blas_path <- sessionInfo()$BLAS
blas_threads <- if (grepl("openblas", blas_path, ignore.case = TRUE)) {
  env <- Sys.getenv(c("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"))
  env <- suppressWarnings(as.integer(env[nzchar(env)]))
  if (length(env) && !is.na(env[1])) env[1] else logical_cpus
} else 1L

start_time <- Sys.time()
stamp  <- format(start_time, "%Y%m%d-%H%M%S")
run_id <- paste0("r_", stamp)
batch  <- Sys.getenv("BENCH_BATCH", stamp)
results_file <- file.path(out_dir, paste0(run_id, ".csv"))
meta_file    <- file.path(out_dir, paste0(run_id, "_meta.csv"))

meta <- list(
  run_id = run_id, batch = batch, language = "R", mode = mode,
  language_version = paste("R", getRversion()), build = R.home(), blas = blas_path, numpy_version = "",
  cpu = field("model name")[1], physical_cores = phys_cores, logical_cpus = logical_cpus,
  ram_gib = sprintf("%.1f", meminfo_kb("MemTotal") / 2^20),
  power = if (on_ac) "AC" else "battery",
  platform_profile = read1("/sys/firmware/acpi/platform_profile"),
  cpu_temp_start_c = cpu_temp(), cpu_temp_end_c = "",
  started = format(start_time, "%Y-%m-%dT%H:%M:%S"), elapsed_s = "", host = machine, machine = machine,
  gpu_platform = "", gpu_device = "", gpu_compute_units = "", gpu_max_clock_mhz = "", gpu_driver = "",
  skipped = "", failed = "")
skipped <- character(); failed <- character()

# ---------------------------------------------------------------- helpers
PHI  <- 0.6180339887498949
PHI2 <- 0.7548776662466927
w  <- function(i) { v <- i * PHI;  v - floor(v) }    # Weyl sequence, SPEC section 2
w2 <- function(i) { v <- i * PHI2; v - floor(v) }
now   <- function() Sys.time()
since <- function(t0) as.numeric(Sys.time()) - as.numeric(t0)
num17 <- function(x) sprintf("%.17g", x)

fmt_rate <- function(rate, unit) {
  for (p in list(c(1e9, "G"), c(1e6, "M"), c(1e3, "k")))
    if (rate >= as.numeric(p[1])) return(sprintf("%8.2f %s%s/s", rate / as.numeric(p[1]), p[2], unit))
  sprintf("%8.2f %s/s", rate, unit)
}

rows <- list()
write_results <- function() {
  if (length(rows)) write.csv(do.call(rbind, rows), results_file, row.names = FALSE)
}
write_meta <- function() {
  meta$skipped <- paste(skipped, collapse = "; ")
  meta$failed  <- paste(failed, collapse = "; ")
  vals <- vapply(meta, function(v) if (length(v) == 0 || is.na(v[1])) "" else tilde(as.character(v[1])), "")
  write.csv(data.frame(key = names(meta), value = vals), meta_file, row.names = FALSE)
}

# Run fn() (which times itself and returns list(sec, check)) for the test's repeats,
# record one row per timed repetition and print one summary line.
measure <- function(t, size, threads, work, fn, reps = t$reps) {
  if (reps > 1 && mode != "verify") invisible(fn())          # warm-up, not recorded
  secs <- numeric(reps); chk <- NA_real_
  for (r in seq_len(reps)) {
    invisible(gc())
    res <- fn()
    secs[r] <- res$sec; chk <- res$check
  }
  rows[[length(rows) + 1]] <<- data.frame(
    run_id = run_id, batch = batch, language = "R", mode = mode, category = t$category,
    test = t$test, style = t$style, threads = threads, size = size, rep = seq_len(reps),
    seconds = sprintf("%.9g", secs), work = num17(work), unit = t$unit,
    rate = sprintf("%.9g", work / secs), check = num17(chk))
  label <- if (t$test %in% c("parallel_mandelbrot", "mem_triad")) sprintf("%s x%d", t$test, threads) else t$test
  cat(sprintf("  %-11s %-22s size=%-12s %8.3f s %s\n", t$category, label, size,
              median(secs), fmt_rate(work / median(secs), t$unit)))
  flush.console()
  write_results()
}

# ---------------------------------------------------------------- loop-style kernels (byte-compiled)
scalar_loop <- function(N) {
  s <- 0
  for (i in 0:(N - 1)) s <- s + sqrt(i)
  s
}

mandel_rows <- function(W, y0, y1) {          # rows y0..y1 (0-based, inclusive)
  total <- 0
  for (y in y0:y1) {
    ci <- -1.25 + (2.5 * y) / W
    for (x in 0:(W - 1)) {
      cr <- -2.0 + (2.5 * x) / W
      zr <- 0; zi <- 0; it <- 0L
      while (it < 100L) {
        t <- zr * zr - zi * zi + cr
        zi <- 2.0 * zr * zi + ci
        zr <- t
        it <- it + 1L
        if (zr * zr + zi * zi > 4.0) break
      }
      total <- total + it
    }
  }
  total
}

fib <- function(k) if (k < 2) k else fib(k - 1) + fib(k - 2)
fib_calls <- function(n) { a <- 0; b <- 1; for (k in 0:n) { t <- a + b; a <- b; b <- t }; 2 * a - 1 }

hash_run <- function(ks) {
  e <- new.env(hash = TRUE)
  N <- length(ks)
  for (i in seq_len(N)) e[[ks[i]]] <- i - 1
  total <- 0
  for (i in seq_len(N)) total <- total + e[[ks[i]]]
  total
}

string_run <- function(N) {
  s <- paste0("item", 0:(N - 1))
  j <- paste(s, collapse = ",")
  p <- strsplit(j, ",", fixed = TRUE)[[1]]
  v <- as.integer(substring(p, 5))
  sum(as.numeric(v))
}

# ---------------------------------------------------------------- tests
run_cpu_ram <- function(t, size) {
  S <- suppressWarnings(as.numeric(size))   # NA for the ";"-separated mem_alloc list
  switch(t$test,
    scalar_loop = measure(t, size, 1, S, function() {
      t0 <- now(); s <- scalar_loop(S); list(sec = since(t0), check = s) }),

    mandelbrot = measure(t, size, 1, S * S, function() {
      t0 <- now(); s <- mandel_rows(S, 0, S - 1); list(sec = since(t0), check = s) }),

    fib_recursive = measure(t, size, 1, fib_calls(S), function() {
      t0 <- now(); s <- fib(S); list(sec = since(t0), check = s) }),

    vector_math = {
      x <- 100 * w(0:(S - 1))
      measure(t, size, 1, S, function() {
        t0 <- now(); s <- sum(sqrt(x) * x + 1); list(sec = since(t0), check = s) })
    },

    sort = {
      x <- w(0:(S - 1))
      measure(t, size, 1, S, function() {
        t0 <- now(); y <- sort(x); sec <- since(t0)
        list(sec = sec, check = y[1] + y[floor(S / 2) + 1] + y[S]) })
    },

    hashmap = {
      ks <- as.character(floor(w(0:(S - 1)) * 2^31))
      measure(t, size, 1, 2 * S, function() {
        t0 <- now(); s <- hash_run(ks); list(sec = since(t0), check = s) })
    },

    string_ops = measure(t, size, 1, S, function() {
      t0 <- now(); s <- string_run(S); list(sec = since(t0), check = s) }),

    parallel_mandelbrot = {
      W <- S; nchunk <- min(96, W)
      bounds <- floor((0:nchunk) * W / nchunk)          # chunk c covers rows bounds[c]..bounds[c+1]-1
      chunks <- lapply(seq_len(nchunk), function(k) c(bounds[k], bounds[k + 1] - 1))
      chunks <- chunks[vapply(chunks, function(b) b[2] >= b[1], TRUE)]
      counts <- sort(unique(c(1, 2, 4, phys_cores, 8, logical_cpus)))
      counts <- counts[counts >= 1 & counts <= logical_cpus]
      for (workers in counts)
        measure(t, size, workers, W * W, function() {
          t0 <- now()
          parts <- mclapply(chunks, function(b) mandel_rows(W, b[1], b[2]), mc.cores = workers)
          sec <- since(t0)
          list(sec = sec, check = sum(unlist(parts))) })
    },

    matmul_blas = {
      N <- S
      A <- matrix(w(0:(N * N - 1)), N, N, byrow = TRUE)
      B <- matrix(w(N * N + 0:(N * N - 1)), N, N, byrow = TRUE)
      measure(t, size, blas_threads, 2 * N^3, function() {
        t0 <- now(); C <- A %*% B; sec <- since(t0); list(sec = sec, check = sum(C)) })
    },

    mem_copy = {
      n <- S * 2^20 / 8
      src <- w(0:(n - 1))
      measure(t, size, 1, 2 * S * 2^20, function() {
        dst <- numeric(n)                                 # allocated and zero-filled, not timed
        t0 <- now(); dst[] <- src; sec <- since(t0)
        list(sec = sec, check = dst[n]) })
    },

    mem_triad = {
      n <- S * 2^20 / 8
      b <- rep(1, n); c <- rep(2, n); q <- 3
      measure(t, size, 1, 3 * S * 2^20, function() {
        t0 <- now(); a <- b + q * c; sec <- since(t0)
        list(sec = sec, check = a[n / 2 + 1]) })
    },

    mem_gather = {
      n <- S * 2^20 / 8; m <- n / 4
      tab <- w(0:(n - 1))
      idx <- as.integer(floor(w2(0:(m - 1)) * n)) + 1L
      measure(t, size, 1, m, function() {
        t0 <- now(); s <- sum(tab[idx]); list(sec = since(t0), check = s) })
    },

    mem_alloc = {
      avail <- meminfo_kb("MemAvailable") * 1024
      for (g in strsplit(size, ";")[[1]]) {
        G <- as.numeric(g); bytes <- G * 2^30; n <- bytes / 8
        if (bytes > 0.4 * avail) {
          skipped <<- c(skipped, sprintf("mem_alloc %s GiB: more than 40%% of MemAvailable (%.1f GiB)", g, avail / 2^30))
          cat(sprintf("  %-11s %-22s size=%-12s skipped (needs more than 40%% of available RAM)\n", t$category, t$test, g))
          next
        }
        measure(t, g, 1, 2 * bytes, function() {
          t0 <- now()
          x <- numeric(n); x[] <- 1; s <- sum(x); rm(x)
          sec <- since(t0)
          invisible(gc())
          list(sec = sec, check = s) }, reps = t$reps)
      }
    },
    stop("no R implementation for test ", t$test))
}

# ---------------------------------------------------------------- GPU (OpenCL package)
# Notes on the CRAN "OpenCL" package (0.2-x), which shape the code below:
# - oclRun(kernel, size, ..., dim) allocates a new output buffer of `size` elements on every call
#   and always passes `size` as the kernel's 2nd argument (the output length).
# - oclRun only enqueues the kernel; reading from the result buffer blocks until the kernel has
#   finished, so each timed launch ends by reading one element (out[1]).
# - Buffer reads/writes are blocking and convert between R doubles and device floats on the host.
gpu <- NULL
gpu_setup <- function() {
  if (Sys.getenv("RUSTICL_ENABLE") == "") Sys.setenv(RUSTICL_ENABLE = "radeonsi,iris")  # expose AMD/Intel GPUs
  if (!requireNamespace("OpenCL", quietly = TRUE))
    return("R package 'OpenCL' is not installed")
  # The first GPU (over all platforms) whose name contains BENCH_GPU (case-insensitive),
  # or simply the first GPU when BENCH_GPU is unset.
  want <- tolower(Sys.getenv("BENCH_GPU"))
  dev <- NULL; plat <- NULL; seen <- character()
  for (p in tryCatch(OpenCL::oclPlatforms(), error = function(e) list())) {
    for (d in tryCatch(OpenCL::oclDevices(p, type = "gpu"), error = function(e) list())) {
      name <- trimws(OpenCL::oclInfo(d)$name)
      seen <- c(seen, name)
      if (is.null(dev) && (!nzchar(want) || grepl(want, tolower(name), fixed = TRUE))) { dev <- d; plat <- p }
    }
  }
  if (is.null(dev)) {
    if (!length(seen)) return("no OpenCL GPU device found (is RUSTICL_ENABLE set for your GPU driver?)")
    return(sprintf("no OpenCL GPU matches BENCH_GPU=%s (available: %s)", Sys.getenv("BENCH_GPU"),
                   paste(seen, collapse = " | ")))
  }
  info  <- OpenCL::oclInfo(dev)
  pinfo <- OpenCL::oclInfo(plat)
  ctx <- OpenCL::oclContext(dev, precision = "single")
  code <- paste(readLines(kernels_path), collapse = "\n")
  kern <- list(
    fma_peak = OpenCL::oclSimpleKernel(ctx, "fma_peak", code, "single"),
    copy4    = OpenCL::oclSimpleKernel(ctx, "copy4", code, "single"),
    sgemm4x4 = OpenCL::oclSimpleKernel(ctx, "sgemm4x4", code, "single"))
  cu <- tryCatch(OpenCL:::.oclDeviceInfoEntry(dev, 0x1002L, 4L), error = function(e) NA)  # CL_DEVICE_MAX_COMPUTE_UNITS
  meta$gpu_platform      <<- paste(pinfo$name, pinfo$version)
  meta$gpu_device        <<- info$name
  meta$gpu_compute_units <<- if (length(cu) && !is.na(cu[1])) as.character(cu[1]) else ""
  meta$gpu_max_clock_mhz <<- as.character(info$max.frequency)
  meta$gpu_driver        <<- as.character(info$driver.ver)
  gpu <<- list(ctx = ctx, kern = kern)
  NULL
}

run_gpu <- function(t, size) {
  S <- as.numeric(size); ctx <- gpu$ctx; k <- gpu$kern
  switch(t$test,
    gpu_fp32_peak = {
      n <- as.integer(S)
      measure(t, size, 0, S * 1024 * 32, function() {
        t0 <- now(); out <- OpenCL::oclRun(k$fma_peak, n); invisible(out[1]); sec <- since(t0)
        list(sec = sec, check = out[1]) })
    },
    gpu_bandwidth = {
      n <- as.integer(S * 2^20 / 4)
      inp <- OpenCL::as.clBuffer(rep(1.5, n), ctx, "single")
      measure(t, size, 0, 2 * S * 2^20, function() {
        t0 <- now(); out <- OpenCL::oclRun(k$copy4, n, inp, dim = n %/% 4L); invisible(out[1]); sec <- since(t0)
        list(sec = sec, check = out[n]) })
    },
    gpu_sgemm = {
      N <- as.integer(S)
      A <- OpenCL::as.clBuffer(w(0:(S * S - 1)), ctx, "single")       # row-major, as in kernels.cl
      B <- OpenCL::as.clBuffer(w(S * S + 0:(S * S - 1)), ctx, "single")
      measure(t, size, 0, 2 * S^3, function() {
        t0 <- now()
        C <- OpenCL::oclRun(k$sgemm4x4, N * N, A, B, N, dim = (N %/% 4L)^2)  # count = N*N, then A, B, n
        invisible(C[1]); sec <- since(t0)
        list(sec = sec, check = sum(C[])) })
    },
    gpu_upload = {
      n <- as.integer(S * 2^20 / 4)
      host <- rep(2.5, n)
      buf <- OpenCL::clBuffer(ctx, n, "single")                          # existing device buffer
      measure(t, size, 0, S * 2^20, function() {
        t0 <- now(); buf[] <- host; sec <- since(t0)                      # blocking, incl. double -> float
        list(sec = sec, check = length(buf)) })
    },
    gpu_download = {
      n <- as.integer(S * 2^20 / 4)
      buf <- OpenCL::as.clBuffer(rep(2.5, n), ctx, "single")
      measure(t, size, 0, S * 2^20, function() {
        t0 <- now(); host <- buf[]; sec <- since(t0)                      # blocking, incl. float -> double
        list(sec = sec, check = host[1]) })
    },
    stop("no R implementation for test ", t$test))
}

# ---------------------------------------------------------------- main
size_col <- switch(mode, full = "full_r", quick = "quick_r", verify = "verify")
cat("Common benchmark - R\n", strrep("=", 72), "\n", sep = "")
gpu_reason <- NULL
if (any(tests$category == "gpu")) gpu_reason <- tryCatch(gpu_setup(), error = function(e) conditionMessage(e))
hdr <- c(Language = meta$language_version, Machine = machine, CPU = meta$cpu,
         Cores = sprintf("%d physical / %d logical", phys_cores, logical_cpus),
         BLAS = sprintf("%s (%d threads)", blas_path, blas_threads),
         GPU = if (!is.null(gpu)) meta$gpu_device else if (any(tests$category == "gpu")) paste("none -", gpu_reason) else "not tested",
         Mode = mode, Output = tilde(results_file))
cat(sprintf("  %-9s %s\n", names(hdr), hdr), sep = "")
cat("\n")

for (r in seq_len(nrow(tests))) {
  t <- as.list(tests[r, ])
  t$reps <- switch(mode, full = as.integer(t$repeats), quick = min(as.integer(t$repeats), 3L), verify = 1L)
  size <- t[[size_col]]
  if (t$category == "gpu" && is.null(gpu)) {
    skipped <- c(skipped, sprintf("%s: %s", t$test, gpu_reason))
    cat(sprintf("  %-11s %-22s skipped (%s)\n", t$category, t$test, gpu_reason))
    next
  }
  ok <- tryCatch({
    if (t$category == "gpu") run_gpu(t, size) else run_cpu_ram(t, size)
    TRUE
  }, error = function(e) {
    cat(sprintf("  %-11s %-22s FAILED: %s\n", t$category, t$test, conditionMessage(e)))
    failed <<- c(failed, sprintf("%s: %s", t$test, conditionMessage(e)))
    FALSE
  })
}

meta$cpu_temp_end_c <- cpu_temp()
meta$elapsed_s <- sprintf("%.1f", since(start_time))
write_results(); write_meta()
cat(sprintf("\nDone in %s s. CPU %.0f C -> %.0f C\nSaved %s\n      %s\n", meta$elapsed_s,
            as.numeric(meta$cpu_temp_start_c), meta$cpu_temp_end_c, tilde(results_file), tilde(meta_file)))
