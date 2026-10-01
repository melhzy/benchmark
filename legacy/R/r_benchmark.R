#!/usr/bin/env Rscript
# r_benchmark.R -- measure how fast this computer runs R. Base R only, no packages needed.
#
# Usage
#   Rscript r_benchmark.R                         full run (about 3-4 minutes)
#   Rscript r_benchmark.R r25                     run selected sections (comma-separated)
#   Rscript r_benchmark.R r25,parallel out.csv    also choose where results are saved
#   RBENCH_LABEL=after-openblas Rscript r_benchmark.R    tag the run (stored in the CSV)
#
# Sections
#   r25        R-benchmark-25 (Urbanek/Grosjean): matrix calculation, matrix functions, programmation
#   practical  everyday work: loops, data frames, merge, lm/glm, strings, bootstrap, 8 GB memory test
#   parallel   mclapply scaling from 1 worker up to all logical cores
#
# Results are saved to results/r_<date>_<time>.csv next to this script unless a path is given.
# Matrix tests depend on the BLAS library R uses, which is printed at the top of the run.
all_sections <- c("r25", "practical", "parallel")
args <- commandArgs(TRUE)
sections <- if (length(args) >= 1) strsplit(args[1], ",")[[1]] else all_sections
if (length(bad <- setdiff(sections, all_sections)))
  stop("unknown section(s): ", paste(bad, collapse = ", "), "; choose from ", paste(all_sections, collapse = ","))
file_arg   <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else getwd()
out_csv    <- if (length(args) >= 2) args[2] else
  file.path(script_dir, "results", format(Sys.time(), "r_%Y-%m-%d_%H%M.csv"))
dir.create(dirname(out_csv), showWarnings = FALSE, recursive = TRUE)
label      <- Sys.getenv("RBENCH_LABEL", "default")

results <- data.frame(label = character(), section = character(), test = character(),
                      seconds = numeric(), stringsAsFactors = FALSE)
record <- function(section, test, secs) {
  results[nrow(results) + 1, ] <<- list(label, section, test, secs)
  cat(sprintf("  %-58s %8.3f s\n", test, secs)); flush.console()
}
# median of `runs` timings (= R-benchmark-25's trimmed mean for 3 runs)
bench <- function(section, test, setup = NULL, expr, runs = 3) {
  t <- numeric(runs)
  for (i in seq_len(runs)) {
    env <- new.env()
    if (!is.null(setup)) eval(setup, env)
    invisible(gc())
    t[i] <- system.time(eval(expr, env))[["elapsed"]]
  }
  record(section, test, median(t))
}

read1 <- function(path) tryCatch(readLines(path, n = 1, warn = FALSE), error = function(e) NA)
cpu_temp <- function() {
  for (h in Sys.glob("/sys/class/hwmon/hwmon*"))
    if (read1(file.path(h, "name")) %in% c("k10temp", "zenpower", "coretemp"))
      return(as.numeric(read1(file.path(h, "temp1_input"))) / 1000)
  NA
}
cpuinfo <- readLines("/proc/cpuinfo")
field <- function(key) sub(".*: ", "", grep(paste0("^", key), cpuinfo, value = TRUE))
phys_cores <- length(unique(paste(field("physical id"), field("core id"))))  # detectCores(logical = FALSE) is unreliable on Linux
on_ac <- any(sapply(Sys.glob("/sys/class/power_supply/*/online"), read1) == "1")

set.seed(42)
cat("R benchmark\n", strrep("=", 72), "\n", sep = "")
cat(sep = "", sprintf("  %-9s %s\n", c("Date", "CPU", "Cores", "R", "BLAS", "Power", "Profile", "CPU temp", "Label", "Output"),
    c(format(Sys.time(), "%Y-%m-%d %H:%M"),
      field("model name")[1],
      paste(phys_cores, "physical /", parallel::detectCores(), "logical"),
      as.character(getRversion()), sessionInfo()$BLAS,
      if (on_ac) "AC power" else "battery", read1("/sys/firmware/acpi/platform_profile"),
      sprintf("%.0f C", cpu_temp()), label, out_csv)))

# ---------------------------------------------------------------- R-benchmark-25
if ("r25" %in% sections) {
  cat("\nI. Matrix calculation\n")
  bench("r25-I", "2500x2500 matrix creation, transpose, deform", expr = quote({
    a <- matrix(rnorm(2500 * 2500) / 10, ncol = 2500, nrow = 2500)
    b <- t(a); dim(b) <- c(1250, 5000); a <- t(b) }))
  bench("r25-I", "2500x2500 normal random matrix ^1000",
        setup = quote(a <- abs(matrix(rnorm(2500 * 2500) / 2, ncol = 2500))),
        expr = quote(b <- a^1000))
  bench("r25-I", "Sort 7,000,000 random values",
        setup = quote(a <- rnorm(7e6)), expr = quote(b <- sort(a, method = "quick")))
  bench("r25-I", "2800x2800 cross-product (a'a)",
        setup = quote({a <- rnorm(2800 * 2800); dim(a) <- c(2800, 2800)}),
        expr = quote(b <- crossprod(a)))
  bench("r25-I", "Linear regression, 2000x2000 (normal equations)",
        setup = quote({a <- matrix(rnorm(2000 * 2000), 2000); b <- matrix(as.double(1:2000), ncol = 1)}),
        expr = quote(cc <- solve(crossprod(a), crossprod(a, b))))

  cat("\nII. Matrix functions\n")
  bench("r25-II", "FFT over 2,400,000 random values",
        setup = quote(a <- rnorm(2400000)), expr = quote(b <- fft(a)))
  bench("r25-II", "Eigenvalues of a 600x600 random matrix",
        setup = quote(a <- array(rnorm(600 * 600), dim = c(600, 600))),
        expr = quote(b <- eigen(a, symmetric = FALSE, only.values = TRUE)$values))
  bench("r25-II", "Determinant of a 2500x2500 random matrix",
        setup = quote({a <- rnorm(2500 * 2500); dim(a) <- c(2500, 2500)}),
        expr = quote(b <- det(a)))
  bench("r25-II", "Cholesky decomposition of a 3000x3000 matrix",
        setup = quote(a <- crossprod(matrix(rnorm(3000 * 3000), 3000))),
        expr = quote(b <- chol(a)))
  bench("r25-II", "Inverse of a 1600x1600 random matrix",
        setup = quote(a <- matrix(rnorm(1600 * 1600), 1600)),
        expr = quote(b <- solve(a)))

  cat("\nIII. Programmation\n")
  bench("r25-III", "3,500,000 Fibonacci numbers (vector calc)",
        setup = quote({a <- floor(runif(3500000) * 1000); phi <- 1.6180339887498949}),
        expr = quote(b <- (phi^a - (-phi)^(-a)) / sqrt(5)))
  bench("r25-III", "3000x3000 Hilbert matrix (matrix calc)", expr = quote({
    a <- 3000; b <- rep(1:a, a); dim(b) <- c(a, a); b <- 1 / (t(b) + 0:(a - 1)) }))
  bench("r25-III", "GCD of 400,000 pairs (recursion)",
        setup = quote({
          gcd2 <- function(x, y) { if (sum(y > 1.0E-4) == 0) x else { y[y == 0] <- x[y == 0]; Recall(y, x %% y) } }
          a <- ceiling(runif(400000) * 1000); b <- ceiling(runif(400000) * 1000) }),
        expr = quote(cc <- gcd2(a, b)))
  bench("r25-III", "500x500 Toeplitz matrix (loops)", expr = quote({
    b <- rep(0, 500 * 500); dim(b) <- c(500, 500)
    for (j in 1:500) for (k in 1:500) b[k, j] <- abs(j - k) + 1 }))
  bench("r25-III", "Escoufier's method on a 45x45 matrix (mixed)",
        setup = quote({
          Trace <- function(y) sum(c(y)[1 + 0:(min(dim(y)) - 1) * (dim(y)[1] + 1)])
          x <- abs(rnorm(45 * 45)); dim(x) <- c(45, 45) }),
        expr = quote({
          p <- ncol(x); vt <- 1:p; vr <- NULL; RV <- 1:p; vrt <- NULL
          for (j in 1:p) {
            Rvmax <- 0
            for (k in 1:(p - j + 1)) {
              x2 <- cbind(x, x[, vr], x[, vt[k]])
              R <- cor(x2)
              Ryy <- R[1:p, 1:p]; Rxx <- R[(p + 1):(p + j), (p + 1):(p + j)]
              Rxy <- R[(p + 1):(p + j), 1:p]; Ryx <- t(Rxy)
              rvt <- Trace(Ryx %*% Rxy) / sqrt(Trace(Ryy %*% Ryy) * Trace(Rxx %*% Rxx))
              if (rvt > Rvmax) { Rvmax <- rvt; vrt <- vt[k] }
            }
            vr[j] <- vrt; RV[j] <- Rvmax; vt <- vt[vt != vr[j]]
          } }))
}

# ---------------------------------------------------------------- practical workloads
if ("practical" %in% sections) {
  cat("\nPractical workloads\n")
  bench("practical", "Interpreted for-loop, 50M iterations (in function)",
        setup = quote(f <- function() { s <- 0; for (i in 1:5e7) s <- s + i %% 7; s }),
        expr = quote(f()))
  bench("practical", "Memory: fill+sum+copy 8 GB of doubles (1e9)", runs = 1, expr = quote({
    x <- numeric(1e9); x[] <- 1; s <- sum(x); y <- x * 2; rm(x, y) }))
  bench("practical", "data.frame 10M rows: tapply group means (1000 grp)",
        setup = quote(df <- data.frame(g = sample.int(1000, 1e7, TRUE), v = rnorm(1e7))),
        expr = quote(m <- tapply(df$v, df$g, mean)))
  bench("practical", "data.frame 10M rows: order by 2 keys",
        setup = quote(df <- data.frame(g = sample.int(1000, 1e7, TRUE), v = rnorm(1e7))),
        expr = quote(o <- df[order(df$g, df$v), ]))
  bench("practical", "merge() 1M x 1M rows on integer key",
        setup = quote({
          a <- data.frame(k = sample.int(2e6, 1e6), x = rnorm(1e6))
          b <- data.frame(k = sample.int(2e6, 1e6), y = rnorm(1e6)) }),
        expr = quote(m <- merge(a, b, by = "k")))
  bench("practical", "lm() 1M rows x 20 predictors",
        setup = quote({ X <- matrix(rnorm(2e7), ncol = 20); y <- X %*% rnorm(20) + rnorm(1e6)
                        d <- data.frame(y = y, X) }),
        expr = quote(fit <- lm(y ~ ., data = d)))
  bench("practical", "glm() logistic 500k rows x 10 predictors",
        setup = quote({ X <- matrix(rnorm(5e6), ncol = 10); p <- plogis(X %*% rnorm(10, 0, .3))
                        d <- data.frame(y = rbinom(5e5, 1, p), X) }),
        expr = quote(fit <- glm(y ~ ., family = binomial, data = d)))
  bench("practical", "Strings: paste + regex gsub on 2M strings", expr = quote({
    s <- paste0("id_", sample.int(1e6, 2e6, TRUE), "_", sample(letters, 2e6, TRUE))
    r <- gsub("_([a-z])$", "-\\1", s); n <- sum(grepl("-[aeiou]$", r)) }))
  bench("practical", "Bootstrap: 2000 resamples of median (n=10k), sapply", expr = quote({
    x <- rnorm(1e4); b <- sapply(1:2000, function(i) median(sample(x, replace = TRUE))) }))
  bench("practical", "Distance matrix dist() 5000x50", setup = quote(m <- matrix(rnorm(5000 * 50), 5000)),
        expr = quote(d <- dist(m)))
}

# ---------------------------------------------------------------- parallel scaling
if ("parallel" %in% sections) {
  library(parallel)
  cat(sprintf("\nParallel scaling (mclapply, 48 CPU-bound tasks, %d logical cores)\n", detectCores()))
  task <- function(i) { s <- 0; for (k in 1:3e6) s <- s + sqrt(k); s }
  for (nc in sort(unique(pmin(c(1, 2, 4, phys_cores, 8, detectCores()), detectCores())))) {
    t <- system.time(r <- mclapply(1:48, task, mc.cores = nc))[["elapsed"]]
    record("parallel", sprintf("mclapply %2d cores", nc), t)
  }
}

# ---------------------------------------------------------------- summary
gm <- function(x) exp(mean(log(x)))
r25 <- results[startsWith(results$section, "r25"), ]
if (nrow(r25)) {
  cat("\nR-benchmark-25 geometric means:\n")
  for (s in unique(r25$section)) cat(sprintf("  %-8s %7.3f s\n", s, gm(r25$seconds[r25$section == s])))
  cat(sprintf("  Overall  %7.3f s\n", gm(r25$seconds)))
}
write.csv(results, out_csv, row.names = FALSE)
cat(sprintf("\nCPU temp at end: %.0f C\nSaved %s\n", cpu_temp(), out_csv))
