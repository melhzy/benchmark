#!/usr/bin/env python3
"""
build_page.py -- build the results page (docs/index.html, served by GitHub Pages) from results/.

Usage
  python3 docs/build_page.py                    every machine with a full run (newest shown first)
  python3 docs/build_page.py --machine NAME     only these machines (repeat the option for several)
  python3 docs/build_page.py --list             list machines that have full runs
  python3 docs/build_page.py --out FILE         write somewhere else (default: docs/index.html)
  python3 docs/build_page.py --fragment         page body only (no <html>/<head> wrapper)

The page has a section comparing the machines and, below it, a machine switcher that shows one machine at a
time: the newest full-mode run of each language, the spread across all full runs, the sensor log of the newest
full run, and - if the machine's folder holds a verify run - whether all languages computed identical answers.
Text, chart ranges, theoretical peaks and findings are all computed from the results, including the hardware facts
the runners record (measured CPU clocks, GPU memory bus from the CUDA driver, the highest GPU clock logged).
Values can be overridden per machine in an optional, hand-written results/<machine>/hardware.csv (key,value,source). Template: docs/page_template.html. Needs numpy and pandas.
"""

import argparse
import csv
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "common"))
import verify  # noqa: E402  (cross-language agreement check, common/verify.py)

LANGS = ["C++", "Python", "R", "JavaScript"]
CSS_KEY = {"C++": "cpp", "Python": "py", "R": "r", "JavaScript": "js"}
CATS = ["cpu_single", "cpu_multi", "ram", "gpu"]
CAT_TITLES = {"cpu_single": "CPU, one core", "cpu_multi": "CPU, all cores", "ram": "RAM", "gpu": "GPU"}
DISPLAY = {"FLOP": (1e9, "GFLOP/s"), "B": (1e9, "GB/s"), "access": (1e6, "M accesses/s"), "px": (1e6, "M px/s"),
           "iter": (1e6, "M iter/s"), "call": (1e6, "M calls/s"), "elem": (1e6, "M elem/s"),
           "item": (1e6, "M items/s"), "op": (1e6, "M ops/s")}
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
NOT_RESULTS = ("sensors_", "system_", "hardware")


# ---------------------------------------------------------------------------- loading
def read_kv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r[0]: r[1] for r in csv.reader(f) if len(r) >= 2 and r[0] != "key"}


def is_result(f):
    return f.suffix == ".csv" and not f.name.startswith(NOT_RESULTS) and not f.name.endswith("_meta.csv")


def machines_with_full_runs(results):
    """{machine folder: (oldest full batch, newest full batch)}"""
    found = {}
    for folder in sorted(p for p in results.iterdir() if p.is_dir()):
        batches = [read_kv(m).get("batch", "") for m in folder.glob("*_meta.csv") if read_kv(m).get("mode") == "full"]
        if batches:
            found[folder.name] = (min(batches), max(batches))
    return found


def load(folder):
    frames, metas = [], {}
    for f in sorted(p for p in folder.glob("*.csv") if is_result(p)):
        d = pd.read_csv(f, dtype={"batch": str})
        if d.empty:
            continue
        frames.append(d)
        meta = f.with_name(f.stem + "_meta.csv")
        metas[d.run_id.iloc[0]] = read_kv(meta) if meta.exists() else {}
    rows = pd.concat(frames, ignore_index=True)
    systems = sorted((read_kv(f) for f in folder.glob("system_*.csv")), key=lambda d: d.get("batch", ""))
    hardware = read_kv(folder / "hardware.csv") if (folder / "hardware.csv").exists() else {}
    return rows, metas, systems, hardware


# ---------------------------------------------------------------------------- helpers
def gmean(values):
    v = [x for x in values if x is not None and np.isfinite(x) and x > 0]
    return float(np.exp(np.mean(np.log(v)))) if v else None


def fx(v):
    """Speed ratio as text: 0.75x, 7.4x, 0.0017x (two significant digits)."""
    if v is None:
        return "-"
    return f"{v:.0f}×" if v >= 10 else f"{float(f'{v:.2g}'):g}×"


def num(v):
    return f"{v:,.0f}" if abs(v) >= 100 else f"{float(f'{v:.3g}'):g}"


def word(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def esc(s):
    return html.escape(str(s), quote=False)


def mono(s):
    return f'<span class="mono">{esc(s)}</span>'


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def plural(n, one, many=None):
    return one if n == 1 else (many or one + "s")


# ---------------------------------------------------------------------------- theoretical peaks
DISCRETE_GPU = r"nvidia|geforce|rtx|gtx|quadro|radeon rx|radeon pro|arc a\d"


def machine_peaks(system, metas):
    """Same estimates as analysis.ipynb (section 2): CPU FP64, RAM bandwidth, GPU FP32 and GPU memory.

    `system` is the system snapshot with results/<machine>/hardware.csv applied on top."""
    first = lambda key: next((m[key] for m in metas if m.get(key) not in (None, "")), "")
    cpu = system.get("cpu") or first("cpu")
    flags = set(system.get("cpu_flags", "").split())
    cores = fnum(system.get("physical_cores"), fnum(first("physical_cores"), 1))
    p_cores, e_cores = fnum(system.get("performance_cores")), fnum(system.get("efficiency_cores"), 0)
    ghz, e_ghz = fnum(system.get("cpu_max_mhz")) / 1000, fnum(system.get("cpu_e_max_mhz")) / 1000
    if "avx512f" in flags and "intel" in cpu.lower():
        fpc = 32
    elif {"avx2", "fma"} <= flags:
        fpc = 16
    elif "avx" in flags or "asimd" in flags:
        fpc = 8
    else:
        fpc = 4
    if np.isfinite(p_cores) and e_cores > 0 and np.isfinite(e_ghz):
        # Hybrid Intel CPU: efficiency cores have half the FMA width of performance cores.
        cpu_peak = p_cores * fpc * ghz + e_cores * fpc / 2 * e_ghz
    else:
        cpu_peak = cores * fpc * ghz
    ram_type, mts = system.get("ram_type", ""), fnum(system.get("ram_speed_mts"))
    modules = int(fnum(system.get("ram_modules"), 0))
    channels = 2 if ram_type.startswith("LPDDR") or modules >= 2 else max(modules, 1)
    ram_peak = mts * 8 * channels / 1000
    ocl = [m for m in metas if m.get("gpu_compute_units")]
    gpu = ocl[0] if ocl else {}
    name = gpu.get("gpu_device", "")
    lanes = 128 if re.search(r"nvidia|geforce|rtx|quadro", name, re.I) else 8 if re.search(r"intel|iris|uhd|arc", name, re.I) else 64
    cus = fnum(gpu.get("gpu_compute_units"))
    # The GPU may boost above the clock its driver reports: the highest clock the sensor log saw counts if higher.
    mhz = max((v for v in (fnum(gpu.get("gpu_max_clock_mhz")), fnum(system.get("gpu_clock_max_logged_mhz")),
                           fnum(system.get("gpu_max_clock_mhz"))) if np.isfinite(v)), default=float("nan"))
    discrete = bool(re.search(DISCRETE_GPU, name, re.I))
    gpu_mem = fnum(system.get("gpu_memory_gbs")) if discrete else ram_peak   # an integrated GPU uses system RAM
    rnd = lambda v, d=1: round(v, d) if np.isfinite(v) else None
    return ({"cpu_fp64": rnd(cpu_peak), "ram": rnd(ram_peak), "gpu_fp32": rnd(cus * lanes * 2 * mhz / 1000), "gpu_mem": rnd(gpu_mem)},
            {"channels": channels, "cus": cus, "mhz": mhz, "discrete": discrete, "gpu_name": re.sub(r"\s*\(.*\)\s*$", "", name),
             "p_cores": p_cores, "e_cores": e_cores, "ghz": ghz, "e_ghz": e_ghz,
             "clock_measured": system.get("cpu_max_mhz_source", "").startswith("measured")})


# ---------------------------------------------------------------------------- data for one machine
def build(folder):
    rows, metas, systems, hardware = load(folder)
    full = rows[rows["mode"] == "full"]
    if full.empty:
        raise SystemExit(f"{folder.name}: no full-mode runs (run ./run_all.sh or run_all.cmd first)")
    spec = pd.read_csv(ROOT / "common" / "tests.csv")
    order, cat_of, style_of = list(spec.test), spec.set_index("test").category, spec.set_index("test")["style"]
    langs = [l for l in LANGS if l in set(full.language)]

    med = full.groupby(["batch", "language", "test", "unit", "threads", "size"], as_index=False).agg(
        rate=("rate", "median"), seconds=("seconds", "median"))
    best_b = med.groupby(["batch", "language", "test", "unit"], as_index=False).rate.max()
    newest = {l: best_b[best_b.language == l].batch.max() for l in langs}
    best = best_b[best_b.apply(lambda r: r.batch == newest[r.language], axis=1)]
    rate = best.pivot(index="test", columns="language", values="rate").reindex([t for t in order if t in set(best.test)])
    rate = rate.reindex(columns=langs)
    spread = best_b.groupby(["language", "test"]).rate.agg(["min", "max", "count", "mean", "std"])
    ratio = rate.div(rate["C++"], axis=0) if "C++" in rate else rate * np.nan

    categories = {c: {l: gmean(ratio.loc[[t for t in ratio.index if cat_of[t] == c], l]) for l in langs}
                  for c in CATS if any(cat_of[t] == c for t in ratio.index)}
    overall = {l: gmean([categories[c][l] for c in categories]) for l in langs}
    unit = best.drop_duplicates("test").set_index("test").unit

    tests = []
    for t in rate.index:
        scale, label = DISPLAY[unit[t]]
        tests.append({"test": t, "category": cat_of[t], "style": style_of[t], "unit": label,
                      "rate": {l: (round(rate.loc[t, l] / scale, 4) if np.isfinite(rate.loc[t, l]) else None) for l in langs},
                      "range": {l: ([round(spread.loc[(l, t), "min"] / scale, 4), round(spread.loc[(l, t), "max"] / scale, 4),
                                     int(spread.loc[(l, t), "count"])] if (l, t) in spread.index else None) for l in langs}})

    def newest_rows(test):
        m = med[med.test == test]
        return m[m.apply(lambda r: r.batch == newest.get(r.language), axis=1)]

    pm = newest_rows("parallel_mandelbrot")
    workers = sorted(int(w) for w in pm.threads.unique())
    parallel = {}
    for l in langs:
        p = pm[pm.language == l].set_index("threads").rate
        parallel[l] = [round(float(p[w] / p[1]), 2) if (w in p and 1 in p) else None for w in workers]
    tri = newest_rows("mem_triad")
    triad = {str(int(th)): {r.language: round(r.rate / 1e9, 2) for _, r in tri[tri.threads == th].iterrows()}
             for th in sorted(tri.threads.unique())}
    al = newest_rows("mem_alloc")
    sizes = sorted(float(s) for s in al["size"].unique())
    pick = lambda l, s, col: (lambda x: round(float(x.iloc[0]), 2) if len(x) else None)(al[(al.language == l) & (al["size"] == s)][col])
    alloc = {l: [None if (v := pick(l, s, "rate")) is None else round(v / 1e9, 2) for s in sizes] for l in langs}
    alloc_s = {l: [pick(l, s, "seconds") for s in sizes] for l in langs}

    run_meta = {l: next(m for rid, m in metas.items() if m.get("language") == l and m.get("batch") == newest[l] and m.get("mode") == "full")
                for l in langs}
    system = dict(next((s for s in reversed(systems) if s.get("batch") in set(newest.values())), systems[-1] if systems else {}))
    # Measured clocks depend on the load at that moment: use the highest any snapshot of this machine recorded.
    for k in ("cpu_max_mhz", "cpu_e_max_mhz", "gpu_clock_max_logged_mhz"):
        best = max((fnum(s.get(k)) for s in systems if np.isfinite(fnum(s.get(k)))), default=None)
        if best is not None:
            system[k] = f"{best:.0f}"
    system.update({k: v for k, v in hardware.items() if v != ""})        # optional hand-written hardware.csv wins
    peaks, hw = machine_peaks(system, list(run_meta.values()))

    def best_of(ts, l):
        v = [rate.loc[t, l] for t in ts if t in rate.index and np.isfinite(rate.loc[t, l])]
        return max(v) if v else None
    util = []
    for name, ts, peak, u in [("CPU compute (FP64 matrix multiply)", ["matmul_blas"], peaks["cpu_fp64"], "GFLOP/s"),
                              ("RAM bandwidth (copy or triad, one core)", ["mem_copy", "mem_triad"], peaks["ram"], "GB/s"),
                              ("GPU compute (FP32 peak kernel)", ["gpu_fp32_peak"], peaks["gpu_fp32"], "GFLOP/s"),
                              ("GPU memory bandwidth", ["gpu_bandwidth"], peaks["gpu_mem"], "GB/s")]:
        pct = {l: (round(100 * best_of(ts, l) / 1e9 / peak, 1) if peak and best_of(ts, l) else None) for l in langs}
        if any(v is not None for v in pct.values()):
            util.append({"name": name, "peak": peak, "unit": u, "pct": pct})

    sensors = None
    for b in sorted(set(newest.values()), reverse=True):
        f = folder / f"sensors_{b}.csv"
        if f.exists():
            s = pd.read_csv(f)
            s["t"] = (s.time - s.time.iloc[0]).round(1)
            col = lambda c, f: [None if pd.isna(v) else f(v) for v in s[c]] if c in s else [None] * len(s)
            sensors = {"batch": b, "t": s.t.tolist(), "phase": s.phase.fillna("idle").tolist(),
                       "cpu_temp": col("cpu_temp_c", lambda v: round(v, 1)), "cpu_mhz": col("cpu_mhz_avg", int),
                       "gpu_busy": [0 if v is None else v for v in col("gpu_busy_pct", int)],
                       "ram": col("mem_used_gib", lambda v: round(v, 2))}
            break

    labels = {}
    for l in langs:
        m, v = run_meta[l], run_meta[l].get("language_version", "")
        if l == "C++":
            v += ", -O3" if "-O3" in m.get("build", "") else ""
        elif l == "Python" and m.get("numpy_version"):
            v += f" + NumPy {m['numpy_version']}"
        elif l == "JavaScript":
            v = re.sub(r"\s*\(V8.*\)$", "", v)
        labels[l] = v

    full_batches = sorted(full.batch.unique())
    check = verify.compare(str(folder))
    return {"id": folder.name, "langs": langs, "categories": categories, "overall": overall, "tests": tests, "workers": workers,
            "parallel": parallel, "triad": triad, "alloc_sizes": sizes, "alloc": alloc, "alloc_seconds": alloc_s,
            "peaks": peaks, "util": util, "sensors": sensors,
            "system": {k: system.get(k, "") for k in ["vendor", "product", "chassis", "cpu", "cpu_max_mhz", "physical_cores",
                                                      "logical_cpus", "ram_gib", "ram_type", "ram_speed_mts", "ram_modules",
                                                      "ram_module_sizes_gib", "gpus", "os", "kernel", "power", "platform_profile",
                                                      "suite_tests_sha"]},
            "version_labels": labels, "newest_batch": newest,
            "_hw": hw, "_ratio": ratio, "_rate": rate, "_style": style_of, "_cat": cat_of, "_meta": run_meta, "_check": check,
            "_full_batches": full_batches, "_spread": spread, "_hardware": hardware}


# ---------------------------------------------------------------------------- text for one machine
def is_windows(S):
    return S.get("os", "").lower().startswith("windows")


def machine_name(S):
    vendor = re.sub(r",?\s+Inc\.?$", "", S.get("vendor", "")).strip()
    product = S.get("product", "").strip()
    return product if vendor and product.lower().startswith(vendor.lower()) else f"{vendor} {product}".strip()


def texts(D):
    S, P, hw, langs = D["system"], D["peaks"], D["_hw"], D["langs"]
    ratio, style, cat = D["_ratio"], D["_style"], D["_cat"]
    others = [l for l in langs if l != "C++"]
    windows = is_windows(S)
    kind = S.get("chassis") or "machine"
    cores, threads = S.get("physical_cores", "?"), S.get("logical_cpus", "?")
    hybrid = np.isfinite(hw["p_cores"]) and hw["e_cores"] > 0
    max_ghz = hw["ghz"]
    n_full = len(D["_full_batches"])
    cv = (D["_spread"]["std"] / D["_spread"]["mean"]).dropna()
    T = {}
    T["kind"] = kind
    T["machine_name"] = machine_name(S)
    T["os_family"] = "Windows" if windows else "Linux"
    T["switch_label"] = f"{T['machine_name']} · {T['os_family']}"
    T["runs_text"] = f"{word(n_full)} full run{'s' if n_full != 1 else ''}"
    T["spread_text"] = f"typically {100 * cv.median():.1f}% between runs" if len(cv) else "a single run, so no spread yet"
    day = datetime.strptime(max(D["newest_batch"].values())[:8], "%Y%m%d")
    T["date_text"] = f"{day.day} {day:%B %Y}"
    secs = sum(fnum(D["_meta"][l].get("elapsed_s"), 0) for l in langs)
    T["minutes_text"] = f"A full run took about {max(1, round(secs / 60))} minutes on this {kind}."
    core_desc = (f"{cores} physical cores ({hw['p_cores']:.0f} performance and {hw['e_cores']:.0f} efficiency cores)"
                 if hybrid else f"{cores} physical cores")
    T["cores_text"] = (f"The same Mandelbrot work split across 1 to {max(D['workers'])} workers. The CPU has {core_desc} "
                       f"and {threads} hardware threads. C++ and JavaScript run their workers as threads; Python and R "
                       f"start separate processes" + (" (on Windows both start them fresh, as Windows cannot fork)." if windows else "."))
    channels = hw["channels"]
    T["mem_text"] = ((f"Bandwidth against the memory's theoretical peak of {num(P['ram'])} GB/s ({S.get('ram_type')}-"
                      f"{S.get('ram_speed_mts')}, {word(channels)} channel{'s' if channels != 1 else ''}). ")
                     if P["ram"] else "Memory bandwidth (the memory's theoretical peak is unknown for this machine). ") + \
        "Python and R work on one core; C++ also runs the triad on all cores."
    sizes = S.get("ram_module_sizes_gib", "").split(" ")[0]
    memory_text = (f"{S.get('ram_modules')} × {sizes} GB {S.get('ram_type')}-{S.get('ram_speed_mts')}"
                   + (f", {word(channels)} channel{'s' if channels != 1 else ''} ({num(P['ram'])} GB/s peak)" if P["ram"] else ""))
    gpu_name = hw["gpu_name"] or "GPU"
    if np.isfinite(hw["cus"]):
        T["gpu_title"] = f"GPU ({gpu_name}, {hw['cus']:.0f} compute units)"
        gpu_spec = f"{gpu_name} · {hw['cus']:.0f} compute units · {hw['mhz'] / 1000:.2g} GHz"
    else:
        T["gpu_title"] = "GPU"
        gpu_spec = gpu_name if hw["gpu_name"] else (S.get("gpus") or "none found")
    gpu_rates = {t["test"]: t["rate"] for t in D["tests"] if t["category"] == "gpu"}
    up = gpu_rates.get("gpu_upload", {})
    slower = f", so its uploads are {up['C++'] / up['R']:.0f}× slower than C++'s here" if up.get("R") and up.get("C++") else ""
    api = "WebGPU translation (Direct3D 12 underneath on Windows)" if windows else "WebGPU translation (Vulkan underneath)"
    where = (" This GPU has its own memory, so uploads and downloads cross the PCIe bus." if hw["discrete"]
             else " This GPU shares the system RAM.")
    T["gpu_text"] = (f"C++, Python and R run identical OpenCL kernels; JavaScript runs a line-by-line {api}. Once a "
                     "kernel runs, the language barely matters. Moving data does: R has no 32-bit number type and converts every "
                     f"value{slower}.{where}")
    if not P["cpu_fp64"]:
        cpu_peak = "The CPU's maximum clock is unknown, so its compute peak is left out"
    elif hybrid and np.isfinite(hw["e_ghz"]):
        cpu_peak = (f"The CPU peak assumes all {hw['p_cores']:.0f} performance cores at {max_ghz:.1f} GHz and all "
                    f"{hw['e_cores']:.0f} efficiency cores at {hw['e_ghz']:.1f} GHz"
                    + (" (the highest clocks measured on one busy core of each type)" if hw["clock_measured"] else ""))
    else:
        cpu_peak = (f"The CPU peak assumes all {cores} cores at full boost ({max_ghz:.1f} GHz"
                    + (", the highest clock measured on one busy core)" if hw["clock_measured"] else ")"))
    T["peak_text"] = (f"Best measured rate as a share of the theoretical peak. {cpu_peak}, which few laptops hold under sustained "
                      "load; matrix multiply goes through OpenBLAS"
                      + (" in every language except R, which on Windows uses its own reference BLAS." if windows and "rblas" in
                         D["_meta"].get("R", {}).get("blas", "").lower() else " in every language."))
    s = D["sensors"]
    T["heat_title"] = "Heat and clock speed during a run" if s and any(v is not None for v in s["cpu_temp"]) else "Clock speed during a run"
    if s:
        busy = [m for m, p in zip(s["cpu_mhz"], s["phase"]) if m is not None and p != "idle"]
        temps = [v for v in s["cpu_temp"] if v is not None]
        clock = (f"while the benchmark ran the average clock across all cores was {np.mean(busy) / 1000:.1f} GHz"
                 + (f" against a {max_ghz:.1f} GHz maximum" if np.isfinite(max_ghz) else ""))
        if temps:
            T["heat_text"] = f"One full run, sampled every second. The CPU peaked at {max(temps):.0f} °C, and {clock}."
        else:
            T["heat_text"] = (f"One full run, sampled every second. Windows lets no program read the CPU temperature without "
                              f"administrator rights, so this shows the clock only (Windows' effective-clock counter): {clock}.")
    profile = S.get("platform_profile")
    T["power_text"] = (f"The {kind} was on {S.get('power') or 'unknown'} power"
                       + (f" with Windows' {profile} power mode." if windows and profile else
                          f" with the {profile} power profile." if profile else "."))
    T["cool_text"] = ("Windows lets no program read the CPU temperature without administrator rights, so the runner pauses "
                      "30 seconds before each language instead of waiting for the CPU to cool."
                      if windows else "Before each language, the runner waits for the CPU to cool below 60 °C.")
    if windows:
        T["alloc_note"] = ("Windows hands every language normal 4 KB pages (its large pages need a special privilege) and zeroes "
                           "each page on first touch, so first use of new memory costs every language about the same.")
    else:
        T["alloc_note"] = ("NumPy asks Linux for 2 MB memory pages for big arrays, which cuts the number of page faults when new "
                           "memory is first touched; the other languages use normal 4 KB pages.")
    T["runs_label"] = f"{T['runs_text']}, {S.get('power') or '?'} power" + (f", {profile} {'mode' if windows else 'profile'}" if profile else "")

    cpu_name = re.sub(r" with .*Graphics$", "", S.get("cpu", ""))
    threads_text = f"{cores} cores ({hw['p_cores']:.0f} P + {hw['e_cores']:.0f} E) / {threads} threads" if hybrid else f"{cores} cores / {threads} threads"
    os_text = (f"{S.get('os')} · build {'.'.join(S.get('kernel', '').split('.')[2:])}" if windows
               else f"{S.get('os')} · Linux {S.get('kernel', '').split('-')[0]}")
    T["spec"] = [["Machine", T["machine_name"]],
                 ["CPU", f"{cpu_name} · {threads_text}" + (f" · up to {max_ghz:.1f} GHz" if np.isfinite(max_ghz) else "")],
                 ["Memory", memory_text], ["GPU", gpu_spec], ["System", os_text], ["Runs", T["runs_label"]]]

    chk = D["_check"]
    if chk and chk["bad"] == 0:
        n = len({r["test"] for r in chk["rows"]})
        skipped = sorted({l for r in chk["rows"] for l, v in r["values"].items() if v == "skipped"})
        T["verify_badge"] = (f"All {word(len(chk['langs']))} languages computed identical answers on all {n} tests"
                             + (f" ({', '.join(skipped)} skipped some)" if skipped else ""))
        T["verify_text"] = (f"A verify mode runs tiny identical inputs and checks that all languages produce the same answers; "
                            f"on this {kind} they agree on all {n} tests, to the last digit for integer results.")
    elif chk:
        T["verify_text"] = f"A verify run on this {kind} found {chk['bad']} test(s) where the languages disagree."
    else:
        T["verify_text"] = "Run the runner with --verify to check that all languages produce the same answers."

    # findings
    items = []
    loops = [t for t in ratio.index if style[t] == "loop"]
    libs = [t for t in ratio.index if style[t] in ("vectorized", "builtin", "blas") and cat[t] != "gpu"]
    for l in sorted(others, key=lambda l: -(D["overall"][l] or 0)):
        r = ratio[l].dropna()
        if r.empty:
            continue
        hi, lo = r.idxmax(), r.idxmin()
        lead = f"{esc(l)}: {fx(D['overall'][l])} of C++ speed overall."
        best = (f"Beats C++ on {mono(hi)} ({fx(r[hi])})" if r[hi] > 1.05 else f"Comes closest on {mono(hi)} ({fx(r[hi])})")
        worst = f"falls furthest behind on {mono(lo)} ({fx(r[lo])}, {1 / r[lo]:.0f}× slower)" if r[lo] < 0.95 else ""
        a, b = gmean(ratio.loc[loops, l]), gmean(ratio.loc[libs, l])
        mix = (f" Plain loops run at {fx(a)} of C++ and library or vectorized code at {fx(b)}." if a and b else "")
        items.append((l, f"<b>{lead}</b> {best}{' and ' + worst if worst else ''}.{mix}"))
    par = {l: max(v for v in D["parallel"][l] if v is not None) for l in langs if any(v is not None for v in D["parallel"][l])}
    if par:
        parts = ", ".join(f"{esc(l)} {fx(v)}" for l, v in sorted(par.items(), key=lambda kv: -kv[1]))
        items.append((None, f"<b>All cores:</b> the best parallel speedups are {parts}, on {cores} physical cores."))
    ceil = []
    mm = next((t for t in D["tests"] if t["test"] == "matmul_blas"), None)
    if mm and P["cpu_fp64"]:
        g = max(v for v in mm["rate"].values() if v)
        ceil.append(f"matrix multiply reaches {num(g)} GFLOP/s at best ({100 * g / P['cpu_fp64']:.0f}% of the CPU's theoretical peak)")
    gp = next((t for t in D["tests"] if t["test"] == "gpu_fp32_peak"), None)
    if gp and P["gpu_fp32"]:
        g = max(v for v in gp["rate"].values() if v)
        ceil.append(f"GPU compute reaches {num(g)} GFLOP/s ({100 * g / P['gpu_fp32']:.0f}% of its peak)")
    if ceil:
        items.append((None, f"<b>The hardware sets the ceiling:</b> {'; '.join(ceil)}, about the same from every language"
                            + (" that uses OpenBLAS." if windows and mm and mm["rate"].get("R") and "rblas" in
                               D["_meta"].get("R", {}).get("blas", "").lower() else ".")))
    key = lambda l: ' style="--k: var(--c-%s)"' % CSS_KEY[l] if l else ""
    T["findings"] = "\n".join(f"<li{key(l)}><span>{body}</span></li>" for l, body in items)
    return T


# ---------------------------------------------------------------------------- comparing machines
def compare(Ds):
    """Data and text for the machine comparison. The machine with the oldest full run is the reference."""
    base, others = Ds[0], Ds[1:]
    langs = [l for l in LANGS if any(l in D["langs"] for D in Ds)]
    tests = [t for t in base["_rate"].index if all(t in D["_rate"].index for D in others)]
    cat = base["_cat"]
    out = {"base": base["id"], "others": [D["id"] for D in others], "langs": langs, "tests": [], "overall": {}, "categories": {}}
    rels = {}                                    # machine -> language -> test -> speed relative to the reference
    for D in others:
        def ratio(l, t):
            if l not in D["_rate"] or l not in base["_rate"]:
                return None
            r = D["_rate"].loc[t, l] / base["_rate"].loc[t, l]
            return float(r) if np.isfinite(r) else None
        rel = rels[D["id"]] = {l: {t: ratio(l, t) for t in tests} for l in langs}
        cats = {c: {l: gmean([rel[l][t] for t in tests if cat[t] == c]) for l in langs} for c in CATS if any(cat[t] == c for t in tests)}
        out["categories"][D["id"]] = cats
        out["overall"][D["id"]] = {l: gmean([cats[c][l] for c in cats]) for l in langs}
    for t in tests:
        out["tests"].append({"test": t, "category": cat[t],
                             "ratio": {m: {l: (round(r, 4) if (r := rels[m][l][t]) else None) for l in langs} for m in rels}})

    # text (two machines: the newest compared with the reference)
    B, O = base, others[-1]
    nb, no = machine_name(B["system"]), machine_name(O["system"])
    os_b = "Windows" if is_windows(B["system"]) else "Linux"
    os_o = "Windows" if is_windows(O["system"]) else "Linux"
    gpu_b, gpu_o = B["_hw"]["gpu_name"] or "its GPU", O["_hw"]["gpu_name"] or "its GPU"
    T = {"h1": f"Four languages, {word(len(Ds))} {Ds[0]['system'].get('chassis') or 'machine'}s"
               if len({D['system'].get('chassis') for D in Ds}) == 1 else f"Four languages, {word(len(Ds))} machines",
         "names": {D["id"]: machine_name(D["system"]) for D in Ds}}
    T["lede"] = (f"The same 18 tests, written in C++, Python, R and JavaScript, run on {word(len(Ds))} "
                 f"{'laptops' if T['h1'].endswith('laptops') else 'machines'} - "
                 + " and ".join(f"{machine_name(D['system'])} on {'Windows' if is_windows(D['system']) else 'Linux'}" for D in Ds)
                 + " - to see how each language uses a computer's CPU, memory and GPU.").replace(" - ", " — ")
    shas = {D["system"].get("suite_tests_sha") for D in Ds} - {"", None}
    same = (f"Both ran the same version of the tests (suite {next(iter(shas))})." if len(shas) == 1
            else "Warning: the machines ran different versions of the tests, so these ratios may not be comparable.")
    T["compare_text"] = (f"How many times faster each language runs on the {no} ({os_o}) than on the {nb} ({os_b}), "
                         f"test by test. {same} The differences come from the hardware - and, for a few tests, from the "
                         "operating system and how each language is installed on it.")
    ov = out["overall"][O["id"]]
    pc = lambda c, l: out["categories"][O["id"]].get(c, {}).get(l)
    items = []
    fastest = sorted(((l, v) for l, v in ov.items() if v), key=lambda kv: -kv[1])
    if fastest:
        items.append(f"<b>Overall</b>, the {esc(no)} is " + ", ".join(f"{fx(v)} as fast for {esc(l)}" for l, v in fastest)
                     + f" (geometric mean of the four hardware areas).")
    if pc("gpu", "C++"):
        items.append(f"<b>GPU:</b> the {esc(gpu_o)} runs the same kernels {fx(pc('gpu', 'C++'))} as fast as the {esc(gpu_b)} "
                     f"(C++; averaged over the five GPU tests), the largest gap of any area.")
    if pc("cpu_single", "C++") and pc("cpu_multi", "C++"):
        items.append(f"<b>CPU:</b> one core is {fx(pc('cpu_single', 'C++'))} as fast and all cores together "
                     f"{fx(pc('cpu_multi', 'C++'))} (C++), with {O['system'].get('physical_cores')} cores against "
                     f"{B['system'].get('physical_cores')}.")
    if pc("ram", "C++"):
        items.append(f"<b>RAM:</b> {fx(pc('ram', 'C++'))} as fast for C++ ({O['system'].get('ram_type')}-{O['system'].get('ram_speed_mts')} "
                     f"against {B['system'].get('ram_type')}-{B['system'].get('ram_speed_mts')}).")
    # tests where a language moves very differently from C++ point at the software, not the hardware
    odd = []
    rel = rels[O["id"]]
    for t in tests:
        c = rel.get("C++", {}).get(t)
        for l in langs:
            r = rel.get(l, {}).get(t)
            if l != "C++" and c and r and (r / c > 2.5 or c / r > 2.5):
                odd.append((abs(np.log(r / c)), l, t, r, c))
    for _, l, t, r, c in sorted(odd, reverse=True)[:3]:
        why = ""
        if l == "R" and t == "matmul_blas" and ("rblas" in O["_meta"].get("R", {}).get("blas", "").lower()
                                                  or "rblas" in B["_meta"].get("R", {}).get("blas", "").lower()):
            why = (" R for Windows ships a single-threaded reference BLAS, while R on Ubuntu uses the system OpenBLAS: "
                   "that is the software, not the hardware.")
        items.append(f"<b>{esc(l)} on {mono(t)}:</b> {fx(r)} as fast on the {esc(no)}, while C++ is {fx(c)} as fast.{why}")
    T["compare_findings"] = "\n".join(f"<li><span>{x}</span></li>" for x in items)
    out["text"] = T
    return out


# ---------------------------------------------------------------------------- output
def public(D):
    return {k: v for k, v in D.items() if not k.startswith("_")}


def render(Ds, cmp, fragment=False):
    page = (ROOT / "docs" / "page_template.html").read_text(encoding="utf-8")
    page = re.sub(r"\A<!--.*?-->\s*", "", page, flags=re.S)        # the template's own comment
    if cmp is None:
        page = re.sub(r"\s*<!-- optional:compare -->.*?<!-- /optional:compare -->", "", page, flags=re.S)
    T = cmp["text"] if cmp else {"h1": f"Four languages, one {Ds[0]['T']['kind']}",
                                 "lede": (f"The same 18 tests, written in C++, Python, R and JavaScript, run on one "
                                          f"{Ds[0]['T']['kind']} to see how each language uses its CPU, memory and GPU.")}
    page = page.replace("{{h1}}", esc(T["h1"])).replace("{{lede}}", esc(T["lede"]))
    left = re.findall(r"\{\{\w+\}\}", page)
    if left:
        raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
    data = {"machines": Ds, "default": max(Ds, key=lambda D: max(D["newest_batch"].values()))["id"], "compare": cmp}
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace("__DATA__", blob)
    if fragment:
        return page
    cut = page.index("</style>") + len("</style>")
    names = " and ".join(D["T"]["machine_name"] for D in Ds)
    desc = f"How C++, Python, R and JavaScript use a computer's CPU, memory and GPU ({names}), measured with one shared benchmark suite."
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{html.escape(desc)}">
<meta property="og:title" content="{html.escape(T['h1'])}">
<meta property="og:description" content="{html.escape(desc)}">
<meta property="og:type" content="website">
<style>
:root {{ color-scheme: light; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }}
body {{ margin: 0; }}
img {{ max-width: 100%; }}
[hidden] {{ display: none !important; }}
</style>
{page[:cut].strip()}
</head>
<body>
{page[cut:].strip()}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--machine", action="append", help="machine folder in results/ (repeatable; default: all with full runs)")
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "index.html"))
    parser.add_argument("--fragment", action="store_true", help="write the page body only, without <html>/<head>")
    parser.add_argument("--list", action="store_true", help="list machines with full runs and exit")
    args = parser.parse_args()

    machines = machines_with_full_runs(Path(args.results))
    if args.list or not machines:
        for m, (_, b) in sorted(machines.items(), key=lambda kv: kv[1][1], reverse=True):
            print(f"{m}  (newest full run {b})")
        return 0 if machines else 1
    chosen = args.machine or list(machines)
    unknown = [m for m in chosen if m not in machines]
    if unknown:
        raise SystemExit(f"no full runs for {', '.join(unknown)}. Machines: {', '.join(machines)}")
    chosen.sort(key=lambda m: machines[m][0])                      # oldest first: the comparison's reference
    built = [build(Path(args.results) / m) for m in chosen]
    cmp = compare(built) if len(built) > 1 else None
    Ds = []
    for D in built:
        P = public(D)
        P["T"] = texts(D)
        Ds.append(P)
    Path(args.out).write_text(render(Ds, cmp, args.fragment), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: " + "; ".join(f"{D['id']} ({', '.join(D['langs'])}; newest full runs "
                                            f"{sorted(set(D['newest_batch'].values()))})" for D in Ds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
