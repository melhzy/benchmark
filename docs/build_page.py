#!/usr/bin/env python3
"""
build_page.py -- build the results page (docs/index.html, served by GitHub Pages) from results/.

Usage
  python3 docs/build_page.py                    the machine with the newest full run
  python3 docs/build_page.py --machine NAME     a specific machine (its folder name in results/)
  python3 docs/build_page.py --list             list machines that have full runs
  python3 docs/build_page.py --out FILE         write somewhere else (default: docs/index.html)
  python3 docs/build_page.py --fragment         page body only (no <html>/<head> wrapper)

The page shows one machine: the newest full-mode run of each language, the spread across all
full runs, the sensor log of the newest full run, and - if the machine's folder holds a verify
run - whether all languages computed identical answers. Text, chart ranges, theoretical peaks
and findings are all computed from that machine's results. Template: docs/page_template.html.
Needs numpy and pandas (requirements.txt).
"""

import argparse
import csv
import html
import json
import math
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
DISPLAY = {"FLOP": (1e9, "GFLOP/s"), "B": (1e9, "GB/s"), "access": (1e6, "M accesses/s"), "px": (1e6, "M px/s"),
           "iter": (1e6, "M iter/s"), "call": (1e6, "M calls/s"), "elem": (1e6, "M elem/s"),
           "item": (1e6, "M items/s"), "op": (1e6, "M ops/s")}
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


# ---------------------------------------------------------------------------- loading
def read_kv(path):
    with open(path, newline="") as f:
        return {r[0]: r[1] for r in csv.reader(f) if len(r) == 2 and r[0] != "key"}


def machines_with_full_runs(results):
    found = {}
    for folder in sorted(p for p in results.iterdir() if p.is_dir()):
        batches = [read_kv(m).get("batch", "") for m in folder.glob("*_meta.csv") if read_kv(m).get("mode") == "full"]
        if batches:
            found[folder.name] = max(batches)
    return found


def load(folder):
    frames, metas = [], {}
    for f in sorted(folder.glob("*.csv")):
        if f.name.startswith(("sensors_", "system_")) or f.name.endswith("_meta.csv"):
            continue
        d = pd.read_csv(f, dtype={"batch": str})
        if d.empty:
            continue
        frames.append(d)
        meta = f.with_name(f.stem + "_meta.csv")
        metas[d.run_id.iloc[0]] = read_kv(meta) if meta.exists() else {}
    rows = pd.concat(frames, ignore_index=True)
    systems = sorted((read_kv(f) for f in folder.glob("system_*.csv")), key=lambda d: d.get("batch", ""))
    return rows, metas, systems


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


# ---------------------------------------------------------------------------- theoretical peaks
def machine_peaks(system, metas):
    """Same estimates as analysis.ipynb (section 2): CPU FP64, RAM bandwidth, GPU FP32."""
    first = lambda key: next((m[key] for m in metas if m.get(key) not in (None, "")), "")
    cpu = system.get("cpu") or first("cpu")
    flags = set(system.get("cpu_flags", "").split())
    cores = fnum(system.get("physical_cores"), fnum(first("physical_cores"), 1))
    ghz = fnum(system.get("cpu_max_mhz")) / 1000
    if "avx512f" in flags and "intel" in cpu.lower():
        fpc = 32
    elif {"avx2", "fma"} <= flags:
        fpc = 16
    elif "avx" in flags or "asimd" in flags:
        fpc = 8
    else:
        fpc = 4
    ram_type, mts = system.get("ram_type", ""), fnum(system.get("ram_speed_mts"))
    modules = int(fnum(system.get("ram_modules"), 0))
    channels = 2 if ram_type.startswith("LPDDR") or modules >= 2 else max(modules, 1)
    ocl = [m for m in metas if m.get("gpu_compute_units")]
    gpu = ocl[0] if ocl else {}
    lanes = 128 if re.search(r"nvidia|geforce|rtx|quadro", gpu.get("gpu_device", ""), re.I) else \
        8 if re.search(r"intel|iris|uhd|arc", gpu.get("gpu_device", ""), re.I) else 64
    cus, mhz = fnum(gpu.get("gpu_compute_units")), fnum(gpu.get("gpu_max_clock_mhz"))
    rnd = lambda v, d=1: round(v, d) if np.isfinite(v) else None
    return ({"cpu_fp64": rnd(cores * fpc * ghz), "ram": rnd(mts * 8 * channels / 1000), "gpu_fp32": rnd(cus * lanes * 2 * mhz / 1000)},
            {"channels": channels, "cus": cus, "mhz": mhz, "gpu_name": re.sub(r"\s*\(.*\)\s*$", "", gpu.get("gpu_device", ""))})


# ---------------------------------------------------------------------------- data for the page
def build(folder):
    rows, metas, systems = load(folder)
    full = rows[rows["mode"] == "full"]
    if full.empty:
        raise SystemExit(f"{folder.name}: no full-mode runs (run ./run_all.sh first)")
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
    system = next((s for s in reversed(systems) if s.get("batch") in set(newest.values())), systems[-1] if systems else {})
    peaks, hw = machine_peaks(system, list(run_meta.values()))

    def best_of(ts, l):
        v = [rate.loc[t, l] for t in ts if t in rate.index and np.isfinite(rate.loc[t, l])]
        return max(v) if v else None
    util = []
    for name, ts, peak, u in [("CPU compute (FP64 matrix multiply)", ["matmul_blas"], peaks["cpu_fp64"], "GFLOP/s"),
                              ("RAM bandwidth (copy or triad, one core)", ["mem_copy", "mem_triad"], peaks["ram"], "GB/s"),
                              ("GPU compute (FP32 peak kernel)", ["gpu_fp32_peak"], peaks["gpu_fp32"], "GFLOP/s"),
                              ("GPU memory bandwidth", ["gpu_bandwidth"], peaks["ram"], "GB/s")]:
        pct = {l: (round(100 * best_of(ts, l) / 1e9 / peak, 1) if peak and best_of(ts, l) else None) for l in langs}
        if any(v is not None for v in pct.values()):
            util.append({"name": name, "peak": peak, "unit": u, "pct": pct})

    sensors = None
    for b in sorted(set(newest.values()), reverse=True):
        f = folder / f"sensors_{b}.csv"
        if f.exists():
            s = pd.read_csv(f)
            s["t"] = (s.time - s.time.iloc[0]).round(1)
            sensors = {"batch": b, "t": s.t.tolist(), "phase": s.phase.fillna("idle").tolist(),
                       "cpu_temp": [None if pd.isna(v) else round(v, 1) for v in s.cpu_temp_c],
                       "cpu_mhz": [None if pd.isna(v) else int(v) for v in s.cpu_mhz_avg],
                       "gpu_busy": [0 if pd.isna(v) else int(v) for v in s.gpu_busy_pct],
                       "ram": [None if pd.isna(v) else round(v, 2) for v in s.mem_used_gib]}
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
    return {"langs": langs, "categories": categories, "overall": overall, "tests": tests, "workers": workers,
            "parallel": parallel, "triad": triad, "alloc_sizes": sizes, "alloc": alloc, "alloc_seconds": alloc_s,
            "peaks": peaks, "util": util, "sensors": sensors,
            "system": {k: system.get(k, "") for k in ["vendor", "product", "chassis", "cpu", "cpu_max_mhz", "physical_cores",
                                                      "logical_cpus", "ram_gib", "ram_type", "ram_speed_mts", "ram_modules",
                                                      "ram_module_sizes_gib", "gpus", "os", "kernel", "power", "platform_profile"]},
            "version_labels": labels, "newest_batch": newest,
            "_hw": hw, "_ratio": ratio, "_style": style_of, "_cat": cat_of, "_meta": run_meta, "_check": check,
            "_full_batches": full_batches, "_spread": spread}


# ---------------------------------------------------------------------------- text
def texts(D):
    S, P, hw, langs = D["system"], D["peaks"], D["_hw"], D["langs"]
    ratio, style, cat = D["_ratio"], D["_style"], D["_cat"]
    others = [l for l in langs if l != "C++"]
    machine_name = re.sub(r"\s+Inc\.?$", "", S.get("vendor", "")).strip() + " " + S.get("product", "")
    kind = S.get("chassis") or "machine"
    cores, threads = S.get("physical_cores", "?"), S.get("logical_cpus", "?")
    max_ghz = fnum(S.get("cpu_max_mhz")) / 1000
    n_full = len(D["_full_batches"])
    cv = (D["_spread"]["std"] / D["_spread"]["mean"]).dropna()
    T = {}
    T["h1"] = f"Four languages, one {kind}"
    T["kind"] = kind
    T["machine_name"] = machine_name.strip()
    T["runs_text"] = f"{word(n_full)} full run{'s' if n_full != 1 else ''}"
    T["spread_text"] = f"typically {100 * cv.median():.1f}% between runs" if len(cv) else "a single run, so no spread yet"
    T["date_text"] = datetime.strptime(max(D["newest_batch"].values())[:8], "%Y%m%d").strftime("%-d %B %Y")
    secs = sum(fnum(D["_meta"][l].get("elapsed_s"), 0) for l in langs)
    T["minutes_text"] = f"about {max(1, round(secs / 60))} minutes on this {kind}"
    T["cores_text"] = (f"The same Mandelbrot work split across 1 to {max(D['workers'])} workers. The CPU has {cores} physical "
                       f"cores and {threads} hardware threads. C++ and JavaScript run their workers as threads; Python and R "
                       f"start separate processes.")
    channels = hw["channels"]
    T["mem_text"] = ((f"Bandwidth against the memory's theoretical peak of {num(P['ram'])} GB/s ({S.get('ram_type')}-"
                      f"{S.get('ram_speed_mts')}, {word(channels)} channel{'s' if channels != 1 else ''}). ")
                     if P["ram"] else "Memory bandwidth (the memory's theoretical peak is unknown for this machine). ") + \
        "Python and R work on one core; C++ also runs the triad on all cores."
    D["memory_text"] = (f"{S.get('ram_modules')} × {S.get('ram_module_sizes_gib', '').split(' ')[0]} GB {S.get('ram_type')}-"
                        f"{S.get('ram_speed_mts')}" + (f", {word(channels)} channel{'s' if channels != 1 else ''}"
                                                        f" ({num(P['ram'])} GB/s peak)" if P["ram"] else ""))
    gpu_name = hw["gpu_name"] or "GPU"
    if np.isfinite(hw["cus"]):
        T["gpu_title"] = f"GPU ({gpu_name}, {hw['cus']:.0f} compute units)"
        D["gpu_text"] = f"{gpu_name} · {hw['cus']:.0f} compute units · {hw['mhz'] / 1000:.1f} GHz"
    else:
        T["gpu_title"] = "GPU"
        D["gpu_text"] = gpu_name if hw["gpu_name"] else (S.get("gpus") or "none found")
    gpu_rates = {t["test"]: t["rate"] for t in D["tests"] if t["category"] == "gpu"}
    up = gpu_rates.get("gpu_upload", {})
    slower = f", so its uploads are {up['C++'] / up['R']:.0f}× slower than C++'s here" if up.get("R") and up.get("C++") else ""
    T["gpu_text"] = ("C++, Python and R run identical OpenCL kernels; JavaScript runs a line-by-line WebGPU translation. Once a "
                     "kernel runs, the language barely matters. Moving data does: R has no 32-bit number type and converts every "
                     f"value{slower}.")
    T["peak_text"] = (f"Best measured rate as a share of the theoretical peak. The CPU peak assumes all {cores} cores at full boost "
                      f"({max_ghz:.1f} GHz), which few machines hold under sustained load; matrix multiply goes through OpenBLAS "
                      f"in every language.")
    s = D["sensors"]
    if s:
        busy = [m for m, p in zip(s["cpu_mhz"], s["phase"]) if m is not None and p != "idle"]
        tmax = max(v for v in s["cpu_temp"] if v is not None)
        T["heat_text"] = (f"One full run, sampled every second. The CPU peaked at {tmax:.0f} °C, and while the benchmark ran its "
                          f"average clock was {np.mean(busy) / 1000:.1f} GHz against a {max_ghz:.1f} GHz maximum.")
    profile = S.get("platform_profile")
    T["power_text"] = f"The {kind} was on {S.get('power') or 'unknown'} power" + (f" with the {profile} power profile." if profile else ".")
    D["runs_text"] = f"{T['runs_text']}, {S.get('power') or '?'} power" + (f", {profile} profile" if profile else "")

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
        T["verify_text"] = "Run ./run_all.sh --verify to check that all languages produce the same answers."

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
        items.append((None, f"<b>The hardware sets the ceiling:</b> {'; '.join(ceil)}, about the same from every language."))
    lis = []
    for l, body in items:
        key = ' style="--k: var(--c-%s)"' % CSS_KEY[l] if l else ""
        lis.append(f"      <li{key}><span>{body}</span></li>")
    T["findings"] = "\n".join(lis)
    return T


# ---------------------------------------------------------------------------- output
def render(D, T, fragment=False):
    page = (ROOT / "docs" / "page_template.html").read_text()
    for key in ("verify", "gpu", "heat"):
        missing = {"verify": "verify_badge" not in T, "gpu": not any(t["category"] == "gpu" for t in D["tests"]),
                   "heat": D["sensors"] is None}[key]
        if missing:
            page = re.sub(rf"\s*<!-- optional:{key} -->.*?<!-- /optional:{key} -->", "", page, flags=re.S)
        else:
            page = page.replace(f"<!-- optional:{key} -->", "").replace(f"<!-- /optional:{key} -->", "")
    page = re.sub(r"\A<!--.*?-->\s*", "", page, flags=re.S)        # the template's own comment
    for key, value in T.items():
        page = page.replace("{{" + key + "}}", value if key == "findings" else esc(value))
    left = re.findall(r"\{\{\w+\}\}", page)
    if left:
        raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
    data = {k: v for k, v in D.items() if not k.startswith("_")}
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = page.replace("__DATA__", blob)
    if fragment:
        return page
    cut = page.index("</style>") + len("</style>")
    desc = (f"How C++, Python, R and JavaScript use a {T['kind']}'s CPU, memory and GPU ({T['machine_name']}), measured "
            f"with one shared benchmark suite.")
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
    parser.add_argument("--machine", help="machine folder in results/ (default: the newest full run)")
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "index.html"))
    parser.add_argument("--fragment", action="store_true", help="write the page body only, without <html>/<head>")
    parser.add_argument("--list", action="store_true", help="list machines with full runs and exit")
    args = parser.parse_args()

    machines = machines_with_full_runs(Path(args.results))
    if args.list or not machines:
        for m, b in sorted(machines.items(), key=lambda kv: kv[1], reverse=True):
            print(f"{m}  (newest full run {b})")
        return 0 if machines else 1
    machine = args.machine or max(machines, key=machines.get)
    if machine not in machines:
        raise SystemExit(f"no full runs for '{machine}'. Machines: {', '.join(machines)}")
    D = build(Path(args.results) / machine)
    T = texts(D)
    Path(args.out).write_text(render(D, T, args.fragment))
    print(f"wrote {args.out} for {machine} ({', '.join(D['langs'])}; newest full runs {sorted(set(D['newest_batch'].values()))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
