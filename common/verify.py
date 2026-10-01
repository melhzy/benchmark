#!/usr/bin/env python3
"""
verify.py -- check that C++, Python, R and JavaScript computed the same answers.

Compares the `check` column of runs made with --verify (identical sizes in every language).
Standard library only.

Usage
  python3 common/verify.py                    latest verify runs on this machine (results/<machine>/)
  python3 common/verify.py --batch 20261001-130000
  python3 common/verify.py --dir results/some-other-machine
"""

import argparse
import csv
import glob
import math
import os
import platform
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def machine_id():
    """This computer's results folder name; the same rule as the benchmark programs (common/SPEC.md)."""
    name = os.environ.get("BENCH_MACHINE", "").strip()
    if name:
        return slug(name)
    vendor = product = ""
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\BIOS") as k:
                vendor = str(winreg.QueryValueEx(k, "SystemManufacturer")[0])
                product = str(winreg.QueryValueEx(k, "SystemProductName")[0])
        except OSError:
            pass
    else:
        for f in ("sys_vendor", "product_name"):
            try:
                with open(f"/sys/class/dmi/id/{f}") as fh:
                    value = fh.read().strip()
            except OSError:
                value = ""
            vendor, product = (value, product) if f == "sys_vendor" else (vendor, value)
    v, p = slug(vendor), slug(product)
    return (p if v and (p == v or p.startswith(v + "-")) else slug(f"{vendor} {product}")) or slug(platform.node())

# Relative tolerance per test. Integer results must match exactly; floating-point sums may
# differ in the last digits because languages add numbers in different orders.
EXACT = {"mandelbrot", "parallel_mandelbrot", "fib_recursive", "hashmap", "string_ops",
         "mem_alloc", "mem_copy", "mem_triad", "gpu_upload", "gpu_download", "gpu_bandwidth"}
# GPU: OpenCL (C++/Python/R) and WebGPU (JavaScript) compilers may fuse multiply-adds differently
# (one rounding instead of two), which moves float32 results in the 6th digit.
TOLERANCE = {"gpu_sgemm": 1e-5, "gpu_fp32_peak": 1e-4}
DEFAULT_TOLERANCE = 1e-9
LANGS = ("C++", "Python", "R", "JavaScript")


def load(directory, batch):
    """Return {language: rows}: for each test, the rows of the newest verify-mode run that has it.

    Runs made with --only contain some tests; combining runs this way still gives a full table.
    """
    newest = {}  # (language, test) -> (run_id, rows)
    for path in glob.glob(os.path.join(directory, "*.csv")):
        name = os.path.basename(path)
        if name.endswith("_meta.csv") or name.startswith("sensors_"):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or rows[0].get("mode") != "verify":
            continue
        if batch and rows[0].get("batch") != batch:
            continue
        run_id = rows[0]["run_id"]
        for row in rows:
            row["size"] = f"{float(row['size']):g}"   # "1e+05" and "100000" are the same size
        for test in {r["test"] for r in rows}:
            key = (rows[0]["language"], test)
            if key not in newest or run_id.split("_")[-1] > newest[key][0].split("_")[-1]:
                newest[key] = (run_id, [r for r in rows if r["test"] == test])
    runs = {}
    for (lang, _), (run_id, rows) in newest.items():
        runs.setdefault(lang, []).extend(rows)
    return runs


def intended_skips(directory, run_ids):
    """Return {language: set of tests} that a run skipped on purpose (meta key `skipped`)."""
    skips = {}
    for run_id in run_ids:
        path = os.path.join(directory, f"{run_id}_meta.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            meta = {r["key"]: r["value"] for r in csv.DictReader(f)}
        tests = {item.split(":")[0].strip() for item in meta.get("skipped", "").split(";") if ":" in item}
        skips.setdefault(meta.get("language", ""), set()).update(tests)
    return skips


def agree(values, tol):
    nums = [float(v) for v in values]
    ref = nums[0]
    if tol == 0:
        return all(v == ref for v in nums)
    scale = max(abs(v) for v in nums) or 1.0
    return all(math.isclose(v, ref, rel_tol=tol, abs_tol=tol * scale) for v in nums)


def compare(directory, batch=None):
    """Compare the newest verify-mode checks of every language in `directory`.

    Returns None if there are no verify runs, else a dict with the languages, the number of runs,
    one row per (test, size) with each language's check value and a status, and the count of
    rows that need attention. Used by main() and by docs/build_page.py.
    """
    runs = load(directory, batch)
    if not runs:
        return None
    langs = [l for l in LANGS if l in runs]
    # (test, size) -> {lang: [checks]} ; several rows per test (repeats, worker counts, threads)
    table = {}
    for lang in langs:
        for row in runs[lang]:
            table.setdefault((row["category"], row["test"], row["size"]), {}).setdefault(lang, []).append(row["check"])
    skipped = intended_skips(directory, {r["run_id"] for l in langs for r in runs[l]})
    serial = {size: c for (cat, test, size), c in table.items() if test == "mandelbrot"}
    rows, bad = [], 0
    for (cat, test, size), by_lang in sorted(table.items()):
        tol = 0 if test in EXACT else TOLERANCE.get(test, DEFAULT_TOLERANCE)
        values = [v for l in langs for v in by_lang.get(l, [])]
        ok = agree(values, tol)
        if test == "parallel_mandelbrot" and size in serial:  # parallel must equal serial
            ok = ok and agree(values + [v for vs in serial[size].values() for v in vs], 0)
        missing = [l for l in langs if l not in by_lang]
        unexplained = [l for l in missing if test not in skipped.get(l, set())]
        if not ok:
            status = "MISMATCH"
        elif unexplained:
            status = "missing: " + ",".join(unexplained)
        else:
            status = "OK" + (f" (skipped by {', '.join(missing)})" if missing else "")
        bad += not status.startswith("OK")
        rows.append({"test": test, "size": size, "status": status,
                     "values": {l: (float(by_lang[l][0]) if l in by_lang else
                                    ("skipped" if test in skipped.get(l, set()) else None)) for l in langs}})
    return {"langs": langs, "runs": len({r["run_id"] for l in langs for r in runs[l]}), "rows": rows, "bad": bad}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dir", default=os.path.join(ROOT, "results", machine_id()))
    parser.add_argument("--batch", help="only runs with this BENCH_BATCH id")
    args = parser.parse_args()

    result = compare(args.dir, args.batch)
    if result is None:
        print(f"No --verify runs found in {args.dir}" + (f" for batch {args.batch}" if args.batch else ""))
        return 1
    langs = result["langs"]
    print(f"\nCross-language check ({result['runs']} verify runs: {', '.join(langs)})")
    print(f"  {'test':<20}{'size':>8}  " + "".join(f"{l:>20}" for l in langs) + "  result")
    for row in result["rows"]:
        cells = "".join(f"{v:>20.14g}" if isinstance(v, float) else f"{v or '-':>20}" for v in row["values"].values())
        print(f"  {row['test']:<20}{row['size']:>8}  {cells}  {row['status']}")
    bad = result["bad"]
    print(f"\n{'All results agree.' if not bad else f'{bad} test(s) need attention.'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
