#!/usr/bin/env python3
"""
python_benchmark.py -- measure how fast this computer runs Python.

Standard library only. If NumPy is installed, an extra section measures
matrix-multiply speed (BLAS, all cores).

Usage
  python3 python_benchmark.py                      full run (about 1-2 minutes)
  python3 python_benchmark.py --quick              smaller workloads, quick check
  python3 python_benchmark.py --only single,multi  run selected sections
  python3 python_benchmark.py --save ac.json       save results to a JSON file
  python3 python_benchmark.py --compare ac.json    compare this run with a saved one
  python3 python_benchmark.py --help               all options

Sections
  single  one CPU core: interpreter speed and C-backed library code
  multi   all cores: process scaling, plus threads to show the GIL effect
  memory  bulk copy / scan bandwidth and object allocation
  disk    sequential write/read and random 4 KiB reads (temporary file)
  numpy   matrix multiply GFLOP/s (only if NumPy is installed)

For numbers you can compare between runs: plug in the charger, close heavy
programs, keep the same power profile, and run two or three times. The
header records power source, power profile and CPU temperature.
"""

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

SECTIONS = ("single", "multi", "memory", "disk", "numpy")
SECTION_TITLES = {
    "single": "Single core",
    "multi": "Multi core",
    "memory": "Memory",
    "disk": "Disk",
    "numpy": "NumPy",
}


# ---------------------------------------------------------------------------
# System information (best effort; Linux gives the most detail)
# ---------------------------------------------------------------------------

def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cpu_model():
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def physical_cores():
    cores, phys = set(), "0"
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            phys = value.strip()
        elif key == "core id":
            cores.add((phys, value.strip()))
    return len(cores) or None


def total_ram_gib():
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 2**20
    return None


def power_source():
    base = "/sys/class/power_supply"
    if not os.path.isdir(base):
        return None
    on_ac, battery = False, None
    for name in os.listdir(base):
        kind = _read(f"{base}/{name}/type")
        if kind in ("Mains", "USB", "USB_C", "USB_PD"):
            on_ac = on_ac or _read(f"{base}/{name}/online") == "1"
        elif kind == "Battery":
            battery = _read(f"{base}/{name}/capacity")
    text = "AC power" if on_ac else "battery"
    return text + (f" (battery {battery}%)" if battery else "")


def power_profile():
    parts = []
    for label, path in (
        ("platform_profile", "/sys/firmware/acpi/platform_profile"),
        ("governor", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        ("epp", "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
    ):
        value = _read(path)
        if value:
            parts.append(f"{label}={value}")
    return ", ".join(parts) or None


def cpu_temp_c():
    base = "/sys/class/hwmon"
    if not os.path.isdir(base):
        return None
    for name in sorted(os.listdir(base)):
        if _read(f"{base}/{name}/name") in ("k10temp", "zenpower", "coretemp", "cpu_thermal"):
            raw = _read(f"{base}/{name}/temp1_input")
            if raw and raw.lstrip("-").isdigit():
                return int(raw) / 1000
    return None


def gil_enabled():
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def system_info(quick):
    ram = total_ram_gib()
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "gil": "enabled" if gil_enabled() else "disabled (free-threaded build)",
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical_cores(),
        "ram_gib": round(ram, 1) if ram else None,
        "power": power_source(),
        "power_profile": power_profile(),
        "cpu_temp_start_c": cpu_temp_c(),
        "mode": "quick" if quick else "full",
    }


# ---------------------------------------------------------------------------
# Timing and output helpers
# ---------------------------------------------------------------------------

def run_timed(fn, repeat):
    """Call fn once to warm up, then `repeat` timed calls. Returns the times."""
    fn()
    times = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:7.1f} ms"
    return f"{seconds:7.2f} s "


def fmt_rate(per_sec, unit):
    for factor, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if per_sec >= factor:
            return f"{per_sec / factor:8.2f} {prefix}{unit}/s"
    return f"{per_sec:8.2f} {unit}/s"


TABLE_HEAD = f"  {'test':<26}{'median':>10}  {'best':>10}  {'spread':>6}  throughput"


def report(results, section, name, times, work, unit, note=""):
    med, best = statistics.median(times), min(times)
    rate = work / med
    results["tests"][f"{section}/{name}"] = {
        "median_s": med, "best_s": best, "runs": len(times), "rate": rate, "unit": unit,
    }
    spread = f"{(max(times) - best) / med:6.1%}" if len(times) > 1 else f"{'-':>6}"
    print(f"  {name:<26}{fmt_time(med)}  {fmt_time(best)}  {spread}  {fmt_rate(rate, unit)}{note}",
          flush=True)


# ---------------------------------------------------------------------------
# Single-core workloads. Each setup function prepares data (not timed) and
# returns (callable to time, units of work per call, unit label).
# ---------------------------------------------------------------------------

def mandelbrot(size, max_iter=50):
    inside = 0
    for y in range(size):
        ci = 2.0 * y / size - 1.0
        for x in range(size):
            cr = 3.0 * x / size - 2.0
            zr = zi = 0.0
            for _ in range(max_iter):
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                if zr * zr + zi * zi > 4.0:
                    break
            else:
                inside += 1
    return inside


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def fib_calls(n):
    """Number of calls fib(n) makes: 2 * F(n+1) - 1."""
    a, b = 0, 1
    for _ in range(n + 1):
        a, b = b, a + b
    return 2 * a - 1


class Vec:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y


def _arccot(x, unity):
    total = power = unity // x
    x2, n, sign = x * x, 3, -1
    while True:
        power //= x2
        term = power // n
        if not term:
            return total
        total += sign * term
        sign, n = -sign, n + 2


def s_int_loop(n):
    def run():
        total = 0
        for i in range(n):
            total += i * i % 7
        return total
    return run, n, "iter"


def s_mandelbrot(size):
    return (lambda: mandelbrot(size)), size * size, "px"


def s_nbody(steps):
    days = 365.24
    solar_mass = 4 * math.pi ** 2
    bodies = [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], solar_mass),
        ([4.84143144246472090, -1.16032004402742839, -0.103622044471123109],
         [0.00166007664274403694 * days, 0.00769901118419740425 * days,
          -0.0000690460016972063023 * days],
         0.000954791938424326609 * solar_mass),
        ([8.34336671824457987, 4.12479856412430479, -0.403523417114321381],
         [-0.00276742510726862411 * days, 0.00499852801234917238 * days,
          0.0000230417297573763929 * days],
         0.000285885980666130812 * solar_mass),
        ([12.8943695621391310, -15.1111514016986312, -0.223307578892655734],
         [0.00296460137564761618 * days, 0.00237847173959480950 * days,
          -0.0000296589568540237556 * days],
         0.0000436624404335156298 * solar_mass),
        ([15.3796971148509165, -25.9193146099879641, 0.179258772950371181],
         [0.00268067772490389322 * days, 0.00162824170038242295 * days,
          -0.0000951592254519715870 * days],
         0.0000515138902046611451 * solar_mass),
    ]

    def run():
        state = [(list(p), list(v), m) for p, v, m in bodies]
        pairs = [(state[i], state[j]) for i in range(5) for j in range(i + 1, 5)]
        dt = 0.01
        for _ in range(steps):
            for (p1, v1, m1), (p2, v2, m2) in pairs:
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dz = p1[2] - p2[2]
                mag = dt * (dx * dx + dy * dy + dz * dz) ** -1.5
                b1, b2 = m1 * mag, m2 * mag
                v1[0] -= dx * b2
                v1[1] -= dy * b2
                v1[2] -= dz * b2
                v2[0] += dx * b1
                v2[1] += dy * b1
                v2[2] += dz * b1
            for p, v, _m in state:
                p[0] += dt * v[0]
                p[1] += dt * v[1]
                p[2] += dt * v[2]
        return state[0][0][0]
    return run, steps, "step"


def s_fib(n):
    return (lambda: fib(n)), fib_calls(n), "call"


def s_objects(n):
    def run():
        acc = Vec(0.0, 0.0)
        total = 0.0
        for i in range(n):
            v = Vec(i, 1.0)
            acc = acc + v
            total += acc.dot(v)
        return total
    return run, n, "iter"


def s_comprehension(n):
    def run():
        squares = [i * i for i in range(n)]
        evens = {x for x in squares if x % 2 == 0}
        total = sum(x for x in squares if x % 3 == 0)
        return len(evens) + total
    return run, n, "item"


WORDS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")


def s_strings(n):
    def run():
        parts = []
        for i in range(n):
            w = WORDS[i % 8]
            parts.append(f"{w.upper()}-{i}:{w[::-1]}")
        text = " ".join(parts)
        return len(text.replace("-", "_").lower().split())
    return run, n, "item"


def s_dict(n):
    keys = [f"key{i}" for i in range(n)]

    def run():
        d = {}
        for i, k in enumerate(keys):
            d[k] = i
        total = 0
        for k in keys:
            total += d[k]
        return total
    return run, 2 * n, "op"


def s_sort(n):
    rng = random.Random(1)
    data = [rng.random() for _ in range(n)]
    return (lambda: sorted(data)), n, "item"


def s_regex(lines):
    rng = random.Random(2)
    text = "\n".join(
        f"user{rng.randint(1, 99999)}@mail{rng.randint(1, 99)}.com GET /page/{rng.randint(1, 9999)} "
        f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} status={rng.choice((200, 301, 404, 500))}"
        for _ in range(lines)
    )
    email = re.compile(r"[\w.]+@[\w.]+\.com")
    clock = re.compile(r"\b(\d\d):(\d\d)\b")
    errors = re.compile(r"status=5\d\d")

    def run():
        return len(email.findall(text)) + len(clock.findall(text)) + len(errors.findall(text))
    return run, len(text), "B"


def s_json(n):
    rng = random.Random(3)
    records = [
        {"id": i, "name": f"item-{i}", "price": round(rng.random() * 100, 2),
         "tags": ["red", "green", "blue"][: i % 4], "active": i % 2 == 0,
         "meta": {"rank": rng.randint(1, 1000), "note": None}}
        for i in range(n)
    ]
    size = len(json.dumps(records))
    return (lambda: json.loads(json.dumps(records))), size, "B"


def s_pidigits(digits):
    def run():
        unity = 10 ** (digits + 10)
        pi = 4 * (4 * _arccot(5, unity) - _arccot(239, unity))
        assert pi // 10 ** (digits + 5) == 314159, "pi digits are wrong"
        return pi
    return run, digits, "digit"


def s_sha256(mb):
    data = os.urandom(mb * 2**20)
    return (lambda: hashlib.sha256(data).digest()), len(data), "B"


def s_zlib(mb):
    rng = random.Random(4)
    words = "lorem ipsum dolor sit amet python laptop benchmark speed memory disk core".split()
    chunk = " ".join(f"{rng.choice(words)}{rng.randint(0, 999)}" for _ in range(100_000)).encode()
    size = mb * 2**20
    data = (chunk * (size // len(chunk) + 1))[:size]
    return (lambda: zlib.decompress(zlib.compress(data, 6))), size, "B"


# name, setup function, size for a full run, size for --quick
SINGLE_TESTS = [
    ("integer loop", s_int_loop, 5_000_000, 1_000_000),
    ("float: mandelbrot", s_mandelbrot, 250, 120),
    ("float: n-body", s_nbody, 50_000, 10_000),
    ("recursion: fib", s_fib, 30, 26),
    ("objects + methods", s_objects, 1_000_000, 200_000),
    ("comprehensions", s_comprehension, 2_000_000, 400_000),
    ("string building", s_strings, 500_000, 100_000),
    ("dict insert + lookup", s_dict, 1_000_000, 200_000),
    ("sort floats", s_sort, 1_000_000, 200_000),
    ("regex scan", s_regex, 200_000, 40_000),
    ("json dumps + loads", s_json, 100_000, 20_000),
    ("big int: pi digits", s_pidigits, 20_000, 8_000),
    ("sha256 (C code)", s_sha256, 256, 64),
    ("zlib round trip (C code)", s_zlib, 32, 8),
]


def single_section(results, repeat, quick):
    print(TABLE_HEAD)
    for name, setup, full, small in SINGLE_TESTS:
        fn, work, unit = setup(small if quick else full)
        report(results, "single", name, run_timed(fn, repeat), work, unit)


# ---------------------------------------------------------------------------
# Multi-core scaling. Worker functions must be top-level so child processes
# can import them.
# ---------------------------------------------------------------------------

def _noop(_):
    return None


def _cpu_chunk(size):
    return mandelbrot(size)


def _multi_row(results, label, t, base, workers, tasks):
    speedup = base / t
    results["tests"][f"multi/{label}"] = {
        "median_s": t, "best_s": t, "runs": 1, "rate": tasks / t, "unit": "task",
        "workers": workers, "speedup": speedup,
    }
    print(f"  {label:<26}{fmt_time(t)}  {speedup:7.2f}x  {speedup / workers:10.0%}", flush=True)


def multi_section(results, quick):
    logical = os.cpu_count() or 1
    phys = physical_cores() or logical
    counts = sorted({c for c in (1, 2, 4, 8, phys, logical) if c <= logical})
    tasks, size = (48, 60) if quick else (96, 100)
    print(f"  {tasks} equal CPU-bound tasks (pure Python); same total work for every row")
    print(f"  {'workers':<26}{'time':>10}  {'speedup':>8}  {'efficiency':>10}")

    base = None
    ctx = multiprocessing.get_context()
    for w in counts:
        with ctx.Pool(processes=w) as pool:
            pool.map(_noop, range(w * 4), chunksize=1)  # make sure all workers are up
            t0 = time.perf_counter()
            pool.map(_cpu_chunk, [size] * tasks, chunksize=1)
            t = time.perf_counter() - t0
        base = base or t
        _multi_row(results, f"processes x{w}", t, base, w, tasks)

    with ThreadPoolExecutor(max_workers=logical) as ex:
        list(ex.map(_noop, range(logical * 4)))
        t0 = time.perf_counter()
        list(ex.map(_cpu_chunk, [size] * tasks))
        t = time.perf_counter() - t0
    _multi_row(results, f"threads x{logical}", t, base, logical, tasks)
    if gil_enabled():
        print("  (threads run Python code one at a time under the GIL, so ~1x is expected)")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def memory_section(results, repeat, quick):
    print(TABLE_HEAD)
    n = (128 if quick else 512) * 2**20
    src = bytearray(b"\xa5") * n
    dst = bytearray(n)

    def copy():
        dst[:] = src
    report(results, "memory", "bulk copy (memcpy)", run_timed(copy, repeat), n, "B")
    report(results, "memory", "bulk scan (memchr)",
           run_timed(lambda: src.find(b"\x00"), repeat), n, "B")

    count = 500_000 if quick else 2_000_000
    report(results, "memory", "allocate tuples + str",
           run_timed(lambda: [(i, str(i)) for i in range(count)], repeat), count, "obj")
    big = list(range(count * 5))
    report(results, "memory", "sum list of ints", run_timed(lambda: sum(big), repeat),
           len(big), "item")


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def _write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _drop_cache(fd):
    """Evict the file from the OS page cache so reads hit the disk (Linux/BSD)."""
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _pread(fd, n, offset):
    if hasattr(os, "pread"):
        return os.pread(fd, n, offset)
    os.lseek(fd, offset, os.SEEK_SET)
    return os.read(fd, n)


def disk_section(results, directory, size_mb, quick):
    block = os.urandom(4 * 2**20)  # random data, so filesystem compression can't help
    blocks = max(1, size_mb // 4)
    size = blocks * len(block)
    if shutil.disk_usage(directory).free < 2 * size:
        print(f"  skipped: needs {2 * size // 2**20} MB free in {directory}")
        return
    print(f"  {size // 2**20} MB temporary file in {directory}")
    if not hasattr(os, "posix_fadvise"):
        print("  note: can't drop the OS cache on this platform; reads may show RAM speed")
    print(TABLE_HEAD)

    fd, path = tempfile.mkstemp(prefix="pybench_", dir=directory)
    try:
        t0 = time.perf_counter()
        for _ in range(blocks):
            _write_all(fd, block)
        os.fsync(fd)
        report(results, "disk", "sequential write + fsync", [time.perf_counter() - t0], size, "B")

        _drop_cache(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        t0 = time.perf_counter()
        while os.read(fd, len(block)):
            pass
        report(results, "disk", "sequential read", [time.perf_counter() - t0], size, "B")

        _drop_cache(fd)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # no read-ahead
        count = 1000 if quick else 5000
        rng = random.Random(5)
        offsets = [rng.randrange(size // 4096) * 4096 for _ in range(count)]
        t0 = time.perf_counter()
        for offset in offsets:
            _pread(fd, 4096, offset)
        t = time.perf_counter() - t0
        report(results, "disk", "random 4 KiB read", [t], count, "read",
               note=f"  (avg {t / count * 1e6:.0f} us each)")
    finally:
        os.close(fd)
        os.unlink(path)


# ---------------------------------------------------------------------------
# NumPy (optional)
# ---------------------------------------------------------------------------

def numpy_section(results, repeat, quick):
    try:
        import numpy as np
    except ImportError:
        print("  NumPy is not installed - skipped (pip install numpy to enable)")
        return
    print(f"  NumPy {np.__version__}; BLAS uses all cores")
    print(TABLE_HEAD)
    rng = np.random.default_rng(0)
    n = 1024 if quick else 2048
    for dtype in ("float64", "float32"):
        a = rng.random((n, n)).astype(dtype)
        b = rng.random((n, n)).astype(dtype)
        report(results, "numpy", f"matmul {n}x{n} {dtype}",
               run_timed(lambda: a @ b, repeat), 2 * n**3, "FLOP")


# ---------------------------------------------------------------------------
# Summary and comparison
# ---------------------------------------------------------------------------

def print_header(meta):
    print("Python benchmark")
    print("=" * 72)
    cores = f"{meta['physical_cores'] or '?'} physical / {meta['logical_cpus']} logical"
    temp = meta["cpu_temp_start_c"]
    rows = [
        ("Date", meta["date"]),
        ("CPU", meta["cpu"]),
        ("Cores", cores),
        ("RAM", f"{meta['ram_gib']} GiB" if meta["ram_gib"] else None),
        ("OS", meta["os"]),
        ("Python", f"{meta['python']}, GIL {meta['gil']}"),
        ("Power", meta["power"]),
        ("Profile", meta["power_profile"]),
        ("CPU temp", f"{temp:.0f} C" if temp is not None else None),
        ("Mode", meta["mode"]),
    ]
    for key, value in rows:
        if value:
            print(f"  {key:<10}{value}")


def print_summary(results):
    meta, tests = results["meta"], results["tests"]
    print("\n[Summary]")
    minutes, seconds = divmod(int(meta["elapsed_s"]), 60)
    print(f"  {'total time':<14}{minutes}m {seconds:02d}s")
    t0, t1 = meta["cpu_temp_start_c"], meta["cpu_temp_end_c"]
    if t0 is not None and t1 is not None:
        print(f"  {'CPU temp':<14}{t0:.0f} C at start -> {t1:.0f} C at end")
    procs = [v for k, v in tests.items() if k.startswith("multi/processes")]
    if procs:
        best = max(procs, key=lambda v: v["speedup"])
        print(f"  {'multi-core':<14}best speedup {best['speedup']:.2f}x with {best['workers']} "
              f"processes ({meta['physical_cores'] or '?'} physical cores)")
    singles = [v["rate"] for k, v in tests.items() if k.startswith("single/")]
    if singles:
        print(f"  {'single-core':<14}{len(singles)} tests; use --save / --compare to track "
              "changes between runs")


def compare(results, path):
    with open(path) as f:
        old = json.load(f)
    print(f"\n[Compared with {path}  ({old['meta'].get('date', '?')})]")
    if old["meta"].get("mode") != results["meta"]["mode"]:
        print(f"  warning: that run used mode '{old['meta'].get('mode')}', this one "
              f"'{results['meta']['mode']}'; workloads differ, so ratios are not comparable")
    for label in ("power", "power_profile", "python"):
        if old["meta"].get(label) != results["meta"].get(label):
            print(f"  note: {label} differs: {old['meta'].get(label)} -> {results['meta'].get(label)}")
    print(f"  {'test':<40}{'then':>10}  {'now':>10}   ratio")
    by_section = {}
    for key, new in results["tests"].items():
        prev = old.get("tests", {}).get(key)
        if not prev:
            continue
        ratio = prev["median_s"] / new["median_s"]  # > 1 means faster now
        verdict = "faster" if ratio > 1.03 else "slower" if ratio < 0.97 else "same"
        print(f"  {key:<40}{fmt_time(prev['median_s'])}  {fmt_time(new['median_s'])}"
              f"  {ratio:6.2f}x {verdict}")
        by_section.setdefault(key.split("/")[0], []).append(ratio)
    if by_section:
        print("  geometric mean (above 1.00x = faster now):")
        for section, ratios in by_section.items():
            print(f"    {section:<10}{statistics.geometric_mean(ratios):6.2f}x  ({len(ratios)} tests)")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Python benchmark for this computer.")
    parser.add_argument("--quick", action="store_true", help="smaller workloads, quick check")
    parser.add_argument("--repeat", type=int, help="timed runs per test (default 5, quick 3)")
    parser.add_argument("--only", help="comma-separated sections: " + ",".join(SECTIONS))
    parser.add_argument("--disk-dir", default=os.getcwd(),
                        help="where to put the temporary disk-test file (default: current dir)")
    parser.add_argument("--disk-mb", type=int,
                        help="disk test file size in MB (default 1024, quick 256)")
    parser.add_argument("--save", metavar="FILE", help="write results to FILE as JSON")
    parser.add_argument("--compare", metavar="FILE", help="compare with a run saved by --save")
    args = parser.parse_args()

    sections = [s.strip() for s in args.only.split(",")] if args.only else list(SECTIONS)
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        parser.error(f"unknown section(s): {', '.join(sorted(unknown))}")
    repeat = args.repeat or (3 if args.quick else 5)
    disk_mb = args.disk_mb or (256 if args.quick else 1024)

    results = {"meta": system_info(args.quick), "tests": {}}
    print_header(results["meta"])
    start = time.perf_counter()
    try:
        for section in sections:
            temp = cpu_temp_c()
            print(f"\n[{SECTION_TITLES[section]}]" + (f"   CPU {temp:.0f} C" if temp else ""))
            if section == "single":
                single_section(results, repeat, args.quick)
            elif section == "multi":
                multi_section(results, args.quick)
            elif section == "memory":
                memory_section(results, repeat, args.quick)
            elif section == "disk":
                disk_section(results, args.disk_dir, disk_mb, args.quick)
            elif section == "numpy":
                numpy_section(results, repeat, args.quick)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    results["meta"]["elapsed_s"] = time.perf_counter() - start
    results["meta"]["cpu_temp_end_c"] = cpu_temp_c()
    print_summary(results)
    if args.compare:
        compare(results, args.compare)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
