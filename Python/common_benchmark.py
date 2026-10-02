#!/usr/bin/env python3
"""
common_benchmark.py -- the shared C++ / Python / R benchmark, Python implementation.

Implements common/SPEC.md: the same tests, sizes (common/tests.csv) and output format as
Cpp/common_benchmark and R/common_benchmark.R, so the three can be compared in analysis.ipynb.
Needs NumPy (and pyopencl for the GPU section): run it with the benchmark's virtual environment.

Usage
  python3 common_benchmark.py                 full run
  python3 common_benchmark.py --quick         smaller sizes, quick check
  python3 common_benchmark.py --verify        tiny identical sizes, for checksums
  python3 common_benchmark.py --only cpu_single,mem_copy
  python3 common_benchmark.py --out DIR       results directory (default: results/<machine>)

Categories: cpu_single, cpu_multi, ram, gpu (see common/SPEC.md for every test).
Results: <out>/python_<timestamp>.csv (one row per timed repetition) and
<out>/python_<timestamp>_meta.csv (system information).
"""

import argparse
import csv
import gc
import math
import multiprocessing
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS_CSV = ROOT / "common" / "tests.csv"
KERNELS_CL = ROOT / "common" / "kernels.cl"
LANG_KEY = "python"
CATEGORIES = ("cpu_single", "cpu_multi", "ram", "gpu")
PHI = 0.6180339887498949
PHI2 = 0.7548776662466927
MIB, GIB = 2**20, 2**30
PEAK_ITERS = 1024
RESULT_COLUMNS = ["run_id", "batch", "language", "mode", "category", "test", "style", "threads",
                  "size", "rep", "seconds", "work", "unit", "rate", "check"]
META_KEYS = ["run_id", "batch", "language", "mode", "language_version", "build", "blas",
             "numpy_version", "cpu", "physical_cores", "logical_cpus", "ram_gib", "power",
             "platform_profile", "cpu_temp_start_c", "cpu_temp_end_c", "started", "elapsed_s",
             "host", "machine", "gpu_platform", "gpu_device", "gpu_compute_units", "gpu_max_clock_mhz",
             "gpu_driver", "skipped", "failed"]


# ---------------------------------------------------------------------------
# System information (Linux: /proc and /sys; Windows: registry and Win32 API through ctypes;
# macOS: sysctl, vm_stat, pmset and ioreg, plus Cpp/hw_probe for the temperature)
# ---------------------------------------------------------------------------

WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _reg(key, value):
    """A string value under HKEY_LOCAL_MACHINE (Windows), or None."""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
            return str(winreg.QueryValueEx(k, value)[0]).strip()
    except OSError:
        return None


def _run(*cmd):
    """Standard output of a command, or "" if it cannot run."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sysctl(name):
    return _run("/usr/sbin/sysctl", "-n", name)


def _ioreg(args, key):
    """A string property from `ioreg` output (macOS), e.g. "product-name" = <"MacBook Pro (14-inch, M5 Pro)">."""
    m = re.search(rf'"{re.escape(key)}" = <?"([^"]*)"', _run("/usr/sbin/ioreg", *args))
    return m.group(1).strip() if m else ""


def cpu_model():
    if MACOS:
        return _sysctl("machdep.cpu.brand_string") or platform.processor()
    if WINDOWS:
        return _reg(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString") or platform.processor()
    return (cpuinfo_field("model name") or [platform.processor()])[0]


def cpuinfo_field(key):
    return [line.split(":", 1)[1].strip() for line in (_read("/proc/cpuinfo") or "").splitlines()
            if line.split(":", 1)[0].strip() == key]


def physical_cores():
    if WINDOWS:
        import ctypes
        k32 = ctypes.windll.kernel32
        size = ctypes.c_ulong(0)
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))      # 0 = RelationProcessorCore
        buf = ctypes.create_string_buffer(size.value)
        if size.value and k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            cores, off = 0, 0
            while off < size.value:      # variable-size records: DWORD Relationship, DWORD Size, ...
                off += int.from_bytes(buf.raw[off + 4:off + 8], "little")
                cores += 1
            return cores
        return os.cpu_count()
    if MACOS:
        return int(_sysctl("hw.physicalcpu") or os.cpu_count())
    pairs = set(zip(cpuinfo_field("physical id"), cpuinfo_field("core id")))
    return len(pairs) or os.cpu_count()


def meminfo_kib(key):
    """MemTotal / MemAvailable in KiB (/proc/meminfo; Windows: GlobalMemoryStatusEx; macOS: hw.memsize, and
    free + speculative + inactive pages from vm_stat)."""
    if MACOS:
        if key == "MemTotal":
            return int(_sysctl("hw.memsize") or 0) // 1024
        text = _run("/usr/bin/vm_stat")
        page = int((re.search(r"page size of (\d+) bytes", text) or [0, 16384])[1])
        pages = sum(int(m) for m in re.findall(r"^Pages (?:free|speculative|inactive):\s+(\d+)", text, re.M))
        return pages * page // 1024
    if WINDOWS:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MemoryStatusEx()
        m.dwLength = ctypes.sizeof(m)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return {"MemTotal": m.ullTotalPhys, "MemAvailable": m.ullAvailPhys}[key] // 1024
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return None


def power_source():
    if MACOS:
        text = _run("/usr/bin/pmset", "-g", "batt")
        return "AC" if "'AC Power'" in text else "battery" if text else ""
    if WINDOWS:
        import ctypes

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        s = PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return ""
        return "AC" if s.ACLineStatus == 1 or s.BatteryFlag == 128 else "battery"   # 128 = no battery
    online = [_read(p) for p in Path("/sys/class/power_supply").glob("*/online")]
    return "AC" if "1" in online else "battery"


WINDOWS_POWER_MODES = {"961cc777-2547-4f9d-8174-7d86181b8a7a": "best power efficiency",
                       "00000000-0000-0000-0000-000000000000": "balanced",
                       "ded574b5-45a0-4f42-8737-46345c09c238": "best performance"}


MACOS_POWER_MODES = {"0": "automatic", "1": "low power", "2": "high power"}


def platform_profile():
    """ACPI platform profile (Linux), or the power mode for the current power source (Windows; macOS: pmset's
    powermode, else lowpowermode)."""
    if MACOS:
        text = _run("/usr/bin/pmset", "-g")
        mode = re.search(r"^\s*powermode\s+(\d)", text, re.M)
        if mode:
            return MACOS_POWER_MODES.get(mode.group(1), mode.group(1))
        low = re.search(r"^\s*lowpowermode\s+(\d)", text, re.M)
        return ("low power" if low.group(1) == "1" else "automatic") if low else ""
    if WINDOWS:
        value = "ActiveOverlayAcPowerScheme" if power_source() == "AC" else "ActiveOverlayDcPowerScheme"
        guid = (_reg(r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes", value) or "").lower()
        return WINDOWS_POWER_MODES.get(guid, guid)
    return _read("/sys/firmware/acpi/platform_profile")


CPU_SENSORS = ("k10temp", "zenpower", "coretemp", "cpu_thermal")   # in order of preference


def cpu_temp_c():
    """CPU temperature (Linux hwmon; macOS: the SoC die, read by Cpp/hw_probe --temp). Windows has no sensor
    readable without administrator rights."""
    if MACOS:
        probe = ROOT / "Cpp" / "hw_probe"
        text = _run(str(probe), "--temp") if probe.exists() else ""
        return float(text) if text else None
    hwmons = sorted(Path("/sys/class/hwmon").glob("hwmon*"))
    for want in CPU_SENSORS:
        for hwmon in hwmons:
            if _read(hwmon / "name") == want:
                raw = _read(hwmon / "temp1_input")
                if raw and raw.lstrip("-").isdigit():
                    return int(raw) / 1000
    return None


def slug(text):
    """lowercase; every run of characters outside [a-z0-9] becomes "-"; no "-" at either end."""
    out = []
    for ch in text.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def machine_id():
    """BENCH_MACHINE, else the firmware (DMI / SMBIOS) vendor + product name, else the hostname (SPEC.md).
    The vendor is left out when the product name already starts with it."""
    if os.environ.get("BENCH_MACHINE"):
        return os.environ["BENCH_MACHINE"]
    if WINDOWS:
        bios = r"HARDWARE\DESCRIPTION\System\BIOS"
        vendor, product = _reg(bios, "SystemManufacturer") or "", _reg(bios, "SystemProductName") or ""
    elif MACOS:
        vendor = _ioreg(["-rd1", "-c", "IOPlatformExpertDevice"], "manufacturer")
        product = _ioreg(["-p", "IODeviceTree", "-rd1", "-n", "product"], "product-name") or _sysctl("hw.model")
    else:
        vendor, product = _read("/sys/class/dmi/id/sys_vendor") or "", _read("/sys/class/dmi/id/product_name") or ""
    v, p = slug(vendor), slug(product)
    dmi = p if v and (p == v or p.startswith(v + "-")) else slug(f"{vendor} {product}")
    return dmi or slug(platform.node())


def tilde(value):
    """Replace the user's home directory with "~", so no personal paths end up in results."""
    text = str(value)
    homes = {os.path.expanduser("~"), os.environ.get("USERPROFILE", ""), os.environ.get("HOME", "")}
    for home in sorted({h for h in homes if h and h != "/"} |
                       {h.replace("\\", "/") for h in homes if h and h != "/"}, key=len, reverse=True):
        text = text.replace(home, "~")
    return text


def blas_info():
    try:
        deps = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        known = lambda v: v not in (None, "", "unknown")
        if not known(deps.get("openblas configuration")) and not known(deps.get("version")):
            return f"{deps.get('name')}"     # e.g. "accelerate" (Apple's Accelerate framework, macOS wheels)
        text = deps.get("openblas configuration") or f"{deps.get('name')} {deps.get('version')}"
        return f"{deps.get('name')}: {text}"
    except Exception as exc:  # noqa: BLE001 - informational only
        return f"unknown ({exc})"


def blas_threads():
    """Thread count of the OpenBLAS bundled with NumPy (falls back to the CPU count; on macOS NumPy's wheels use
    Apple's Accelerate, which reports no thread count)."""
    import ctypes
    if MACOS:
        return os.cpu_count()
    if WINDOWS:   # NumPy's wheels keep their DLLs in site-packages/numpy.libs
        paths = {str(p) for p in (Path(np.__file__).parent.parent / "numpy.libs").glob("*openblas*.dll")}
    else:
        paths = {line.split()[-1] for line in (_read("/proc/self/maps") or "").splitlines()
                 if "openblas" in line and line.split()[-1].startswith("/")}
    for path in paths:
        lib = ctypes.CDLL(path)
        for name in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_",
                     "openblas_get_num_threads"):
            fn = getattr(lib, name, None)
            if fn is not None:
                fn.restype = ctypes.c_int
                return fn()
    return os.cpu_count()


# ---------------------------------------------------------------------------
# Shared input data (SPEC §2)
# ---------------------------------------------------------------------------

def weyl(n, start=0, mult=PHI):
    i = (np.arange(n, dtype=np.int64) + start).astype(np.float64) * mult
    return i - np.floor(i)


# ---------------------------------------------------------------------------
# Workloads. Each setup returns (run, check, work); run() is timed, check(result) is not.
# Worker functions are top-level so multiprocessing (forkserver on Linux, spawn on Windows and macOS) can import them.
# ---------------------------------------------------------------------------

def mandel_rows(span):
    y0, y1, w = span
    total = 0
    for y in range(y0, y1):
        ci = -1.25 + (2.5 * y) / w
        for x in range(w):
            cr = -2.0 + (2.5 * x) / w
            zr = 0.0
            zi = 0.0
            it = 0
            while it < 100:
                t = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = t
                it += 1
                if zr * zr + zi * zi > 4.0:
                    break
            total += it
    return total


def _noop(_):
    return None


def fib(k):
    return k if k < 2 else fib(k - 1) + fib(k - 2)


def fib_calls(n):
    a, b = 0, 1  # a = F(0)
    for _ in range(n + 1):
        a, b = b, a + b
    return 2 * a - 1  # a = F(n+1)


def ident(x):
    return x


def setup_scalar_loop(n):
    def run():
        sqrt = math.sqrt
        s = 0.0
        for i in range(n):
            s += sqrt(i)
        return s
    return run, ident, n


def setup_mandelbrot(w):
    return (lambda: mandel_rows((0, w, w))), ident, w * w


def setup_fib(n):
    return (lambda: fib(n)), ident, fib_calls(n)


def setup_vector_math(n):
    x = 100.0 * weyl(n)
    return (lambda: float(np.sum(np.sqrt(x) * x + 1.0))), ident, n


def setup_sort(n):
    x = weyl(n)

    def check(y):
        return float(y[0] + y[n // 2] + y[n - 1])
    return (lambda: np.sort(x)), check, n


def setup_hashmap(n):
    keys = np.floor(weyl(n) * 2.0**31).astype(np.int64).tolist()

    def run():
        table = {}
        for i, k in enumerate(keys):
            table[k] = i
        total = 0
        for k in keys:
            total += table[k]
        return total
    return run, ident, 2 * n


def setup_string_ops(n):
    def run():
        items = [f"item{i}" for i in range(n)]
        joined = ",".join(items)
        pieces = joined.split(",")
        return sum(int(p[4:]) for p in pieces)
    return run, ident, n


def setup_matmul_blas(n):
    a = weyl(n * n).reshape(n, n)
    b = weyl(n * n, start=n * n).reshape(n, n)
    return (lambda: a @ b), (lambda c: float(c.sum())), 2 * n**3


def setup_mem_copy(size_mib):
    n = int(size_mib * MIB) // 8
    src = weyl(n)
    dst = np.zeros(n)

    def run():
        np.copyto(dst, src)
        return dst
    return run, (lambda d: float(d[n - 1])), 2 * int(size_mib * MIB)


def setup_mem_triad(size_mib):
    n = int(size_mib * MIB) // 8
    b = np.full(n, 1.0)
    c = np.full(n, 2.0)
    q = 3.0

    def run():
        a = b + q * c
        return a
    return run, (lambda a: float(a[n // 2])), 3 * int(size_mib * MIB)


def setup_mem_gather(size_mib):
    n = int(size_mib * MIB) // 8
    t = weyl(n)
    m = n // 4
    idx = np.floor(weyl(m, mult=PHI2) * n).astype(np.intp)
    return (lambda: float(t[idx].sum())), ident, m


CPU_SETUPS = {
    "scalar_loop": setup_scalar_loop,
    "mandelbrot": setup_mandelbrot,
    "fib_recursive": setup_fib,
    "vector_math": setup_vector_math,
    "sort": setup_sort,
    "hashmap": setup_hashmap,
    "string_ops": setup_string_ops,
    "matmul_blas": setup_matmul_blas,
    "mem_copy": setup_mem_copy,
    "mem_triad": setup_mem_triad,
    "mem_gather": setup_mem_gather,
}


# ---------------------------------------------------------------------------
# GPU (OpenCL through pyopencl)
# ---------------------------------------------------------------------------

class Gpu:
    def __init__(self):
        os.environ.setdefault("RUSTICL_ENABLE", "radeonsi,iris")  # let Mesa's rusticl expose AMD/Intel GPUs
        import pyopencl as cl
        self.cl = cl
        # The first GPU (over all platforms) whose name contains BENCH_GPU (case-insensitive),
        # or simply the first GPU when BENCH_GPU is unset.
        want = os.environ.get("BENCH_GPU", "").lower()
        device, seen = None, []
        for plat in cl.get_platforms():
            try:
                devices = plat.get_devices(device_type=cl.device_type.GPU)
            except cl.Error:
                continue
            for dev in devices:
                seen.append(dev.name.strip())
                if device is None and (not want or want in dev.name.lower()):
                    device = dev
        if device is None:
            if not seen:
                raise RuntimeError("no OpenCL GPU device found")
            raise RuntimeError(f"no OpenCL GPU matches BENCH_GPU={os.environ['BENCH_GPU']} "
                               f"(available: {' | '.join(seen)})")
        self.device = device
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, KERNELS_CL.read_text()).build()
        self.kernels = {name: cl.Kernel(self.prg, name) for name in ("fma_peak", "copy4", "sgemm4x4")}

    def info(self):
        d = self.device
        return {
            "gpu_platform": f"{d.platform.name} ({d.platform.version})",
            "gpu_device": d.name,
            "gpu_compute_units": d.max_compute_units,
            "gpu_max_clock_mhz": d.max_clock_frequency,
            "gpu_driver": d.driver_version,
        }

    def _buffer(self, nbytes, host=None):
        mf = self.cl.mem_flags
        if host is None:
            return self.cl.Buffer(self.ctx, mf.READ_WRITE, nbytes)
        return self.cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host)

    def _launch(self, name, global_size, *args):
        self.kernels[name](self.queue, (global_size,), None, *args)
        self.queue.finish()

    def _read(self, buf, count, first=0):
        """Read `count` floats starting at float index `first` (not timed)."""
        host = np.empty(count, dtype=np.float32)
        self.cl.enqueue_copy(self.queue, host, buf, src_offset=4 * first, is_blocking=True)
        return host

    def warmup(self, seconds):
        """Untimed, before the first GPU test: fma_peak back to back, so a GPU that idles at a low clock
        (NVIDIA laptop GPUs do) is at its working clock when timing starts (SPEC section 4, gpu)."""
        n = 1 << 20
        out = self._buffer(n * 4)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            self._launch("fma_peak", n, out, np.uint32(n))

    def setup_fp32_peak(self, n):
        n = int(n)
        out = self._buffer(n * 4)
        return ((lambda: self._launch("fma_peak", n, out, np.uint32(n))),
                (lambda _: float(self._read(out, 1)[0])), n * PEAK_ITERS * 32)

    def setup_bandwidth(self, size_mib):
        nbytes = int(size_mib * MIB)
        nf = nbytes // 4
        src = self._buffer(nbytes, np.full(nf, 1.5, dtype=np.float32))
        out = self._buffer(nbytes)
        return ((lambda: self._launch("copy4", nf // 4, out, np.uint32(nf), src)),
                (lambda _: float(self._read(out, 1, first=nf - 1)[0])), 2 * nbytes)

    def setup_sgemm(self, n):
        n = int(n)
        a = self._buffer(n * n * 4, weyl(n * n).astype(np.float32))
        b = self._buffer(n * n * 4, weyl(n * n, start=n * n).astype(np.float32))
        c = self._buffer(n * n * 4)
        return ((lambda: self._launch("sgemm4x4", (n // 4) ** 2, c, np.uint32(n * n), a, b, np.uint32(n))),
                (lambda _: float(self._read(c, n * n).sum(dtype=np.float64))), 2 * n**3)

    def setup_upload(self, size_mib):
        nbytes = int(size_mib * MIB)
        nf = nbytes // 4
        host = np.full(nf, 2.5, dtype=np.float32)
        buf = self._buffer(nbytes)

        def run():
            self.cl.enqueue_copy(self.queue, buf, host, is_blocking=True)
        return run, (lambda _: nf), nbytes

    def setup_download(self, size_mib):
        nbytes = int(size_mib * MIB)
        nf = nbytes // 4
        buf = self._buffer(nbytes, np.full(nf, 2.5, dtype=np.float32))
        host = np.empty(nf, dtype=np.float32)

        def run():
            self.cl.enqueue_copy(self.queue, host, buf, is_blocking=True)
            return host
        return run, (lambda h: float(h[0])), nbytes


GPU_SETUPS = {
    "gpu_fp32_peak": "setup_fp32_peak",
    "gpu_bandwidth": "setup_bandwidth",
    "gpu_sgemm": "setup_sgemm",
    "gpu_upload": "setup_upload",
    "gpu_download": "setup_download",
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def fmt_rate(per_sec, unit):
    for factor, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if per_sec >= factor:
            return f"{per_sec / factor:8.2f} {prefix}{unit}/s"
    return f"{per_sec:8.2f} {unit}/s"


def fmt_check(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.17g}"


def parse_size(text):
    value = float(text)
    return int(value) if value.is_integer() else value


class Runner:
    def __init__(self, args):
        self.mode = "verify" if args.verify else "quick" if args.quick else "full"
        self.started = datetime.now()
        stamp = self.started.strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{LANG_KEY}_{stamp}"
        self.batch = os.environ.get("BENCH_BATCH") or stamp
        self.machine = machine_id()
        self.out_dir = Path(args.out).expanduser().resolve() if args.out else ROOT / "results" / self.machine
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / f"{self.run_id}.csv"
        self.meta_path = self.out_dir / f"{self.run_id}_meta.csv"
        self.skipped, self.failed = [], []
        self.gpu, self.gpu_error, self.gpu_warm = None, None, False
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(RESULT_COLUMNS)

    def size_for(self, spec):
        column = "verify" if self.mode == "verify" else f"{self.mode}_{LANG_KEY}"
        return spec[column]

    def repeats_for(self, spec):
        repeats = int(spec["repeats"])
        return 1 if self.mode == "verify" else min(repeats, 3) if self.mode == "quick" else repeats

    def record(self, spec, threads, size, rep, seconds, work, check):
        self._writer.writerow([
            self.run_id, self.batch, "Python", self.mode, spec["category"], spec["test"],
            spec["style"], threads, size, rep, repr(seconds), work, spec["unit"],
            repr(work / seconds), fmt_check(check)])
        self._file.flush()

    def measure(self, spec, setup, threads, size, label=""):
        """Warm-up (if repeats > 1 and not verify), then timed repetitions; one row each."""
        run, check, work = setup
        repeats = self.repeats_for(spec)
        if repeats > 1 and self.mode != "verify":
            run()
        times = []
        for rep in range(1, repeats + 1):
            gc.collect()
            t0 = time.perf_counter()
            result = run()
            seconds = time.perf_counter() - t0
            times.append(seconds)
            self.record(spec, threads, size, rep, seconds, work, check(result))
        med = statistics.median(times)
        name = spec["test"] + (f" {label}" if label else "")
        print(f"  {spec['category']:<11} {name:<26} size={str(size):<10} {med:9.4f} s  "
              f"{fmt_rate(work / med, spec['unit'])}", flush=True)

    # -- tests that need their own loop ---------------------------------------------------

    def run_parallel_mandelbrot(self, spec, size):
        w = int(size)
        tasks = min(96, w)
        chunks = [((k * w) // tasks, ((k + 1) * w) // tasks, w) for k in range(tasks)]
        logical = os.cpu_count() or 1
        counts = sorted({c for c in (1, 2, 4, physical_cores(), 8, logical) if 1 <= c <= logical})
        ctx = multiprocessing.get_context()
        for workers in counts:
            with ctx.Pool(processes=workers) as pool:
                pool.map(_noop, range(workers * 4), chunksize=1)  # all workers up and imported
                setup = ((lambda: sum(pool.map(mandel_rows, chunks, chunksize=1))), ident, w * w)
                self.measure(spec, setup, workers, size, label=f"x{workers}")

    def run_mem_alloc(self, spec, size_text):
        avail = (meminfo_kib("MemAvailable") or 0) * 1024
        for g_text in str(size_text).split(";"):
            g = parse_size(g_text)
            nbytes = int(g * GIB)
            if nbytes > 0.4 * avail:
                self.skipped.append(f"mem_alloc {g} GiB: larger than 40% of MemAvailable "
                                    f"({avail / GIB:.1f} GiB)")
                print(f"  {'ram':<11} {'mem_alloc':<26} size={g!s:<10} skipped (too big)")
                continue
            n = nbytes // 8

            def run(n=n):
                x = np.empty(n)
                x.fill(1.0)
                s = x.sum()
                del x
                return float(s)
            self.measure(spec, (run, ident, 2 * nbytes), 1, g, label=f"{g} GiB")

    def init_gpu(self):
        if self.gpu or self.gpu_error:
            return
        try:
            self.gpu = Gpu()
        except Exception as exc:  # noqa: BLE001 - any OpenCL problem means "skip GPU"
            self.gpu_error = f"{type(exc).__name__}: {exc}".strip().replace("\n", " ")

    def run_test(self, spec):
        test, size_text = spec["test"], self.size_for(spec)
        if test == "parallel_mandelbrot":
            return self.run_parallel_mandelbrot(spec, parse_size(size_text))
        if test == "mem_alloc":
            return self.run_mem_alloc(spec, size_text)
        size = parse_size(size_text)
        if test in GPU_SETUPS:
            self.init_gpu()
            if self.gpu is None:
                self.skipped.append(f"{test}: {self.gpu_error}")
                print(f"  {spec['category']:<11} {test:<26} skipped ({self.gpu_error})")
                return
            if not self.gpu_warm and self.mode != "verify":
                print(f"  {'gpu':<11} {'warm-up':<26} 2 s of fma_peak, not timed", flush=True)
                self.gpu.warmup(2.0)
            self.gpu_warm = True
            setup = getattr(self.gpu, GPU_SETUPS[test])(size)
            return self.measure(spec, setup, 0, size)
        threads = blas_threads() if test == "matmul_blas" else 1
        return self.measure(spec, CPU_SETUPS[test](size), threads, size)

    def write_meta(self, elapsed):
        temp_end = cpu_temp_c()
        mem_kib = meminfo_kib("MemTotal")
        meta = {
            "run_id": self.run_id, "batch": self.batch, "language": "Python", "mode": self.mode,
            "language_version": f"{platform.python_implementation()} {platform.python_version()}",
            "build": sys.executable, "blas": blas_info(), "numpy_version": np.__version__,
            "cpu": cpu_model(),
            "physical_cores": physical_cores(), "logical_cpus": os.cpu_count(),
            "ram_gib": round(mem_kib / 2**20, 1) if mem_kib else "",
            "power": power_source(), "platform_profile": platform_profile(),
            "cpu_temp_start_c": self.temp_start, "cpu_temp_end_c": temp_end,
            "started": self.started.isoformat(timespec="seconds"), "elapsed_s": round(elapsed, 3),
            "host": self.machine, "machine": self.machine,
            "skipped": "; ".join(self.skipped), "failed": "; ".join(self.failed),
        }
        if self.gpu:
            meta.update(self.gpu.info())
        with open(self.meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for key in META_KEYS:
                value = meta.get(key)
                writer.writerow([key, "" if value is None else tilde(value)])

    def run(self, specs):
        self.temp_start = cpu_temp_c()
        if any(s["category"] == "gpu" for s in specs):
            self.init_gpu()
        gpu_text = self.gpu.device.name if self.gpu else f"none ({self.gpu_error})" if self.gpu_error else "-"
        print("Common benchmark - Python")
        print("=" * 72)
        for key, value in (
            ("Python", f"{platform.python_implementation()} {platform.python_version()} ({tilde(sys.executable)})"),
            ("Machine", self.machine),
            ("NumPy", f"{np.__version__}; {blas_info()}"),
            ("CPU", f"{cpu_model() or '?'}, "
                    f"{physical_cores()} physical / {os.cpu_count()} logical"),
            ("GPU", gpu_text),
            ("Power", f"{power_source()}, profile {platform_profile() or '-'}"),
            ("CPU temp", f"{self.temp_start:.0f} C" if self.temp_start is not None else "?"),
            ("Mode", self.mode),
            ("Output", tilde(self.csv_path)),
        ):
            print(f"  {key:<9} {value}")
        print()
        start = time.perf_counter()
        try:
            for spec in specs:
                try:
                    self.run_test(spec)
                except Exception as exc:  # noqa: BLE001 - one failing test must not stop the run
                    msg = f"{type(exc).__name__}: {exc}".replace("\n", " ")
                    self.failed.append(f"{spec['test']}: {msg}")
                    print(f"  {spec['category']:<11} {spec['test']:<26} FAILED: {msg}", flush=True)
        finally:
            self._file.close()
            elapsed = time.perf_counter() - start
            self.write_meta(elapsed)
        minutes, seconds = divmod(int(elapsed), 60)
        temp_end = cpu_temp_c()
        print(f"\nDone in {minutes}m {seconds:02d}s"
              + (f"; CPU temp {self.temp_start:.0f} C -> {temp_end:.0f} C" if temp_end is not None
                 and self.temp_start is not None else ""))
        print(f"Results: {tilde(self.csv_path)}\nMeta:    {tilde(self.meta_path)}")


def load_tests():
    with open(TESTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(
        description="Common C++/Python/R benchmark - Python implementation (common/SPEC.md).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="smaller sizes, quick check")
    group.add_argument("--verify", action="store_true",
                       help="tiny sizes identical in all languages, for comparing checksums")
    parser.add_argument("--only", help="comma-separated categories and/or test names")
    parser.add_argument("--out", help="results directory (default: <benchmark root>/results/<machine>)")
    args = parser.parse_args()

    specs = load_tests()
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        known = set(CATEGORIES) | {s["test"] for s in specs}
        unknown = [w for w in wanted if w not in known]
        if unknown:
            parser.error(f"unknown test or category: {', '.join(unknown)}")
        specs = [s for s in specs if s["category"] in wanted or s["test"] in wanted]

    try:
        Runner(args).run(specs)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
