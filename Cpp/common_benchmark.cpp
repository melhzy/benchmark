// common_benchmark.cpp -- C++ implementation of the common benchmark (see ../common/SPEC.md).
//
// Build   make                      (in this folder; see Makefile. Windows: MinGW-w64 g++ and make,
//                                    e.g. from Rtools; run_all.ps1 builds it)
// Usage   ./common_benchmark                    full run
//         ./common_benchmark --quick            smaller sizes, quick check
//         ./common_benchmark --verify           tiny identical sizes, for comparing checksums
//         ./common_benchmark --only cpu_single,mem_copy   chosen categories / tests
//         ./common_benchmark --out DIR          results directory (default <root>/results)
//
// Sections: cpu_single, cpu_multi, ram, gpu (OpenCL). Each run writes <run_id>.csv
// (one row per timed repetition) and <run_id>_meta.csv (system information).

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef NO_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#ifndef BENCH_BUILD
#define BENCH_BUILD "unknown"
#endif

// OpenBLAS (system library, located by the Makefile), declared here so no cblas.h is needed.
// Built with -DNO_BLAS when the Makefile finds no OpenBLAS; matmul_blas is then skipped.
#ifndef NO_BLAS
extern "C" {
void cblas_dgemm(int order, int transa, int transb, int m, int n, int k, double alpha,
                 const double* a, int lda, const double* b, int ldb, double beta,
                 double* c, int ldc);
char* openblas_get_config(void);
int openblas_get_num_threads(void);
}
constexpr int CBLAS_ROW_MAJOR = 101, CBLAS_NO_TRANS = 111;
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// Keep the optimizer from deleting or caching timed work.
template <class T> inline void keep(T const& v) { asm volatile("" : : "r,m"(v) : "memory"); }
template <class T> inline T opaque(T v) { asm volatile("" : "+r"(v)); return v; }

// Shared input data (SPEC §2): Weyl sequences, bit-identical in every language.
constexpr double PHI1 = 0.6180339887498949, PHI2 = 0.7548776662466927;
inline double frac(double x) { return x - std::floor(x); }
inline double w(size_t i) { return frac((double)i * PHI1); }
inline double w2(size_t i) { return frac((double)i * PHI2); }

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

static std::string lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) out.push_back(item);
    if (!s.empty() && s.back() == sep) out.push_back("");
    return out;
}

static std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string read_line1(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    if (f) std::getline(f, line);
    return trim(line);
}

static std::string num(double v, int prec = 17) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", prec, v);
    return buf;
}

static std::string fmt_size(double v) {
    char buf[64];
    if (v == std::floor(v)) std::snprintf(buf, sizeof buf, "%.0f", v);
    else std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

static std::string fmt_rate(double rate, const std::string& unit) {
    const char* prefix = "";
    double f = 1;
    if (rate >= 1e9) { prefix = "G"; f = 1e9; }
    else if (rate >= 1e6) { prefix = "M"; f = 1e6; }
    else if (rate >= 1e3) { prefix = "k"; f = 1e3; }
    bool joined = unit == "B" || unit == "FLOP";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%8.2f %s%s%s/s", rate / f, prefix, joined ? "" : " ", unit.c_str());
    return buf;
}

static std::string now_string(const char* fmt) {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof buf, fmt, std::localtime(&t));
    return buf;
}

static std::string csv_field(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) out += c == '"' ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------------------
// System information (Linux: /proc and /sys; Windows: registry and Win32 API)
// ---------------------------------------------------------------------------

#ifdef _WIN32

static std::string reg_string(const char* key, const char* value) {
    char buf[512];
    DWORD size = sizeof buf;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, key, value, RRF_RT_REG_SZ, nullptr, buf, &size) != ERROR_SUCCESS) return "";
    return trim(buf);
}

static std::string cpu_model() {
    return reg_string("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString");
}

static int logical_cpus() { return (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS); }

static int physical_cores() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (!len || !GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()), &len))
        return logical_cpus();
    int cores = 0;
    for (DWORD off = 0; off < len; ++cores)
        off += reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off)->Size;
    return cores;
}

static double ram_total_kib() {
    MEMORYSTATUSEX m{sizeof m};
    return GlobalMemoryStatusEx(&m) ? m.ullTotalPhys / 1024.0 : 0;
}

static double ram_available_kib() {
    MEMORYSTATUSEX m{sizeof m};
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys / 1024.0 : 0;
}

static std::string power_source() {
    SYSTEM_POWER_STATUS s;
    if (!GetSystemPowerStatus(&s)) return "";
    return s.ACLineStatus == 1 || s.BatteryFlag == 128 ? "AC" : "battery";   // 128 = no battery
}

// The Windows "power mode" (Settings > System > Power) for the current power source.
static std::string platform_profile() {
    const char* key = "SYSTEM\\CurrentControlSet\\Control\\Power\\User\\PowerSchemes";
    std::string guid = lower_ascii(reg_string(key, power_source() == "AC" ? "ActiveOverlayAcPowerScheme"
                                                                         : "ActiveOverlayDcPowerScheme"));
    if (guid == "961cc777-2547-4f9d-8174-7d86181b8a7a") return "best power efficiency";
    if (guid == "00000000-0000-0000-0000-000000000000") return "balanced";
    if (guid == "ded574b5-45a0-4f42-8737-46345c09c238") return "best performance";
    return guid;
}

// Windows has no CPU temperature sensor that programs can read without administrator rights.
static std::string cpu_temp_c() { return ""; }

static std::string hostname() {
    char buf[MAX_COMPUTERNAME_LENGTH + 1] = {0};
    DWORD n = sizeof buf;
    GetComputerNameA(buf, &n);
    return buf;
}

static std::string firmware_vendor() { return reg_string("HARDWARE\\DESCRIPTION\\System\\BIOS", "SystemManufacturer"); }
static std::string firmware_product() { return reg_string("HARDWARE\\DESCRIPTION\\System\\BIOS", "SystemProductName"); }

static fs::path self_exe() {
    std::wstring buf(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
    buf.resize(n);
    return fs::path(buf);
}

#else  // Linux

static std::string cpu_model() {
    std::istringstream f(read_text("/proc/cpuinfo"));
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return trim(line.substr(line.find(':') + 1));
    return "";
}

static int logical_cpus() { return (int)sysconf(_SC_NPROCESSORS_ONLN); }

static int physical_cores() {
    std::istringstream f(read_text("/proc/cpuinfo"));
    std::string line, phys = "0";
    std::set<std::pair<std::string, std::string>> cores;
    while (std::getline(f, line)) {
        size_t pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::string key = trim(line.substr(0, pos)), val = trim(line.substr(pos + 1));
        if (key == "physical id") phys = val;
        else if (key == "core id") cores.insert({phys, val});
    }
    return cores.empty() ? logical_cpus() : (int)cores.size();
}

static double meminfo_kib(const std::string& key) {
    std::istringstream f(read_text("/proc/meminfo"));
    std::string line;
    while (std::getline(f, line))
        if (line.rfind(key + ":", 0) == 0) return std::stod(line.substr(key.size() + 1));
    return 0;
}

static double ram_total_kib() { return meminfo_kib("MemTotal"); }
static double ram_available_kib() { return meminfo_kib("MemAvailable"); }

static std::string power_source() {
    std::error_code ec;
    for (auto& e : fs::directory_iterator("/sys/class/power_supply", ec))
        if (read_line1((e.path() / "online").string()) == "1") return "AC";
    return "battery";
}

static std::string platform_profile() { return read_line1("/sys/firmware/acpi/platform_profile"); }

// CPU temperature from the first hwmon sensor of a known CPU driver, in order of preference.
static std::string cpu_temp_c() {
    for (const char* want : {"k10temp", "zenpower", "coretemp", "cpu_thermal"}) {
        std::error_code ec;
        for (auto& e : fs::directory_iterator("/sys/class/hwmon", ec))
            if (read_line1((e.path() / "name").string()) == want) {
                std::string raw = read_line1((e.path() / "temp1_input").string());
                if (!raw.empty()) return num(std::stod(raw) / 1000, 4);
            }
    }
    return "";
}

static std::string hostname() {
    char buf[256] = {0};
    gethostname(buf, sizeof buf - 1);
    return buf;
}

static std::string firmware_vendor() { return read_line1("/sys/class/dmi/id/sys_vendor"); }
static std::string firmware_product() { return read_line1("/sys/class/dmi/id/product_name"); }
static fs::path self_exe() { return fs::canonical("/proc/self/exe"); }

#endif  // _WIN32

// lowercase; every run of characters outside [a-z0-9] becomes "-"; no "-" at either end.
static std::string slug(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        c = (unsigned char)std::tolower(c);
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) out += (char)c;
        else if (!out.empty() && out.back() != '-') out += '-';
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out;
}

// Machine id (SPEC.md): BENCH_MACHINE, else the firmware (DMI / SMBIOS) vendor + product name, else the
// hostname. The vendor is left out when the product name already starts with it.
static std::string machine_id() {
    const char* env = std::getenv("BENCH_MACHINE");
    if (env && *env) return env;
    const std::string vendor = trim(firmware_vendor()), product = trim(firmware_product());
    const std::string v = slug(vendor), p = slug(product);
    std::string dmi = !v.empty() && (p == v || p.rfind(v + "-", 0) == 0) ? p : slug(vendor + " " + product);
    return dmi.empty() ? slug(hostname()) : dmi;
}

// Replaces the user's home directory with "~", so no personal paths end up in results.
// (Windows: %USERPROFILE%, written with \ or /; Git Bash and MSYS2 also set HOME.)
static std::string tilde(std::string s) {
    std::vector<std::string> homes;
    for (const char* var : {"HOME", "USERPROFILE"}) {
        const char* h = std::getenv(var);
        if (!h || !*h || std::string(h) == "/") continue;
        std::string back = h, fwd = h;
        std::replace(back.begin(), back.end(), '/', '\\');
        std::replace(fwd.begin(), fwd.end(), '\\', '/');
        homes.insert(homes.end(), {std::string(h), back, fwd});
    }
    for (const std::string& h : homes)
        for (size_t pos = 0; (pos = s.find(h, pos)) != std::string::npos;) {
            s.replace(pos, h.size(), "~");
            pos += 1;
        }
    return s;
}

// ---------------------------------------------------------------------------
// Test list (common/tests.csv) and result recording
// ---------------------------------------------------------------------------

struct TestDef {
    std::string category, test, style, unit, size;
    int repeats = 1;
};

static std::vector<TestDef> load_tests(const std::string& path, const std::string& mode) {
    std::istringstream f(read_text(path));
    if (!f.str().size()) throw std::runtime_error("cannot read " + path);
    std::string line;
    std::getline(f, line);
    std::vector<std::string> head = split(trim(line), ',');
    std::string size_col = mode == "verify" ? "verify" : mode + "_cpp";
    auto col = [&](const std::string& name) {
        auto it = std::find(head.begin(), head.end(), name);
        if (it == head.end()) throw std::runtime_error("tests.csv has no column " + name);
        return (size_t)(it - head.begin());
    };
    size_t c_cat = col("category"), c_test = col("test"), c_style = col("style"),
           c_unit = col("unit"), c_rep = col("repeats"), c_size = col(size_col);
    std::vector<TestDef> tests;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::vector<std::string> v = split(line, ',');
        tests.push_back({v[c_cat], v[c_test], v[c_style], v[c_unit], v[c_size], std::stoi(v[c_rep])});
    }
    return tests;
}

struct Row {
    std::string category, test, style, unit;
    int threads;
    double size;
    int rep;
    double seconds, work, check;
};

struct Bench {
    std::string mode;
    int phys = 1, logical = 1;
    std::vector<Row> rows;
    std::vector<std::string> skipped, failed;

    int repeats_for(const TestDef& t) const {
        return mode == "verify" ? 1 : mode == "quick" ? std::min(t.repeats, 3) : t.repeats;
    }

    // Times fn() repeatedly (after one untimed warm-up when repeats > 1, except in verify mode)
    // and records one row per repetition. fn returns the check value; if `post` is given it is
    // called once after timing to compute the check instead (so the check is not timed).
    void measure(const TestDef& t, double size, int threads, double work,
                 const std::function<double()>& fn,
                 const std::function<double()>& post = nullptr) {
        int reps = repeats_for(t);
        if (reps > 1 && mode != "verify") keep(fn());
        std::vector<double> secs;
        double check = 0;
        for (int r = 0; r < reps; ++r) {
            auto t0 = Clock::now();
            double c = fn();
            keep(c);
            auto t1 = Clock::now();
            secs.push_back(std::chrono::duration<double>(t1 - t0).count());
            check = c;
        }
        if (post) check = post();
        for (int r = 0; r < reps; ++r)
            rows.push_back({t.category, t.test, t.style, t.unit, threads, size, r + 1, secs[r], work, check});
        double med = median(secs);
        std::string label = t.test;
        if (t.style == "parallel" || t.style == "blas" || (t.test == "mem_triad" && threads != 1))
            label += " [" + std::to_string(threads) + " thr]";
        std::printf("  %-11s %-30s size=%-11s %9.4f s  %s\n", t.category.c_str(), label.c_str(),
                    fmt_size(size).c_str(), med, fmt_rate(work / med, t.unit).c_str());
        std::fflush(stdout);
    }

    std::vector<int> worker_counts() const {
        std::set<int> s;
        for (int c : {1, 2, 4, phys, 8, logical})
            if (c >= 1 && c <= logical) s.insert(c);
        return {s.begin(), s.end()};
    }
};

// ---------------------------------------------------------------------------
// cpu_single
// ---------------------------------------------------------------------------

static uint64_t mandel_rows(int W, int y0, int y1) {
    uint64_t total = 0;
    const double Wd = W;
    for (int y = y0; y < y1; ++y) {
        double ci = -1.25 + (2.5 * y) / Wd;
        for (int x = 0; x < W; ++x) {
            double cr = -2.0 + (2.5 * x) / Wd;
            double zr = 0.0, zi = 0.0;
            int it = 0;
            while (it < 100) {
                double t = zr * zr - zi * zi + cr;
                zi = 2.0 * zr * zi + ci;
                zr = t;
                ++it;
                if (zr * zr + zi * zi > 4.0) break;
            }
            total += it;
        }
    }
    return total;
}

static uint64_t fib(int n) { return n < 2 ? (uint64_t)n : fib(n - 1) + fib(n - 2); }

static double fib_calls(int n) {  // 2 * F(n+1) - 1
    double a = 0, b = 1;
    for (int i = 0; i < n + 1; ++i) { double t = a + b; a = b; b = t; }
    return 2 * a - 1;
}

static void t_scalar_loop(Bench& b, const TestDef& t) {
    size_t n = (size_t)std::stod(t.size);
    b.measure(t, (double)n, 1, (double)n, [&] {
        size_t nn = opaque(n);
        double s = 0.0;
        for (size_t i = 0; i < nn; ++i) s += std::sqrt((double)i);
        return s;
    });
}

static void t_mandelbrot(Bench& b, const TestDef& t) {
    int W = std::stoi(t.size);
    b.measure(t, W, 1, (double)W * W, [&] { return (double)mandel_rows(opaque(W), 0, W); });
}

static void t_fib(Bench& b, const TestDef& t) {
    int n = std::stoi(t.size);
    b.measure(t, n, 1, fib_calls(n), [&] { return (double)fib(opaque(n)); });
}

static void t_vector_math(Bench& b, const TestDef& t) {
    size_t n = (size_t)std::stod(t.size);
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = 100.0 * w(i);
    b.measure(t, (double)n, 1, (double)n, [&] {
        const double* p = opaque(x.data());
        double s = 0.0;
        for (size_t i = 0; i < n; ++i) s += std::sqrt(p[i]) * p[i] + 1.0;
        return s;
    });
}

static void t_sort(Bench& b, const TestDef& t) {
    size_t n = (size_t)std::stod(t.size);
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = w(i);
    b.measure(t, (double)n, 1, (double)n, [&] {
        std::vector<double> y(x);
        std::sort(y.begin(), y.end());
        return y[0] + y[n / 2] + y[n - 1];
    });
}

static void t_hashmap(Bench& b, const TestDef& t) {
    size_t n = (size_t)std::stod(t.size);
    std::vector<int64_t> k(n);
    for (size_t i = 0; i < n; ++i) k[i] = (int64_t)std::floor(w(i) * 2147483648.0);
    b.measure(t, (double)n, 1, 2.0 * n, [&] {
        std::unordered_map<int64_t, int64_t> m;
        for (size_t i = 0; i < n; ++i) m[k[i]] = (int64_t)i;
        int64_t total = 0;
        for (size_t i = 0; i < n; ++i) total += m.find(k[i])->second;
        return (double)total;
    });
}

static void t_string_ops(Bench& b, const TestDef& t) {
    size_t n = (size_t)std::stod(t.size);
    b.measure(t, (double)n, 1, (double)n, [&] {
        size_t nn = opaque(n);
        std::vector<std::string> items;
        items.reserve(nn);
        for (size_t i = 0; i < nn; ++i) items.push_back("item" + std::to_string(i));
        std::string joined;
        size_t len = 0;
        for (auto& s : items) len += s.size() + 1;
        joined.reserve(len);
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) joined += ',';
            joined += items[i];
        }
        std::vector<std::string> parts;
        size_t start = 0;
        for (;;) {
            size_t pos = joined.find(',', start);
            parts.emplace_back(joined, start, pos == std::string::npos ? std::string::npos : pos - start);
            if (pos == std::string::npos) break;
            start = pos + 1;
        }
        int64_t sum = 0;
        for (auto& p : parts) {
            int64_t v = 0;
            std::from_chars(p.data() + 4, p.data() + p.size(), v);
            sum += v;
        }
        return (double)sum;
    });
}

// ---------------------------------------------------------------------------
// cpu_multi
// ---------------------------------------------------------------------------

static uint64_t parallel_mandel(int W, int workers) {
    const int T = std::min(96, W);
    std::atomic<int> next{0};
    std::vector<uint64_t> part(workers, 0);
    std::vector<std::thread> pool;
    for (int k = 0; k < workers; ++k)
        pool.emplace_back([&, k] {
            uint64_t local = 0;
            for (;;) {
                int c = next.fetch_add(1);
                if (c >= T) break;
                int y0 = (int)((int64_t)c * W / T), y1 = (int)((int64_t)(c + 1) * W / T);
                local += mandel_rows(W, y0, y1);
            }
            part[k] = local;
        });
    for (auto& th : pool) th.join();
    return std::accumulate(part.begin(), part.end(), uint64_t(0));
}

static void t_parallel_mandelbrot(Bench& b, const TestDef& t) {
    int W = std::stoi(t.size);
    for (int workers : b.worker_counts())
        b.measure(t, W, workers, (double)W * W, [&] { return (double)parallel_mandel(W, workers); });
}

#ifndef NO_BLAS
static void t_matmul_blas(Bench& b, const TestDef& t) {
    int N = std::stoi(t.size);
    size_t nn = (size_t)N * N;
    std::vector<double> A(nn), B(nn), C(nn);
    for (size_t i = 0; i < nn; ++i) { A[i] = w(i); B[i] = w(nn + i); }
    b.measure(t, N, openblas_get_num_threads(), 2.0 * N * N * N,
        [&] {
            cblas_dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS, N, N, N, 1.0,
                        A.data(), N, B.data(), N, 0.0, C.data(), N);
            keep(C.data());
            return 0.0;
        },
        [&] { double s = 0; for (double v : C) s += v; return s; });
}
#endif

// ---------------------------------------------------------------------------
// ram
// ---------------------------------------------------------------------------

static size_t mib_doubles(double mib) { return (size_t)(mib * 1048576.0 / 8); }

static void t_mem_copy(Bench& b, const TestDef& t) {
    double S = std::stod(t.size);
    size_t n = mib_doubles(S);
    std::vector<double> src(n), dst(n, 0.0);
    for (size_t i = 0; i < n; ++i) src[i] = w(i);
    b.measure(t, S, 1, 2.0 * S * 1048576.0,
        [&] { std::memcpy(dst.data(), src.data(), n * sizeof(double)); keep(dst.data()); return 0.0; },
        [&] { return dst[n - 1]; });
}

static void triad_range(double* a, const double* bb, const double* c, double q, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) a[i] = bb[i] + q * c[i];
}

static void t_mem_triad(Bench& b, const TestDef& t) {
    double S = std::stod(t.size);
    size_t n = mib_doubles(S);
    std::vector<double> a(n, 0.0), bb(n, 1.0), c(n, 2.0);
    const double q = 3.0;
    double work = 3.0 * S * 1048576.0;
    std::set<int> counts = {1, b.phys, b.logical};
    for (int th : counts) {
        b.measure(t, S, th, work,
            [&] {
                double qq = opaque(q);
                if (th == 1) {
                    triad_range(a.data(), bb.data(), c.data(), qq, 0, n);
                } else {
                    std::vector<std::thread> pool;
                    for (int k = 0; k < th; ++k)
                        pool.emplace_back(triad_range, a.data(), bb.data(), c.data(), qq,
                                          n * k / th, n * (k + 1) / th);
                    for (auto& p : pool) p.join();
                }
                keep(a.data());
                return 0.0;
            },
            [&] { return a[n / 2]; });
    }
}

static void t_mem_gather(Bench& b, const TestDef& t) {
    double S = std::stod(t.size);
    size_t n = mib_doubles(S), m = n / 4;
    std::vector<double> tab(n);
    std::vector<int64_t> idx(m);
    for (size_t i = 0; i < n; ++i) tab[i] = w(i);
    for (size_t j = 0; j < m; ++j) idx[j] = (int64_t)std::floor(w2(j) * (double)n);
    b.measure(t, S, 1, (double)m, [&] {
        const double* p = opaque(tab.data());
        double s = 0.0;
        for (size_t j = 0; j < m; ++j) s += p[idx[j]];
        return s;
    });
}

static void t_mem_alloc(Bench& b, const TestDef& t) {
    double avail = ram_available_kib() * 1024.0;
    for (const std::string& g : split(t.size, ';')) {
        double G = std::stod(g), bytes = G * 1073741824.0;
        if (bytes > 0.4 * avail) {
            b.skipped.push_back("mem_alloc " + fmt_size(G) + " GiB: more than 40% of available RAM");
            std::printf("  %-11s %-30s size=%-11s skipped (more than 40%% of available RAM)\n",
                        t.category.c_str(), t.test.c_str(), fmt_size(G).c_str());
            continue;
        }
        size_t n = (size_t)(bytes / 8);
        b.measure(t, G, 1, 2.0 * bytes, [&] {
            double* p = new double[n];
            std::fill(p, p + n, 1.0);
            keep(p);
            double s = std::accumulate(p, p + n, 0.0);
            delete[] p;
            return s;
        });
    }
}

// ---------------------------------------------------------------------------
// gpu (OpenCL)
// ---------------------------------------------------------------------------

struct GpuInfo {
    std::string platform, device, driver, compute_units, max_clock_mhz;
};

#ifndef NO_OPENCL

#define CL_OK(expr)                                                                         \
    do {                                                                                    \
        cl_int e_ = (expr);                                                                 \
        if (e_ != CL_SUCCESS)                                                               \
            throw std::runtime_error(std::string("OpenCL error ") + std::to_string(e_) +    \
                                     " in " + #expr);                                       \
    } while (0)

struct ClMem {
    cl_mem m = nullptr;
    explicit ClMem(cl_mem x) : m(x) {}
    ~ClMem() { if (m) clReleaseMemObject(m); }
    ClMem(const ClMem&) = delete;
};

struct ClKernel {
    cl_kernel k = nullptr;
    explicit ClKernel(cl_kernel x) : k(x) {}
    ~ClKernel() { if (k) clReleaseKernel(k); }
    ClKernel(const ClKernel&) = delete;
};

struct Gpu {
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    GpuInfo info;

    ~Gpu() {
        if (prog) clReleaseProgram(prog);
        if (q) clReleaseCommandQueue(q);
        if (ctx) clReleaseContext(ctx);
    }

    cl_mem buffer(size_t bytes, const void* host = nullptr) {
        cl_int err;
        cl_mem_flags flags = CL_MEM_READ_WRITE | (host ? CL_MEM_COPY_HOST_PTR : 0);
        cl_mem m = clCreateBuffer(ctx, flags, bytes, const_cast<void*>(host), &err);
        CL_OK(err);
        return m;
    }
    cl_kernel kernel(const char* name) {
        cl_int err;
        cl_kernel k = clCreateKernel(prog, name, &err);
        CL_OK(err);
        return k;
    }
    void run(cl_kernel k, size_t global) {
        CL_OK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
        CL_OK(clFinish(q));
    }
    void read(cl_mem m, size_t offset, size_t bytes, void* dst) {
        CL_OK(clEnqueueReadBuffer(q, m, CL_TRUE, offset, bytes, dst, 0, nullptr, nullptr));
    }
    void write(cl_mem m, size_t bytes, const void* src) {
        CL_OK(clEnqueueWriteBuffer(q, m, CL_TRUE, 0, bytes, src, 0, nullptr, nullptr));
    }
};

static std::string platform_str(cl_platform_id p, cl_platform_info what) {
    size_t n = 0;
    if (clGetPlatformInfo(p, what, 0, nullptr, &n) != CL_SUCCESS) return "";
    std::string s(n, '\0');
    clGetPlatformInfo(p, what, n, s.data(), nullptr);
    return trim(s.c_str());
}

static std::string device_str(cl_device_id d, cl_device_info what) {
    size_t n = 0;
    if (clGetDeviceInfo(d, what, 0, nullptr, &n) != CL_SUCCESS) return "";
    std::string s(n, '\0');
    clGetDeviceInfo(d, what, n, s.data(), nullptr);
    return trim(s.c_str());
}

// Opens a GPU device and builds kernels.cl. Throws with the reason.
// Device: the first GPU (over all platforms) whose name contains BENCH_GPU (case-insensitive),
// or simply the first GPU when BENCH_GPU is unset.
static std::unique_ptr<Gpu> open_gpu(const std::string& kernels_path) {
    cl_uint np = 0;
    if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0)
        throw std::runtime_error("no OpenCL platform found");
    std::vector<cl_platform_id> plats(np);
    CL_OK(clGetPlatformIDs(np, plats.data(), nullptr));
    const char* want_env = std::getenv("BENCH_GPU");
    const std::string want = want_env ? lower_ascii(want_env) : "";
    auto g = std::make_unique<Gpu>();
    std::vector<std::string> seen;
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> devs(nd);
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr) != CL_SUCCESS) continue;
        for (cl_device_id d : devs) {
            std::string name = device_str(d, CL_DEVICE_NAME);
            seen.push_back(name);
            if (!g->dev && (want.empty() || lower_ascii(name).find(want) != std::string::npos)) {
                g->dev = d;
                g->info.platform = platform_str(p, CL_PLATFORM_NAME);
            }
        }
    }
    if (!g->dev) {
        if (seen.empty()) throw std::runtime_error("no OpenCL GPU device found");
        std::string list;
        for (size_t i = 0; i < seen.size(); ++i) list += (i ? " | " : "") + seen[i];
        throw std::runtime_error("no OpenCL GPU matches BENCH_GPU=" + std::string(want_env) +
                                 " (available: " + list + ")");
    }
    cl_uint cus = 0, mhz = 0;
    clGetDeviceInfo(g->dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cus, &cus, nullptr);
    clGetDeviceInfo(g->dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof mhz, &mhz, nullptr);
    g->info.device = device_str(g->dev, CL_DEVICE_NAME);
    g->info.driver = device_str(g->dev, CL_DRIVER_VERSION);
    g->info.compute_units = std::to_string(cus);
    g->info.max_clock_mhz = std::to_string(mhz);

    cl_int err;
    g->ctx = clCreateContext(nullptr, 1, &g->dev, nullptr, nullptr, &err);
    CL_OK(err);
    g->q = clCreateCommandQueue(g->ctx, g->dev, 0, &err);
    CL_OK(err);
    std::string src = read_text(kernels_path);
    if (src.empty()) throw std::runtime_error("cannot read " + kernels_path);
    const char* s = src.c_str();
    g->prog = clCreateProgramWithSource(g->ctx, 1, &s, nullptr, &err);
    CL_OK(err);
    if (clBuildProgram(g->prog, 1, &g->dev, "", nullptr, nullptr) != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(g->prog, g->dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(g->prog, g->dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        throw std::runtime_error("kernel build failed: " + trim(log.c_str()));
    }
    return g;
}

// Untimed, before the first GPU test: fma_peak back to back for `seconds`, so a GPU that idles at a low
// clock (NVIDIA laptop GPUs do) is at its working clock when timing starts (SPEC section 4, gpu).
static void gpu_warmup(Gpu& g, double seconds) {
    const size_t n = size_t(1) << 20;
    ClMem out(g.buffer(n * sizeof(float)));
    ClKernel k(g.kernel("fma_peak"));
    cl_uint nu = (cl_uint)n;
    CL_OK(clSetKernelArg(k.k, 0, sizeof(cl_mem), &out.m));
    CL_OK(clSetKernelArg(k.k, 1, sizeof(cl_uint), &nu));
    for (auto t0 = Clock::now(); std::chrono::duration<double>(Clock::now() - t0).count() < seconds;) g.run(k.k, n);
}

static void t_gpu_fp32_peak(Bench& b, Gpu& g, const TestDef& t) {
    size_t n = (size_t)std::stod(t.size);
    ClMem out(g.buffer(n * sizeof(float)));
    ClKernel k(g.kernel("fma_peak"));
    cl_uint nu = (cl_uint)n;
    CL_OK(clSetKernelArg(k.k, 0, sizeof(cl_mem), &out.m));
    CL_OK(clSetKernelArg(k.k, 1, sizeof(cl_uint), &nu));
    b.measure(t, (double)n, 0, (double)n * 1024 * 32,
        [&] { g.run(k.k, n); return 0.0; },
        [&] { float v; g.read(out.m, 0, sizeof v, &v); return (double)v; });
}

static void t_gpu_bandwidth(Bench& b, Gpu& g, const TestDef& t) {
    double S = std::stod(t.size);
    size_t n = (size_t)(S * 1048576.0 / sizeof(float));
    std::vector<float> host(n, 1.5f);
    ClMem in(g.buffer(n * sizeof(float), host.data()));
    ClMem out(g.buffer(n * sizeof(float)));
    ClKernel k(g.kernel("copy4"));
    cl_uint nu = (cl_uint)n;
    CL_OK(clSetKernelArg(k.k, 0, sizeof(cl_mem), &out.m));
    CL_OK(clSetKernelArg(k.k, 1, sizeof(cl_uint), &nu));
    CL_OK(clSetKernelArg(k.k, 2, sizeof(cl_mem), &in.m));
    b.measure(t, S, 0, 2.0 * S * 1048576.0,
        [&] { g.run(k.k, n / 4); return 0.0; },
        [&] { float v; g.read(out.m, (n - 1) * sizeof(float), sizeof v, &v); return (double)v; });
}

static void t_gpu_sgemm(Bench& b, Gpu& g, const TestDef& t) {
    int N = std::stoi(t.size);
    size_t nn = (size_t)N * N;
    std::vector<float> A(nn), B(nn);
    for (size_t i = 0; i < nn; ++i) { A[i] = (float)w(i); B[i] = (float)w(nn + i); }
    ClMem a(g.buffer(nn * sizeof(float), A.data())), bm(g.buffer(nn * sizeof(float), B.data()));
    ClMem c(g.buffer(nn * sizeof(float)));
    ClKernel k(g.kernel("sgemm4x4"));
    cl_uint count = (cl_uint)nn, nu = (cl_uint)N;
    CL_OK(clSetKernelArg(k.k, 0, sizeof(cl_mem), &c.m));
    CL_OK(clSetKernelArg(k.k, 1, sizeof(cl_uint), &count));
    CL_OK(clSetKernelArg(k.k, 2, sizeof(cl_mem), &a.m));
    CL_OK(clSetKernelArg(k.k, 3, sizeof(cl_mem), &bm.m));
    CL_OK(clSetKernelArg(k.k, 4, sizeof(cl_uint), &nu));
    b.measure(t, N, 0, 2.0 * N * N * N,
        [&] { g.run(k.k, (size_t)(N / 4) * (N / 4)); return 0.0; },
        [&] {
            std::vector<float> C(nn);
            g.read(c.m, 0, nn * sizeof(float), C.data());
            double s = 0;
            for (float v : C) s += v;
            return s;
        });
}

static void t_gpu_upload(Bench& b, Gpu& g, const TestDef& t) {
    double S = std::stod(t.size);
    size_t n = (size_t)(S * 1048576.0 / sizeof(float));
    std::vector<float> host(n, 2.5f);
    ClMem buf(g.buffer(n * sizeof(float)));
    b.measure(t, S, 0, S * 1048576.0, [&] {
        g.write(buf.m, n * sizeof(float), host.data());
        return (double)n;
    });
}

static void t_gpu_download(Bench& b, Gpu& g, const TestDef& t) {
    double S = std::stod(t.size);
    size_t n = (size_t)(S * 1048576.0 / sizeof(float));
    std::vector<float> src(n, 2.5f), dst(n);
    ClMem buf(g.buffer(n * sizeof(float), src.data()));
    b.measure(t, S, 0, S * 1048576.0, [&] {
        g.read(buf.m, 0, n * sizeof(float), dst.data());
        return (double)dst[0];
    });
}

#endif  // NO_OPENCL

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void usage() {
    std::puts("usage: common_benchmark [--quick | --verify] [--only LIST] [--out DIR]\n"
              "  --quick     smaller sizes (quick check)\n"
              "  --verify    tiny sizes identical in all languages, for comparing checksums\n"
              "  --only      comma-separated categories and/or test names\n"
              "  --out DIR   results directory (default: <benchmark root>/results/<machine>)\n"
              "environment: BENCH_MACHINE (machine id), BENCH_GPU (GPU name substring), BENCH_BATCH");
}

int main(int argc, char** argv) {
#ifndef _WIN32
    setenv("RUSTICL_ENABLE", "radeonsi,iris", 0);  // let Mesa's rusticl expose AMD and Intel GPUs
#endif

    std::string mode = "full", only, out;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quick") mode = "quick";
        else if (a == "--verify") mode = "verify";
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 2; }
    }

    fs::path root = self_exe().parent_path().parent_path();
    std::vector<TestDef> tests;
    try {
        tests = load_tests((root / "common" / "tests.csv").string(), mode);
    } catch (std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    if (!only.empty()) {
        std::set<std::string> want;
        for (auto& s : split(only, ',')) want.insert(trim(s));
        for (auto& s : want) {
            bool known = std::any_of(tests.begin(), tests.end(),
                [&](const TestDef& t) { return t.category == s || t.test == s; });
            if (!known) { std::fprintf(stderr, "unknown category or test: %s\n", s.c_str()); return 2; }
        }
        std::erase_if(tests, [&](const TestDef& t) { return !want.count(t.category) && !want.count(t.test); });
    }
    const std::string machine = machine_id();
    fs::path out_dir = out.empty() ? root / "results" / machine : fs::path(out);
    fs::create_directories(out_dir);

    const std::string stamp = now_string("%Y%m%d-%H%M%S");
    const std::string run_id = "cpp_" + stamp;
    const char* batch_env = std::getenv("BENCH_BATCH");
    const std::string batch = batch_env && *batch_env ? batch_env : stamp;
    const std::string started = now_string("%Y-%m-%dT%H:%M:%S");
    const std::string temp_start = cpu_temp_c();
    auto t_start = Clock::now();

    Bench b;
    b.mode = mode;
    b.logical = logical_cpus();
    b.phys = physical_cores();
#ifndef NO_BLAS
    const std::string blas = trim(openblas_get_config());
#else
    const std::string blas = "none";
    const std::string blas_skip = "OpenBLAS not found at build time (install it - README, Setup - or make OPENBLAS=...)";
#endif

    // GPU: opened up front (if any GPU test is selected) so the header can show the device.
    GpuInfo gpu_info;
    std::string gpu_skip;
    bool want_gpu = std::any_of(tests.begin(), tests.end(), [](const TestDef& t) { return t.category == "gpu"; });
#ifndef NO_OPENCL
    std::unique_ptr<Gpu> gpu;
    if (want_gpu) {
        try {
            gpu = open_gpu((root / "common" / "kernels.cl").string());
            gpu_info = gpu->info;
        } catch (std::exception& e) {
            gpu_skip = e.what();
        }
    }
#else
    gpu_skip = "built without OpenCL headers (install them - README, Setup - then run make)";
#endif

    std::printf("C++ common benchmark\n");
    std::printf("========================================================================\n");
    std::printf("  %-9s %s\n", "Language", "C++ (" __VERSION__ ")");
    std::printf("  %-9s %s\n", "Machine", machine.c_str());
    std::printf("  %-9s %s\n", "Build", BENCH_BUILD);
    std::printf("  %-9s %s (%d physical / %d logical)\n", "CPU", cpu_model().c_str(), b.phys, b.logical);
#ifndef NO_BLAS
    std::printf("  %-9s %s\n", "BLAS", blas.c_str());
#else
    std::printf("  %-9s none (%s)\n", "BLAS", blas_skip.c_str());
#endif
    if (want_gpu)
        std::printf("  %-9s %s\n", "GPU", gpu_skip.empty()
            ? (gpu_info.device + " (" + gpu_info.platform + ", " + gpu_info.compute_units + " CU, " +
               gpu_info.max_clock_mhz + " MHz)").c_str()
            : ("none: " + gpu_skip).c_str());
    std::printf("  %-9s %s\n", "Mode", mode.c_str());
    std::printf("  %-9s %s\n", "Output", tilde((out_dir / (run_id + ".csv")).string()).c_str());
    std::printf("\n");
    std::fflush(stdout);

    using Fn = std::function<void(Bench&, const TestDef&)>;
    std::map<std::string, Fn> cpu_tests = {
        {"scalar_loop", t_scalar_loop}, {"mandelbrot", t_mandelbrot}, {"fib_recursive", t_fib},
        {"vector_math", t_vector_math}, {"sort", t_sort}, {"hashmap", t_hashmap},
        {"string_ops", t_string_ops}, {"parallel_mandelbrot", t_parallel_mandelbrot},
#ifndef NO_BLAS
        {"matmul_blas", t_matmul_blas},
#endif
        {"mem_copy", t_mem_copy}, {"mem_triad", t_mem_triad},
        {"mem_gather", t_mem_gather}, {"mem_alloc", t_mem_alloc},
    };
#ifndef NO_OPENCL
    using GpuFn = std::function<void(Bench&, Gpu&, const TestDef&)>;
    std::map<std::string, GpuFn> gpu_tests = {
        {"gpu_fp32_peak", t_gpu_fp32_peak}, {"gpu_bandwidth", t_gpu_bandwidth},
        {"gpu_sgemm", t_gpu_sgemm}, {"gpu_upload", t_gpu_upload}, {"gpu_download", t_gpu_download},
    };
#endif

    bool gpu_warm = false;
    (void)gpu_warm;
    for (const TestDef& t : tests) {
        try {
            if (t.category == "gpu") {
                if (!gpu_skip.empty()) {
                    b.skipped.push_back(t.test + ": " + gpu_skip);
                    std::printf("  %-11s %-30s skipped (%s)\n", t.category.c_str(), t.test.c_str(), gpu_skip.c_str());
                    continue;
                }
#ifndef NO_OPENCL
                auto it = gpu_tests.find(t.test);
                if (it == gpu_tests.end()) throw std::runtime_error("not implemented");
                if (!gpu_warm && mode != "verify") {
                    std::printf("  %-11s %-30s %s\n", "gpu", "warm-up", "2 s of fma_peak, not timed");
                    std::fflush(stdout);
                    gpu_warmup(*gpu, 2.0);
                }
                gpu_warm = true;
                it->second(b, *gpu, t);
#endif
            } else {
#ifdef NO_BLAS
                if (t.test == "matmul_blas") {
                    b.skipped.push_back(t.test + ": " + blas_skip);
                    std::printf("  %-11s %-30s skipped (%s)\n", t.category.c_str(), t.test.c_str(), blas_skip.c_str());
                    continue;
                }
#endif
                auto it = cpu_tests.find(t.test);
                if (it == cpu_tests.end()) throw std::runtime_error("not implemented");
                it->second(b, t);
            }
        } catch (std::exception& e) {
            b.failed.push_back(t.test + ": " + e.what());
            std::printf("  %-11s %-30s FAILED: %s\n", t.category.c_str(), t.test.c_str(), e.what());
        }
        std::fflush(stdout);
    }

    double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
    const std::string temp_end = cpu_temp_c();

    // ---- results CSV
    fs::path res_path = out_dir / (run_id + ".csv");
    std::ofstream res(res_path);
    res << "run_id,batch,language,mode,category,test,style,threads,size,rep,seconds,work,unit,rate,check\n";
    for (const Row& r : b.rows)
        res << run_id << ',' << csv_field(batch) << ",C++," << mode << ',' << r.category << ',' << r.test
            << ',' << r.style << ',' << r.threads << ',' << num(r.size) << ',' << r.rep << ','
            << num(r.seconds, 9) << ',' << num(r.work) << ',' << r.unit << ',' << num(r.work / r.seconds, 9)
            << ',' << num(r.check) << '\n';
    res.close();

    // ---- meta CSV
    auto join = [](const std::vector<std::string>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) s += (i ? "; " : "") + v[i];
        return s;
    };
    std::vector<std::pair<std::string, std::string>> meta = {
        {"run_id", run_id},
        {"batch", batch},
        {"language", "C++"},
        {"mode", mode},
        {"language_version", "g++ " __VERSION__},
        {"build", BENCH_BUILD},
        {"blas", blas},
        {"numpy_version", ""},
        {"cpu", cpu_model()},
        {"physical_cores", std::to_string(b.phys)},
        {"logical_cpus", std::to_string(b.logical)},
        {"ram_gib", num(ram_total_kib() / 1048576.0, 4)},
        {"power", power_source()},
        {"platform_profile", platform_profile()},
        {"cpu_temp_start_c", temp_start},
        {"cpu_temp_end_c", temp_end},
        {"started", started},
        {"elapsed_s", num(elapsed, 6)},
        {"host", machine},
        {"machine", machine},
        {"gpu_platform", gpu_info.platform},
        {"gpu_device", gpu_info.device},
        {"gpu_compute_units", gpu_info.compute_units},
        {"gpu_max_clock_mhz", gpu_info.max_clock_mhz},
        {"gpu_driver", gpu_info.driver},
        {"skipped", join(b.skipped)},
        {"failed", join(b.failed)},
    };
    std::ofstream mf(out_dir / (run_id + "_meta.csv"));
    mf << "key,value\n";
    for (auto& [k, v] : meta) mf << k << ',' << csv_field(tilde(v)) << '\n';
    mf.close();

    const std::string temps = temp_start.empty() ? "" : " (CPU " + temp_start + " C -> " + temp_end + " C)";
    std::printf("\nFinished in %.1f s%s\nSaved %s\n", elapsed, temps.c_str(), tilde(res_path.string()).c_str());
    return 0;
}
