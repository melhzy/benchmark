// hw_probe.cpp -- hardware facts the operating system does not report directly, for the system snapshot that
// run_all.sh / run_all.ps1 write (system_<batch>.csv). Built by the Makefile next to common_benchmark.
//
// Prints key=value lines (keys are left out when unknown):
//   cpu_flags                               instruction-set extensions (x86): sse4_2 avx avx2 fma avx512f
//   performance_cores, efficiency_cores     hybrid CPUs only: physical cores of each type
//   cpu_max_mhz, cpu_e_max_mhz              clock measured on one busy core (of each type, on a hybrid CPU)
//   gpu_cuda_name, gpu_sm_count,            NVIDIA GPUs, from the CUDA driver (nvcuda.dll / libcuda.so.1, part of
//   gpu_boost_clock_mhz,                    the NVIDIA driver): rated boost clock, memory clock and bus width,
//   gpu_memory_clock_mhz,                   and the memory bandwidth they give (2 transfers per clock x bus
//   gpu_memory_bus_bits, gpu_memory_gbs     width; the formula of NVIDIA's deviceQuery sample)
//
// The clock is measured, not read: a chain of dependent register-to-register additions (one cycle each on every
// x86 and ARM core) runs on a thread pinned to the core; additions per second = cycles per second. (Adding an
// immediate would not work: recent Intel P-cores fold chains of those in the register renamer.)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
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
#include <dlfcn.h>
#include <sched.h>
#include <fstream>
#include <sstream>
#endif

// ---------------------------------------------------------------------------------------------- CPU clock

static double add_chain_ghz() {
    using clk = std::chrono::steady_clock;
    const int rounds = 20000;   // x 1000 additions = 2e7 cycles, about 4 ms at 5 GHz
    auto sample = [&] {
        uint64_t x = 0, y = 1;
        auto t0 = clk::now();
        for (int i = 0; i < rounds; ++i) {
#if defined(__x86_64__) || defined(__i386__)
            asm volatile(".rept 1000\n\tadd %1, %0\n\t.endr" : "+r"(x) : "r"(y));
#elif defined(__aarch64__)
            asm volatile(".rept 1000\n\tadd %0, %0, %1\n\t.endr" : "+r"(x) : "r"(y));
#else
            return 0.0;
#endif
        }
        return rounds * 1000.0 / std::chrono::duration<double>(clk::now() - t0).count() / 1e9;
    };
    // 300 ms of load first, so the core has left its idle clock; then the best of 25 samples.
    for (auto t0 = clk::now(); std::chrono::duration<double>(clk::now() - t0).count() < 0.3;) sample();
    double best = 0;
    for (int i = 0; i < 25; ++i) best = std::max(best, sample());
    return best;
}

struct CoreType {
    int cores = 0;
    std::string pin;   // how to pin the measuring thread to one logical CPU of this type (see pin_to)
};

#ifdef _WIN32

// Physical cores per efficiency class (0 = lowest); pin = "group:cpu" of the first logical CPU of the class.
static std::vector<CoreType> core_types() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    std::vector<CoreType> types;
    if (!len || !GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()), &len))
        return types;
    for (DWORD off = 0; off < len;) {
        auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        // PROCESSOR_RELATIONSHIP.EfficiencyClass: the byte after Flags (older MinGW headers call it Reserved[0])
        int cls = reinterpret_cast<const BYTE*>(&e->Processor)[1];
        if ((int)types.size() <= cls) types.resize(cls + 1);
        CoreType& t = types[cls];
        if (!t.cores++) {
            const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
            t.pin = std::to_string(g.Group) + ":" + std::to_string(__builtin_ctzll((unsigned long long)g.Mask));
        }
        off += e->Size;
    }
    types.erase(std::remove_if(types.begin(), types.end(), [](const CoreType& t) { return !t.cores; }), types.end());
    return types;
}

static bool pin_to(const std::string& pin) {
    GROUP_AFFINITY ga{};
    ga.Group = (WORD)std::stoi(pin.substr(0, pin.find(':')));
    ga.Mask = (KAFFINITY)1 << std::stoi(pin.substr(pin.find(':') + 1));
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr);
}

#else  // Linux

static std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

static std::vector<int> cpu_list(const std::string& text) {   // "0-15,24" -> 0..15, 24
    std::vector<int> out;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        size_t dash = part.find('-');
        int a = std::stoi(part), b = dash == std::string::npos ? a : std::stoi(part.substr(dash + 1));
        for (int c = a; c <= b; ++c) out.push_back(c);
    }
    return out;
}

// Intel hybrid CPUs list their core types under /sys/devices/cpu_atom (E) and cpu_core (P); otherwise one type.
static std::vector<CoreType> core_types() {
    std::vector<CoreType> types;
    for (const char* dev : {"cpu_atom", "cpu_core"}) {
        std::vector<int> cpus = cpu_list(read_line(std::string("/sys/devices/") + dev + "/cpus"));
        if (cpus.empty()) continue;
        std::set<std::string> cores;
        for (int c : cpus) {
            std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
            cores.insert(read_line(base + "physical_package_id") + "/" + read_line(base + "core_id"));
        }
        types.push_back({(int)cores.size(), std::to_string(cpus[0])});
    }
    if (types.empty()) types.push_back({0, "0"});
    return types;
}

static bool pin_to(const std::string& pin) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(std::stoi(pin), &set);
    return sched_setaffinity(0, sizeof set, &set) == 0;
}

#endif

// ---------------------------------------------------------------------------------------------- NVIDIA GPU

static void cuda_facts() {
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
    auto sym = [&](const char* n) { return lib ? (void*)GetProcAddress(lib, n) : nullptr; };
#else
    void* lib = dlopen("libcuda.so.1", RTLD_NOW);
    auto sym = [&](const char* n) { return lib ? dlsym(lib, n) : nullptr; };
#endif
    using init_t = int (*)(unsigned);
    using count_t = int (*)(int*);
    using get_t = int (*)(int*, int);
    using name_t = int (*)(char*, int, int);
    using attr_t = int (*)(int*, int, int);
    auto cuInit = (init_t)sym("cuInit");
    auto cuDeviceGetCount = (count_t)sym("cuDeviceGetCount");
    auto cuDeviceGet = (get_t)sym("cuDeviceGet");
    auto cuDeviceGetName = (name_t)sym("cuDeviceGetName");
    auto cuDeviceGetAttribute = (attr_t)sym("cuDeviceGetAttribute");
    int n = 0;
    if (!cuInit || !cuDeviceGetCount || !cuDeviceGet || !cuDeviceGetName || !cuDeviceGetAttribute ||
        cuInit(0) != 0 || cuDeviceGetCount(&n) != 0 || n == 0)
        return;
    // The first device whose name contains BENCH_GPU (as the benchmark programs choose), else device 0.
    const char* want = std::getenv("BENCH_GPU");
    std::string lw = want ? want : "";
    for (char& c : lw) c = (char)std::tolower((unsigned char)c);
    int dev = -1;
    char name[256] = {0};
    for (int i = 0; i < n && dev < 0; ++i) {
        int d;
        if (cuDeviceGet(&d, i) != 0 || cuDeviceGetName(name, sizeof name, d) != 0) continue;
        std::string ln = name;
        for (char& c : ln) c = (char)std::tolower((unsigned char)c);
        if (lw.empty() || ln.find(lw) != std::string::npos) dev = d;
    }
    if (dev < 0) return;
    auto attr = [&](int a) { int v = 0; return cuDeviceGetAttribute(&v, a, dev) == 0 ? v : 0; };
    const int sm = attr(16), clock_khz = attr(13), mem_khz = attr(36), bus = attr(37);
    std::printf("gpu_cuda_name=%s\n", name);
    if (sm) std::printf("gpu_sm_count=%d\n", sm);
    if (clock_khz) std::printf("gpu_boost_clock_mhz=%d\n", clock_khz / 1000);
    if (mem_khz) std::printf("gpu_memory_clock_mhz=%d\n", mem_khz / 1000);
    if (bus) std::printf("gpu_memory_bus_bits=%d\n", bus);
    if (mem_khz && bus) std::printf("gpu_memory_gbs=%.1f\n", 2.0 * mem_khz * 1e3 * bus / 8 / 1e9);
}

int main() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    std::string flags;   // (__builtin_cpu_supports takes string literals only)
    if (__builtin_cpu_supports("sse4.2")) flags += " sse4_2";
    if (__builtin_cpu_supports("avx")) flags += " avx";
    if (__builtin_cpu_supports("avx2")) flags += " avx2";
    if (__builtin_cpu_supports("fma")) flags += " fma";
    if (__builtin_cpu_supports("avx512f")) flags += " avx512f";
    std::printf("cpu_flags=%s\n", flags.empty() ? "" : flags.c_str() + 1);
#endif
    std::vector<CoreType> types = core_types();   // lowest (efficiency) class first
    if (types.size() > 1) {
        std::printf("performance_cores=%d\nefficiency_cores=%d\n", types.back().cores, types.front().cores);
        if (pin_to(types.front().pin)) std::printf("cpu_e_max_mhz=%.0f\n", add_chain_ghz() * 1000);
    }
    if (pin_to(types.back().pin)) {
        double ghz = add_chain_ghz();
        if (ghz > 0) std::printf("cpu_max_mhz=%.0f\n", ghz * 1000);
    }
    std::fflush(stdout);
    cuda_facts();
    return 0;
}
