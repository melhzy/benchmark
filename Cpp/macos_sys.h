// macos_sys.h -- macOS system facts for common_benchmark.cpp and hw_probe.cpp (see ../common/SPEC.md).
// Everything here works without administrator rights:
//   sysctl                       CPU model, core counts per core type, RAM size
//   host_statistics64            available RAM (free + inactive pages, as vm_stat shows them)
//   IOKit power sources / pmset  AC or battery, and the power mode (Low Power / Automatic / High Power)
//   IORegistry                   model name and manufacturer, GPU core count, load and clock table
//   IOHID event system           SoC die temperatures (the "PMU tdie" sensors; a private but long-stable API that
//                                Apple's own tools and utilities such as macmon use)
// Link with -framework IOKit -framework CoreFoundation.
#pragma once
#ifdef __APPLE__

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace macos {

inline std::string sysctl_str(const char* name) {
    size_t n = 0;
    if (sysctlbyname(name, nullptr, &n, nullptr, 0) != 0 || n == 0) return "";
    std::string s(n, '\0');
    if (sysctlbyname(name, s.data(), &n, nullptr, 0) != 0) return "";
    return std::string(s.c_str());
}

inline long long sysctl_int(const char* name, long long fallback = 0) {
    int64_t v64 = 0;
    size_t n = sizeof v64;
    if (sysctlbyname(name, &v64, &n, nullptr, 0) != 0) return fallback;
    if (n == sizeof(int32_t)) { int32_t v32; std::memcpy(&v32, &v64, sizeof v32); return v32; }
    return v64;
}

// Free + inactive pages (vm_stat: "Pages free" + "Pages speculative" + "Pages inactive"), in bytes: memory the
// system can hand out without swapping, the macOS counterpart of Linux's MemAvailable.
inline double ram_available_bytes() {
    vm_statistics64_data_t s;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&s, &count) != KERN_SUCCESS) return 0;
    return ((double)s.free_count + s.inactive_count) * (double)sysctl_int("hw.pagesize", 16384);
}

inline std::string cf_to_string(CFTypeRef v) {
    if (!v) return "";
    char buf[512] = {0};
    if (CFGetTypeID(v) == CFStringGetTypeID()) {
        CFStringGetCString((CFStringRef)v, buf, sizeof buf, kCFStringEncodingUTF8);
    } else if (CFGetTypeID(v) == CFDataGetTypeID()) {          // IORegistry strings are often NUL-terminated data
        CFIndex n = std::min<CFIndex>(CFDataGetLength((CFDataRef)v), sizeof buf - 1);
        std::memcpy(buf, CFDataGetBytePtr((CFDataRef)v), n);
    }
    return std::string(buf);
}

// A property of a registry entry given by path (e.g. "IODeviceTree:/product") or, with by_class, of the first
// service of that class. The caller releases the result.
inline CFTypeRef registry_property(const char* path_or_class, const char* key, bool by_class = false) {
    io_registry_entry_t e = by_class
        ? IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching(path_or_class))
        : IORegistryEntryFromPath(kIOMainPortDefault, path_or_class);
    if (!e) return nullptr;
    CFStringRef k = CFStringCreateWithCString(nullptr, key, kCFStringEncodingUTF8);
    CFTypeRef v = IORegistryEntryCreateCFProperty(e, k, kCFAllocatorDefault, 0);
    CFRelease(k);
    IOObjectRelease(e);
    return v;
}

inline std::string registry_string(const char* path_or_class, const char* key, bool by_class = false) {
    CFTypeRef v = registry_property(path_or_class, key, by_class);
    std::string s = cf_to_string(v);
    if (v) CFRelease(v);
    return s;
}

// Firmware vendor and product, the counterparts of the DMI sys_vendor / product_name (SPEC.md, machine id):
// "Apple Inc." and the model name ("MacBook Pro (14-inch, M5 Pro)"; Intel Macs: the model id, "MacBookPro16,1").
inline std::string vendor() { return registry_string("IOPlatformExpertDevice", "manufacturer", true); }
inline std::string product() {
    std::string p = registry_string("IODeviceTree:/product", "product-name");
    return p.empty() ? sysctl_str("hw.model") : p;
}

inline std::string power_source() {
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (!info) return "";
    CFStringRef type = IOPSGetProvidingPowerSourceType(info);
    std::string s = type && CFStringCompare(type, CFSTR(kIOPMACPowerKey), 0) == kCFCompareEqualTo ? "AC" : "battery";
    CFRelease(info);
    return s;
}

// The power mode for the current power source (System Settings > Battery), from `pmset -g`:
// powermode 0 / 1 / 2 = automatic / low power / high power (Macs that offer High Power), else lowpowermode 0 / 1.
inline std::string power_mode() {
    FILE* p = popen("/usr/bin/pmset -g 2>/dev/null", "r");
    if (!p) return "";
    char line[256];
    std::string mode;
    while (std::fgets(line, sizeof line, p)) {
        int v;
        if (std::sscanf(line, " powermode %d", &v) == 1) mode = v == 1 ? "low power" : v == 2 ? "high power" : "automatic";
        else if (mode.empty() && std::sscanf(line, " lowpowermode %d", &v) == 1) mode = v ? "low power" : "automatic";
    }
    pclose(p);
    return mode;
}

}  // namespace macos

// ---- SoC temperature through the IOHID event system (private API, declared here)
extern "C" {
typedef struct __IOHIDEventSystemClient* IOHIDEventSystemClientRef;
typedef struct __IOHIDServiceClient* IOHIDServiceClientRef;
typedef struct __IOHIDEvent* IOHIDEventRef;
IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef);
int IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef, CFDictionaryRef);
CFArrayRef IOHIDEventSystemClientCopyServices(IOHIDEventSystemClientRef);
CFTypeRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef, CFStringRef);
IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef, int64_t, int32_t, int64_t);
double IOHIDEventGetFloatValue(IOHIDEventRef, int32_t);
}

namespace macos {

// The hottest of the SoC's die sensors ("PMU tdie1", "PMU tdie2", ...; Intel Macs have none), in deg C, or < -100
// if there is none. The CPU and GPU share the die, so this is the counterpart of Linux's CPU temperature.
inline double soc_temp_c() {
    constexpr int64_t kTemperatureEvent = 15;
    IOHIDEventSystemClientRef client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
    if (!client) return -1000;
    int page = 0xff00, usage = 5;   // Apple vendor page, temperature sensors
    CFNumberRef p = CFNumberCreate(nullptr, kCFNumberIntType, &page), u = CFNumberCreate(nullptr, kCFNumberIntType, &usage);
    const void* keys[] = {CFSTR("PrimaryUsagePage"), CFSTR("PrimaryUsage")};
    const void* vals[] = {p, u};
    CFDictionaryRef match = CFDictionaryCreate(nullptr, keys, vals, 2, &kCFTypeDictionaryKeyCallBacks,
                                               &kCFTypeDictionaryValueCallBacks);
    IOHIDEventSystemClientSetMatching(client, match);
    double best = -1000;
    if (CFArrayRef services = IOHIDEventSystemClientCopyServices(client)) {
        for (CFIndex i = 0; i < CFArrayGetCount(services); ++i) {
            auto sc = (IOHIDServiceClientRef)CFArrayGetValueAtIndex(services, i);
            CFTypeRef name = IOHIDServiceClientCopyProperty(sc, CFSTR("Product"));
            bool die = cf_to_string(name).rfind("PMU tdie", 0) == 0;
            if (name) CFRelease(name);
            if (!die) continue;
            if (IOHIDEventRef e = IOHIDServiceClientCopyEvent(sc, kTemperatureEvent, 0, 0)) {
                double t = IOHIDEventGetFloatValue(e, (int32_t)(kTemperatureEvent << 16));
                if (t > 0 && t < 150) best = std::max(best, t);
                CFRelease(e);
            }
        }
        CFRelease(services);
    }
    CFRelease(match); CFRelease(p); CFRelease(u); CFRelease(client);
    return best;
}

// ---- Apple GPU (IORegistry, class AGXAccelerator)

inline int gpu_cores() {
    CFTypeRef v = registry_property("AGXAccelerator", "gpu-core-count", true);
    int n = 0;
    if (v && CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)v, kCFNumberIntType, &n);
    if (v) CFRelease(v);
    return n;
}

// "Device Utilization %" of the GPU's performance statistics (what Activity Monitor's GPU history shows), or -1.
inline int gpu_busy_pct() {
    CFTypeRef stats = registry_property("AGXAccelerator", "PerformanceStatistics", true);
    int pct = -1;
    if (stats && CFGetTypeID(stats) == CFDictionaryGetTypeID()) {
        auto v = (CFNumberRef)CFDictionaryGetValue((CFDictionaryRef)stats, CFSTR("Device Utilization %"));
        if (v) CFNumberGetValue(v, kCFNumberIntType, &pct);
    }
    if (stats) CFRelease(stats);
    return pct;
}

// Highest GPU clock in MHz, from the power manager's DVFS table for the GPU ("voltage-states9" of the "pmgr"
// device: pairs of frequency in Hz and voltage), the table Apple Silicon has used for its GPU since the M1.
// 0 if not found (Intel Macs).
inline double gpu_max_clock_mhz() {
    io_registry_entry_t e = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching("pmgr"));
    if (!e) return 0;
    CFTypeRef v = IORegistryEntryCreateCFProperty(e, CFSTR("voltage-states9"), kCFAllocatorDefault, 0);
    IOObjectRelease(e);
    double best = 0;
    if (v && CFGetTypeID(v) == CFDataGetTypeID()) {
        const uint8_t* b = CFDataGetBytePtr((CFDataRef)v);
        for (CFIndex off = 0; off + 8 <= CFDataGetLength((CFDataRef)v); off += 8) {
            uint32_t hz;
            std::memcpy(&hz, b + off, 4);
            best = std::max(best, hz / 1e6);
        }
    }
    if (v) CFRelease(v);
    return best >= 100 && best <= 10000 ? best : 0;   // anything else is not a GPU clock table in Hz
}

}  // namespace macos

#endif  // __APPLE__
