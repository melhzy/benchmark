#!/usr/bin/env bash
# run_all.sh -- run the common benchmark (common/SPEC.md) in C++, Python, R and JavaScript, one at a time.
#
# Usage
#   ./run_all.sh                       full run, all four languages (about 7 minutes on a 6-core laptop)
#   ./run_all.sh --quick               smaller sizes, quick check
#   ./run_all.sh --verify              tiny identical sizes, then cross-check results between languages
#   ./run_all.sh --check               only report what is installed for each language; runs nothing
#   ./run_all.sh --langs cpp,js        only some languages (cpp, python, r, js)
#   ./run_all.sh --only cpu_single,gpu only some categories / tests (passed to each program)
#   ./run_all.sh --machine NAME        name of this machine's results folder (default: from its model)
#   ./run_all.sh --cool 55             wait until the CPU is below 55 C before each language (default 60)
#   ./run_all.sh --out DIR             results directory (default: results/<machine>/)
#
# Environment: BENCH_PYTHON (Python to use; default: python3 on PATH), BENCH_GPU (choose an OpenCL GPU
# by part of its name), BENCH_GPU_POWER (JavaScript/WebGPU: high-performance or low-power),
# BENCH_OPENBLAS (full path to libopenblas, if it isn't found automatically).
#
# One invocation is a "batch". It writes into results/<machine>/:
#   <lang>_<time>.csv and _meta.csv   per language (see common/SPEC.md)
#   system_<batch>.csv                hardware and software of this machine (no hostname, no home paths)
#   sensors_<batch>.csv               CPU/GPU temperature, clocks, GPU load, RAM in use; 1 sample/second
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE_ARG="" ONLY="" LANGS="cpp,python,r,js" COOL=60 OUT="" CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick|--verify) MODE_ARG="$1" ;;
    --check)   CHECK=1 ;;
    --only)    ONLY="$2"; shift ;;
    --langs)   LANGS="$2"; shift ;;
    --machine) BENCH_MACHINE="$2"; shift ;;
    --cool)    COOL="$2"; shift ;;
    --out)     OUT="$2"; shift ;;
    -h|--help) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

# --------------------------------------------------------------------------- machine identity
# Same rule as the four programs (common/SPEC.md): BENCH_MACHINE, else the DMI vendor + product name,
# else the hostname; lower-case, runs of other characters replaced by "-".
slugify() { tr '[:upper:]' '[:lower:]' <<< "$1" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'; }
trim() { sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' <<< "$1"; }
read1() { [[ -r "$1" ]] && head -n1 "$1" 2>/dev/null || true; }

if [[ -n "${BENCH_MACHINE:-}" ]]; then
  BENCH_MACHINE="$(slugify "$BENCH_MACHINE")"
else
  model="$(trim "$(trim "$(read1 /sys/class/dmi/id/sys_vendor)") $(trim "$(read1 /sys/class/dmi/id/product_name)")")"
  BENCH_MACHINE="$(slugify "${model:-$(hostname)}")"
fi
export BENCH_MACHINE
OUT="${OUT:-$ROOT/results/$BENCH_MACHINE}"

PYTHON="${BENCH_PYTHON:-$(command -v python3 || true)}"
export RUSTICL_ENABLE="${RUSTICL_ENABLE:-radeonsi,iris}"   # let Mesa's rusticl expose AMD and Intel GPUs

# Node.js 20 or newer: the one on PATH if it is new enough, else the newest from nvm (which is only on PATH in
# interactive shells; distributions often ship an older node in /usr/bin).
find_node() {
  local n v best=""
  for n in "$(command -v node 2>/dev/null)" $(ls -d "$HOME"/.nvm/versions/node/*/bin/node 2>/dev/null | sort -V); do
    [[ -x "$n" ]] || continue
    v="$("$n" -p 'process.versions.node.split(".")[0]' 2>/dev/null)"
    (( ${v:-0} >= 20 )) || continue
    [[ -z "$best" || "$n" != "$(command -v node 2>/dev/null)" ]] && best="$n"
    [[ "$n" == "$(command -v node 2>/dev/null)" ]] && break
  done
  [[ -n "$best" ]] && echo "$best"
}

OPENBLAS_CANDIDATES=(/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0
  /usr/lib/x86_64-linux-gnu/libopenblas.so.0 /usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblas.so.0
  /usr/lib64/libopenblasp.so.0 /usr/lib64/libopenblas.so.0 /usr/lib/libopenblas.so.0 /usr/lib/libopenblas.so)
find_openblas() {
  [[ -n "${BENCH_OPENBLAS:-}" ]] && { echo "$BENCH_OPENBLAS"; return; }
  local f; for f in "${OPENBLAS_CANDIDATES[@]}"; do [[ -e "$f" ]] && { echo "$f"; return; }; done
}

# --------------------------------------------------------------------------- sensors
hwmon_by_name() {
  local want h
  for want in "$@"; do
    for h in /sys/class/hwmon/hwmon*; do
      [[ "$(read1 "$h/name")" == "$want" ]] && { echo "$h"; return; }
    done
  done
  return 0   # no such sensor: empty, not an error (set -e would end the script)
}
CPU_HWMON="$(hwmon_by_name k10temp zenpower coretemp cpu_thermal)"
GPU_HWMON="$(hwmon_by_name amdgpu)"
GPU_DEV="$(dirname "$(ls -d /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)" 2>/dev/null || true)"
NVSMI="$(command -v nvidia-smi || true)"
cpu_temp() { [[ -n "$CPU_HWMON" ]] && echo $(( $(cat "$CPU_HWMON/temp1_input") / 1000 )) || echo 0; }

# --------------------------------------------------------------------------- --check
if (( CHECK )); then
  ok()   { printf "  %-34s %s\n" "$1" "$2"; }
  have() { "$@" >/dev/null 2>&1 && echo yes || echo "NO"; }
  echo "Machine: $BENCH_MACHINE"
  echo "C++";        ok "g++" "$(g++ --version 2>/dev/null | head -1 || echo 'NOT FOUND (install g++ make)')"
                     ok "OpenBLAS" "$(find_openblas || true)"; [[ -z "$(find_openblas || true)" ]] && ok "" "NOT FOUND - install libopenblas (matmul_blas will be skipped)"
                     ok "OpenCL headers (CL/cl.h)" "$( [[ -f /usr/include/CL/cl.h ]] && echo yes || echo 'NO - install opencl-headers ocl-icd-opencl-dev')"
  echo "Python";     ok "interpreter" "${PYTHON:-NOT FOUND} $([[ -n "$PYTHON" ]] && "$PYTHON" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)"
                     for m in numpy pyopencl pandas matplotlib jupyterlab; do ok "  $m" "$([[ -n "$PYTHON" ]] && have "$PYTHON" -c "import $m" || echo NO)"; done
  echo "R";          ok "Rscript" "$(command -v Rscript >/dev/null && Rscript --version 2>&1 | head -1 || echo 'NOT FOUND (install r-base)')"
                     ok "  OpenCL package" "$(have Rscript -e 'stopifnot(requireNamespace("OpenCL", quietly = TRUE))')"
  echo "JavaScript"; NODE="$(find_node || true)"; ok "node" "${NODE:-NOT FOUND (install Node.js 20 or newer, e.g. with nvm)} $([[ -n "$NODE" ]] && "$NODE" --version)"
                     for m in webgpu koffi; do ok "  npm $m" "$([[ -d "$ROOT/JavaScript/node_modules/$m" ]] && echo yes || echo 'NO - run npm install in JavaScript/')"; done
  echo "GPU";        ok "OpenCL devices" "$(command -v clinfo >/dev/null && clinfo -l 2>/dev/null | sed -n 's/.*Device #[0-9]*: //p' | paste -sd ';' || echo 'clinfo not installed')"
  echo "Sensors";    ok "CPU temperature" "${CPU_HWMON:-none found}"
                     ok "GPU sensors" "$( [[ -n "$GPU_HWMON" ]] && echo "amdgpu ($GPU_HWMON)" || { [[ -n "$NVSMI" ]] && echo nvidia-smi || echo none; })"
  exit 0
fi

export BENCH_BATCH="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
ARGS=(--out "$OUT")
[[ -n "$MODE_ARG" ]] && ARGS+=("$MODE_ARG")
[[ -n "$ONLY" ]] && ARGS+=(--only "$ONLY")

# --------------------------------------------------------------------------- build
# The C++ program and hw_probe (hardware facts for the snapshot below) are built first.
echo "Batch $BENCH_BATCH   machine: $BENCH_MACHINE   mode: ${MODE_ARG:---full}   languages: $LANGS"
echo "Results: $OUT"
echo "Python:  ${PYTHON/#$HOME/\~}"
echo "Building C++..."
make -s -C "$ROOT/Cpp" || echo "C++ build failed"

# --------------------------------------------------------------------------- system snapshot
# Everything the notebook needs to describe this machine and compute its theoretical peaks. Cpp/hw_probe adds
# what the kernel does not report: per-type core counts and clocks of a hybrid CPU (measured on one busy core
# of each type) and an NVIDIA GPU's memory bus (from the CUDA driver).
write_system() {
  local f="$OUT/system_${BENCH_BATCH}.csv" udev="" k probe="" rated
  command -v udevadm >/dev/null && udev="$(udevadm info -p /sys/devices/virtual/dmi/id 2>/dev/null || true)"
  udev_val() { sed -n "s/^E: $1=//p" <<< "$udev" | head -1; }
  [[ -x "$ROOT/Cpp/hw_probe" ]] && { echo "Measuring CPU clocks (hw_probe)..."; probe="$("$ROOT/Cpp/hw_probe" 2>/dev/null || true)"; }
  probe_val() { sed -n "s/^$1=//p" <<< "$probe" | head -1; }
  rated="$(awk '{printf "%.0f", $1/1000}' /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || true)"
  kv() { local v="${2//\"/\"\"}"; printf '%s,"%s"\n' "$1" "${v//$HOME/\~}"; }
  {
    echo "key,value"
    kv batch "$BENCH_BATCH"; kv machine "$BENCH_MACHINE"; kv mode "$( [[ -n "$MODE_ARG" ]] && echo "${MODE_ARG#--}" || echo full)"; kv date "$(date -Iseconds)"
    kv suite_commit "$(git -C "$ROOT" describe --always --dirty 2>/dev/null || echo none)"
    kv suite_tests_sha "$(cat "$ROOT"/common/tests.csv "$ROOT"/common/kernels.cl "$ROOT"/common/kernels.wgsl | sha256sum | cut -c1-12)"
    kv vendor "$(trim "$(read1 /sys/class/dmi/id/sys_vendor)")"; kv product "$(trim "$(read1 /sys/class/dmi/id/product_name)")"
    kv chassis "$(case "$(read1 /sys/class/dmi/id/chassis_type)" in 8|9|10|14|30|31|32) echo laptop ;; 3|4|5|6|7|13|15|16|35|36) echo desktop ;; 17|23|28|29) echo server ;; *) echo "" ;; esac)"
    kv os "$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-Linux}")"; kv kernel "$(uname -r)"; kv arch "$(uname -m)"
    kv cpu "$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)"
    kv cpu_max_mhz "${rated:-$(probe_val cpu_max_mhz)}"
    kv cpu_max_mhz_source "$( [[ -n "$rated" ]] && echo cpufreq || { [[ -n "$(probe_val cpu_max_mhz)" ]] && echo 'measured by hw_probe on one busy core'; } )"
    for k in cpu_e_max_mhz performance_cores efficiency_cores; do if [[ -n "$(probe_val $k)" ]]; then kv $k "$(probe_val $k)"; fi; done
    kv physical_cores "$(awk -F: '/^physical id/{p=$2} /^core id/{print p "-" $2}' /proc/cpuinfo | sort -u | wc -l)"
    kv logical_cpus "$(nproc)"
    kv smt "$(read1 /sys/devices/system/cpu/smt/active)"
    kv cpu_flags "$(grep -m1 -oE '\b(sse4_2|avx|avx2|fma|avx512f|avx512_bf16|amx_tile|asimd|sve)\b' /proc/cpuinfo | sort -u | paste -sd ' ')"
    kv l3_cache "$(read1 /sys/devices/system/cpu/cpu0/cache/index3/size)"
    kv ram_gib "$(awk '/^MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)"
    kv ram_type "$(udev_val MEMORY_DEVICE_0_TYPE)"
    kv ram_speed_mts "$(udev_val MEMORY_DEVICE_0_CONFIGURED_SPEED_MTS)"
    kv ram_modules "$(grep -cE '^E: MEMORY_DEVICE_[0-9]+_SIZE=' <<< "$udev" || true)"
    kv ram_module_sizes_gib "$(sed -n 's/^E: MEMORY_DEVICE_[0-9]*_SIZE=//p' <<< "$udev" | awk '{printf "%s%g", sep, $1/2^30; sep=" "}')"
    kv gpus "$(lspci 2>/dev/null | grep -iE 'vga|3d controller|display' | sed -E 's/^[^ ]+ [^:]+: //' | paste -sd ';')"
    kv opencl_devices "$(command -v clinfo >/dev/null && clinfo -l 2>/dev/null | sed -n 's/.*Device #[0-9]*: //p' | paste -sd ';')"
    for k in gpu_cuda_name gpu_sm_count gpu_boost_clock_mhz gpu_memory_clock_mhz gpu_memory_bus_bits gpu_memory_gbs; do
      if [[ -n "$(probe_val $k)" ]]; then kv $k "$(probe_val $k)"; fi
    done
    kv power "$(for p in /sys/class/power_supply/*/online; do [[ "$(read1 "$p")" == 1 ]] && { echo AC; break; }; done | grep . || echo battery)"
    kv platform_profile "$(read1 /sys/firmware/acpi/platform_profile)"
    kv governor "$(read1 /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    kv epp "$(read1 /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference)"
    kv transparent_hugepages "$(sed -nE 's/.*\[([a-z]+)\].*/\1/p' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
    kv python "${PYTHON:-}"; kv node "$(find_node || true)"; kv openblas "$(find_openblas || true)"
    for k in BENCH_GPU BENCH_GPU_POWER RUSTICL_ENABLE; do kv "env_$k" "${!k:-}"; done
  } > "$f"
}
write_system

PHASE_FILE="$(mktemp)"
echo "idle" > "$PHASE_FILE"
SENSORS="$OUT/sensors_${BENCH_BATCH}.csv"

sensor_loop() {
  echo "time,phase,cpu_temp_c,cpu_mhz_avg,cpu_mhz_max,gpu_temp_c,gpu_busy_pct,gpu_sclk_mhz,mem_used_gib"
  while true; do
    local t ct mhz gt="" gb="" gs="" mem
    t="$(date +%s.%N)"
    ct="$([[ -n "$CPU_HWMON" ]] && awk '{printf "%.1f", $1/1000}' "$CPU_HWMON/temp1_input")"
    mhz="$(awk -F: '/^cpu MHz/{s+=$2; n++; if ($2>m) m=$2} END{if (n) printf "%.0f,%.0f", s/n, m; else printf ","}' /proc/cpuinfo)"
    if [[ -n "$GPU_HWMON" || -n "$GPU_DEV" ]]; then
      gt="$([[ -n "$GPU_HWMON" ]] && awk '{printf "%.1f", $1/1000}' "$GPU_HWMON/temp1_input" 2>/dev/null)"
      gb="$([[ -n "$GPU_DEV" ]] && cat "$GPU_DEV/gpu_busy_percent" 2>/dev/null)"
      gs="$([[ -n "$GPU_DEV" ]] && awk '/\*/{gsub(/[^0-9]/, "", $2); print $2}' "$GPU_DEV/pp_dpm_sclk" 2>/dev/null)"
    elif [[ -n "$NVSMI" ]]; then
      IFS=', ' read -r gt gb gs < <("$NVSMI" --query-gpu=temperature.gpu,utilization.gpu,clocks.sm \
                                     --format=csv,noheader,nounits -i 0 2>/dev/null || true)
    fi
    mem="$(awk '/^MemTotal/{t=$2} /^MemAvailable/{a=$2} END{printf "%.2f", (t-a)/1048576}' /proc/meminfo)"
    echo "$t,$(cat "$PHASE_FILE"),$ct,$mhz,$gt,$gb,$gs,$mem"
    sleep 1
  done
}
sensor_loop > "$SENSORS" &
SENSOR_PID=$!
cleanup() { kill "$SENSOR_PID" 2>/dev/null || true; rm -f "$PHASE_FILE"; }
trap cleanup EXIT

cool_down() {
  [[ "$MODE_ARG" == "--verify" || -z "$CPU_HWMON" ]] && return
  local waited=0
  while (( $(cpu_temp) >= COOL && waited < 300 )); do
    (( waited == 0 )) && printf "Cooling down to below %s C (now %s C)..." "$COOL" "$(cpu_temp)"
    sleep 5; waited=$((waited + 5))
  done
  (( waited > 0 )) && printf " %s C after %s s\n" "$(cpu_temp)" "$waited"
  return 0
}

# --------------------------------------------------------------------------- run
FAILED=()
IFS=',' read -ra LANG_LIST <<< "$LANGS"
for lang in "${LANG_LIST[@]}"; do
  case "$lang" in
    cpp)    CMD=("$ROOT/Cpp/common_benchmark") ;;
    python) if [[ -z "$PYTHON" ]]; then echo "python3 not found - skipping python" >&2; FAILED+=("python"); continue; fi
            CMD=("$PYTHON" "$ROOT/Python/common_benchmark.py") ;;
    r)      CMD=(Rscript "$ROOT/R/common_benchmark.R") ;;
    js)     NODE="$(find_node || true)"
            if [[ -z "$NODE" ]]; then echo "Node.js not found - skipping js" >&2; FAILED+=("js"); continue; fi
            CMD=("$NODE" --expose-gc "$ROOT/JavaScript/common_benchmark.mjs") ;;
    *) echo "unknown language: $lang (use cpp, python, r, js)" >&2; exit 2 ;;
  esac
  echo; echo "=================================================================== $lang"
  cool_down
  echo "$lang" > "$PHASE_FILE"
  "${CMD[@]}" "${ARGS[@]}" || FAILED+=("$lang")
  echo "idle" > "$PHASE_FILE"
done

sleep 2   # one more idle sensor sample
# The highest GPU clock the sensor log saw (a GPU can boost above the clock the driver reports).
gpu_max="$(awk -F, 'NR > 1 && $8 ~ /^[0-9]+$/ && $8 > m {m = $8} END {if (m) print m}' "$SENSORS")"
if [[ -n "$gpu_max" ]]; then echo "gpu_clock_max_logged_mhz,\"$gpu_max\"" >> "$OUT/system_${BENCH_BATCH}.csv"; fi
echo
echo "System:  $OUT/system_${BENCH_BATCH}.csv"
echo "Sensors: $SENSORS"
if [[ "$MODE_ARG" == "--verify" && -n "$PYTHON" ]]; then
  "$PYTHON" "$ROOT/common/verify.py" --batch "$BENCH_BATCH" --dir "$OUT"
fi
if (( ${#FAILED[@]} )); then
  echo "Finished with errors in: ${FAILED[*]}"; exit 1
fi
echo "Done. Open analysis.ipynb to explore the results."
