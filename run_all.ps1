<#
run_all.ps1 -- Windows version of run_all.sh: run the common benchmark (common/SPEC.md) in C++, Python, R and
JavaScript, one at a time. Start it through run_all.cmd (which bypasses PowerShell's script execution policy):

  .\run_all.cmd                       full run, all four languages
  .\run_all.cmd --quick               smaller sizes, quick check
  .\run_all.cmd --verify              tiny identical sizes, then cross-check results between languages
  .\run_all.cmd --check               only report what is installed for each language; runs nothing
  .\run_all.cmd --langs cpp,js        only some languages (cpp, python, r, js)
  .\run_all.cmd --only cpu_single,gpu only some categories / tests (passed to each program)
  .\run_all.cmd --machine NAME        name of this machine's results folder (default: from its model)
  .\run_all.cmd --pause 30            idle seconds before each language after the first (default 30; see below)
  .\run_all.cmd --out DIR             results directory (default: results\<machine>\)

Environment: BENCH_PYTHON, BENCH_RSCRIPT (Python / Rscript to use), BENCH_GPU (choose an OpenCL GPU by part of
its name), BENCH_GPU_POWER (JavaScript/WebGPU: high-performance or low-power), BENCH_OPENBLAS (libopenblas.dll),
BENCH_OPENCL_SDK (OpenCL SDK folder). setup_windows.ps1 installs the dependencies.

Differences from Linux: Windows lets no program read the CPU temperature without administrator rights, so instead
of waiting for the CPU to cool below 60 C, the runner pauses for a fixed time before each language; the sensor log
has CPU clocks and RAM from Windows performance counters and GPU data from nvidia-smi (NVIDIA GPUs only).

Writes into results\<machine>\ the same files as run_all.sh: <lang>_<time>.csv and _meta.csv, system_<batch>.csv
and sensors_<batch>.csv.
#>
$ErrorActionPreference = 'Continue'   # (not 'Stop': native programs' stderr must not abort the run)
. "$PSScriptRoot\common\windows_tools.ps1"
$Root = $PSScriptRoot

# ------------------------------------------------------------------------------------------------- options
# Same option names as run_all.sh; "--name" and "-name" both work.
$ModeArg = ''; $Only = ''; $Langs = 'cpp,python,r,js'; $Pause = 30; $Out = ''; $Check = $false
$argv = @($args)
for ($i = 0; $i -lt $argv.Count; $i++) {
  $a = "$($argv[$i])".ToLowerInvariant() -replace '^-+', '--'
  $value = $null
  if ($a -in '--only', '--langs', '--machine', '--pause', '--out') {
    if ($i + 1 -ge $argv.Count) { Write-Host "$a needs a value"; exit 2 }
    $i++; $value = "$($argv[$i])"
  }
  switch ($a) {
    { $_ -in '--quick', '--verify' } { $ModeArg = $a }
    '--check'   { $Check = $true }
    '--only'    { $Only = $value }
    '--langs'   { $Langs = $value }
    '--machine' { $env:BENCH_MACHINE = $value }
    '--pause'   { $Pause = [int]$value }
    '--out'     { $Out = $value }
    { $_ -in '--help', '--h', '--?' } {
      $text = Get-Content -LiteralPath $PSCommandPath -Raw
      Write-Host $text.Substring(2, $text.IndexOf('#>') - 2).Trim(); exit 0 }
    default { Write-Host "unknown option: $($argv[$i]) (see --help)"; exit 2 }
  }
}
$Inv = [System.Globalization.CultureInfo]::InvariantCulture   # CSV numbers always use "." (any Windows locale)

$Machine = Get-MachineId
$env:BENCH_MACHINE = $Machine
if (-not $Out) { $Out = Join-Path $Root "results\$Machine" }
$Python = Find-Python
$Rscript = Find-Rscript
$Node = Find-Node
$Toolchain = Find-Toolchain
$OpenBlas = Find-OpenBlas
$OpenClSdk = Find-OpenClSdk
$CppExe = Join-Path $Root 'Cpp\common_benchmark.exe'
$Nvsmi = Get-CommandPath 'nvidia-smi'
$Clinfo = Get-FirstExisting @((Get-CommandPath 'clinfo'), $(if ($OpenClSdk) { Join-Path $OpenClSdk 'bin\clinfo.exe' }))
$env:PYTHONIOENCODING = 'utf-8'

function Get-OpenClDevices {
  if (-not $Clinfo) { return '' }
  $lines = & $Clinfo -l 2>$null
  return (@($lines | ForEach-Object { if ("$_" -match 'Device #\d+: (.*)$') { $Matches[1].Trim() } }) -join ';')
}

# ------------------------------------------------------------------------------------------------- --check
if ($Check) {
  function Ok([string]$Name, [string]$Value) { Write-Host ('  {0,-34} {1}' -f $Name, $Value) }
  function Has([scriptblock]$Test) { try { & $Test 2>$null | Out-Null; if ($LASTEXITCODE -eq 0) { 'yes' } else { 'NO' } } catch { 'NO' } }
  Write-Host "Machine: $Machine"
  Write-Host 'C++'
  Ok 'g++ / make' $(if ($Toolchain) { "$((& $Toolchain.Gxx --version | Select-Object -First 1)) / $($Toolchain.Make)" } else { 'NOT FOUND (install Rtools, or MSYS2 with g++ and make)' })
  Ok 'OpenBLAS' $(if ($OpenBlas) { $OpenBlas } else { 'NOT FOUND - run setup_windows.ps1 (matmul_blas will be skipped)' })
  Ok 'OpenCL SDK (CL/cl.h)' $(if ($OpenClSdk) { $OpenClSdk } else { 'NO - run setup_windows.ps1 (GPU tests will be skipped)' })
  Write-Host 'Python'
  Ok 'interpreter' $(if ($Python) { "$Python $(& $Python -c 'import sys; print(sys.version.split()[0])')" } else { 'NOT FOUND (install Python 3, or set BENCH_PYTHON)' })
  foreach ($m in 'numpy', 'pyopencl', 'pandas', 'matplotlib', 'jupyterlab') {
    Ok "  $m" $(if ($Python) { Has { & $Python -c "import $m" } } else { 'NO' }) }
  Write-Host 'R'
  Ok 'Rscript' $(if ($Rscript) { "$Rscript ($(& $Rscript -e 'cat(R.version.string)'))" } else { 'NOT FOUND (install R, or set BENCH_RSCRIPT)' })
  Ok '  OpenCL package' $(if ($Rscript) { Has { & $Rscript -e "quit(status = !requireNamespace('OpenCL', quietly = TRUE))" } } else { 'NO' })
  Write-Host 'JavaScript'
  Ok 'node' $(if ($Node) { "$Node $(& $Node --version)" } else { 'NOT FOUND (install Node.js 20 or newer)' })
  foreach ($m in 'webgpu', 'koffi') {
    Ok "  npm $m" $(if (Test-Path (Join-Path $Root "JavaScript\node_modules\$m")) { 'yes' } else { 'NO - run setup_windows.ps1 (npm install in JavaScript\)' }) }
  Write-Host 'GPU'
  Ok 'OpenCL devices' $(if ($Clinfo) { Get-OpenClDevices } else { 'clinfo not found (it comes with the OpenCL SDK)' })
  Write-Host 'Sensors'
  Ok 'CPU temperature' 'none (Windows needs administrator rights for it)'
  Ok 'CPU clock, RAM' 'Windows performance counters'
  Ok 'GPU sensors' $(if ($Nvsmi) { 'nvidia-smi' } else { 'none' })
  exit 0
}

$env:BENCH_BATCH = Get-Date -Format 'yyyyMMdd-HHmmss'
$Batch = $env:BENCH_BATCH
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$ProgArgs = @('--out', $Out)
if ($ModeArg) { $ProgArgs += $ModeArg }
if ($Only) { $ProgArgs += @('--only', $Only) }

# ------------------------------------------------------------------------------------------------- build
# The C++ program and hw_probe (hardware facts for the snapshot below) are built first.
Write-Host "Batch $Batch   machine: $Machine   mode: $(if ($ModeArg) { $ModeArg } else { '--full' })   languages: $Langs"
Write-Host "Results: $Out"
Write-Host "Python:  $(if ($Python) { $Python.Replace($env:USERPROFILE, '~') } else { 'not found' })"
if ($Toolchain) {
  Write-Host 'Building C++...'
  $savedPath = $env:PATH
  $env:PATH = (($Toolchain.Path + $env:PATH.Split(';')) -join ';')
  # (the Makefile looks in deps\ itself; pass other locations on)
  if ($OpenClSdk -and $OpenClSdk -ne (Join-Path $DepsDir 'opencl-sdk')) { $env:BENCH_OPENCL_SDK = $OpenClSdk }
  & $Toolchain.Make -s -C (Join-Path $Root 'Cpp')
  if ($LASTEXITCODE -ne 0) { Write-Host 'C++ build failed' }
  $env:PATH = $savedPath
} elseif (Test-Path $CppExe) {
  Write-Host 'No g++/make found - using the existing Cpp\common_benchmark.exe'
}

# ------------------------------------------------------------------------------------------------- system snapshot
# Everything the notebook and the results page need to describe this machine and compute its theoretical peaks;
# the same keys as run_all.sh where Windows has the information. Cpp\hw_probe.exe measures the CPU's clock on one
# busy core of each type (Windows does not report the boost clock) and reads an NVIDIA GPU's memory bus from the
# CUDA driver.
function Get-Probe {
  $probe = Join-Path $Root 'Cpp\hw_probe.exe'
  $h = @{}
  if (Test-Path $probe) {
    Write-Host 'Measuring CPU clocks (hw_probe)...'
    foreach ($line in & $probe) { if ("$line" -match '^([a-z0-9_]+)=(.*)$') { $h[$Matches[1]] = $Matches[2] } }
  }
  return $h
}
function Write-System {
  $f = Join-Path $Out "system_$Batch.csv"
  $home1 = $env:USERPROFILE
  $kv = [System.Collections.Generic.List[string]]::new()
  $kv.Add('key,value')
  function Kv([string]$Key, $Value) {
    $v = "$Value"
    if ($home1) { $v = $v.Replace($home1, '~').Replace(($home1 -replace '\\', '/'), '~') }
    $kv.Add(('{0},"{1}"' -f $Key, $v.Replace('"', '""')))
  }
  $os = Get-CimInstance Win32_OperatingSystem
  $cs = Get-CimInstance Win32_ComputerSystem
  $cpu = @(Get-CimInstance Win32_Processor)
  $mem = @(Get-CimInstance Win32_PhysicalMemory)
  $bios = Get-ItemProperty 'HKLM:\HARDWARE\DESCRIPTION\System\BIOS' -ErrorAction SilentlyContinue
  $cur = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction SilentlyContinue
  $chassis = @((Get-CimInstance Win32_SystemEnclosure).ChassisTypes)[0]
  $probe = Get-Probe
  $ramTypes = @{ 20 = 'DDR'; 21 = 'DDR2'; 24 = 'DDR3'; 26 = 'DDR4'; 27 = 'LPDDR'; 28 = 'LPDDR2'; 29 = 'LPDDR3'; 30 = 'LPDDR4'; 34 = 'DDR5'; 35 = 'LPDDR5' }
  Add-Type -AssemblyName System.Windows.Forms
  $line = [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus
  $modes = @{ '961cc777-2547-4f9d-8174-7d86181b8a7a' = 'best power efficiency'; '00000000-0000-0000-0000-000000000000' = 'balanced'
              'ded574b5-45a0-4f42-8737-46345c09c238' = 'best performance' }
  $pw = Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes' -ErrorAction SilentlyContinue
  $overlay = "$(if ("$line" -eq 'Offline') { $pw.ActiveOverlayDcPowerScheme } else { $pw.ActiveOverlayAcPowerScheme })".ToLowerInvariant()
  $plan = ((powercfg /getactivescheme) -replace '^.*\((.*)\)\s*$', '$1').Trim()
  $testsSha = [System.Security.Cryptography.SHA256]::Create()
  # Hash of tests.csv and both kernel files with LF line endings, so it matches run_all.sh on Linux.
  $bytes = [System.Text.Encoding]::UTF8.GetBytes((@('common\tests.csv', 'common\kernels.cl', 'common\kernels.wgsl') |
    ForEach-Object { [System.IO.File]::ReadAllText((Join-Path $Root $_)) -replace "`r`n", "`n" }) -join '')
  $sha = (-join ($testsSha.ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') })).Substring(0, 12)
  $commit = & git -C $Root describe --always --dirty 2>$null
  Kv batch $Batch; Kv machine $Machine; Kv mode $(if ($ModeArg) { $ModeArg.TrimStart('-') } else { 'full' })
  Kv date (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
  Kv suite_commit $(if ($commit) { $commit } else { 'none' }); Kv suite_tests_sha $sha
  Kv vendor "$($bios.SystemManufacturer)".Trim(); Kv product "$($bios.SystemProductName)".Trim()
  Kv chassis $(if ($chassis -in 8, 9, 10, 14, 30, 31, 32) { 'laptop' } elseif ($chassis -in 3, 4, 5, 6, 7, 13, 15, 16, 35, 36) { 'desktop' } elseif ($chassis -in 17, 23, 28, 29) { 'server' } else { '' })
  Kv os "$($os.Caption -replace '^Microsoft ', '') $($cur.DisplayVersion)".Trim()
  Kv kernel "$($os.Version).$($cur.UBR)"
  Kv arch $(if ($env:PROCESSOR_ARCHITECTURE -eq 'AMD64') { 'x86_64' } elseif ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64' } else { $env:PROCESSOR_ARCHITECTURE })
  Kv cpu "$($cpu[0].Name)".Trim()
  Kv cpu_max_mhz $probe['cpu_max_mhz']
  Kv cpu_max_mhz_source $(if ($probe['cpu_max_mhz']) { 'measured by hw_probe on one busy core (Windows does not report the boost clock)' } else { '' })
  if ($probe['cpu_e_max_mhz']) { Kv cpu_e_max_mhz $probe['cpu_e_max_mhz'] }
  Kv cpu_base_mhz $cpu[0].MaxClockSpeed
  Kv physical_cores (($cpu | Measure-Object -Property NumberOfCores -Sum).Sum)
  Kv logical_cpus $cs.NumberOfLogicalProcessors
  if ($probe['performance_cores']) { Kv performance_cores $probe['performance_cores']; Kv efficiency_cores $probe['efficiency_cores'] }
  Kv smt $(if ($cs.NumberOfLogicalProcessors -gt (($cpu | Measure-Object -Property NumberOfCores -Sum).Sum)) { 1 } else { 0 })
  Kv cpu_flags $probe['cpu_flags']
  Kv l3_cache "$($cpu[0].L3CacheSize)K"
  Kv ram_gib ([string]::Format($Inv, '{0:F1}', $os.TotalVisibleMemorySize / 1MB))
  Kv ram_type $ramTypes[[int]$mem[0].SMBIOSMemoryType]
  Kv ram_speed_mts $mem[0].ConfiguredClockSpeed
  Kv ram_modules $mem.Count
  Kv ram_module_sizes_gib (($mem | ForEach-Object { [string]::Format($Inv, '{0:G}', [math]::Round($_.Capacity / 1GB, 1)) }) -join ' ')
  Kv gpus ((@(Get-CimInstance Win32_VideoController) | ForEach-Object { $_.Name }) -join ';')
  Kv opencl_devices (Get-OpenClDevices)
  foreach ($k in 'gpu_cuda_name', 'gpu_sm_count', 'gpu_boost_clock_mhz', 'gpu_memory_clock_mhz', 'gpu_memory_bus_bits', 'gpu_memory_gbs') {
    if ($probe[$k]) { Kv $k $probe[$k] } }
  Kv power $(if ("$line" -eq 'Offline') { 'battery' } else { 'AC' })
  Kv platform_profile $(if ($modes.ContainsKey($overlay)) { $modes[$overlay] } else { $overlay })
  Kv governor "Windows power plan: $plan"
  Kv epp ''
  Kv transparent_hugepages ''
  Kv python $Python; Kv node $Node; Kv openblas $OpenBlas; Kv rscript $Rscript
  foreach ($k in 'BENCH_GPU', 'BENCH_GPU_POWER', 'RUSTICL_ENABLE') { Kv "env_$k" ([Environment]::GetEnvironmentVariable($k)) }
  [System.IO.File]::WriteAllLines($f, $kv)
}
Write-System

# ------------------------------------------------------------------------------------------------- sensors
# A background job writes one row per second; the phase (language running) is read from a small file.
$PhaseFile = [System.IO.Path]::GetTempFileName()
Set-Content -LiteralPath $PhaseFile -Value 'idle' -NoNewline
$Sensors = Join-Path $Out "sensors_$Batch.csv"
$SensorJob = Start-Job -ArgumentList $Sensors, $PhaseFile, $Nvsmi, ((Get-CimInstance Win32_Processor | Select-Object -First 1).MaxClockSpeed) -ScriptBlock {
  param($Path, $PhaseFile, $Nvsmi, $BaseMhz)
  $w = [System.IO.StreamWriter]::new($Path, $false, [System.Text.UTF8Encoding]::new($false))
  $w.AutoFlush = $true
  $w.WriteLine('time,phase,cpu_temp_c,cpu_mhz_avg,cpu_mhz_max,gpu_temp_c,gpu_busy_pct,gpu_sclk_mhz,mem_used_gib')
  $totalKib = (Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize
  $counters = '\Processor Information(*)\% Processor Performance', '\Memory\Available KBytes'
  # Effective clock = base clock x "% Processor Performance" (what Task Manager shows as "Speed").
  Get-Counter -Counter $counters -SampleInterval 1 -Continuous -ErrorAction SilentlyContinue | ForEach-Object {
    $t = ([DateTimeOffset]$_.Timestamp).ToUnixTimeMilliseconds() / 1000
    $perf = @($_.CounterSamples | Where-Object { $_.Path -like '*% processor performance' })
    $avg = ($perf | Where-Object { $_.InstanceName -eq '_total' }).CookedValue
    $max = ($perf | Where-Object { $_.InstanceName -notlike '*_total' } | Measure-Object -Property CookedValue -Maximum).Maximum
    $availKib = ($_.CounterSamples | Where-Object { $_.Path -like '*available kbytes' }).CookedValue
    $gt = $gb = $gs = ''
    if ($Nvsmi) {
      $q = & $Nvsmi --query-gpu=temperature.gpu,utilization.gpu,clocks.sm --format=csv,noheader,nounits -i 0 2>$null
      if ($q) { $gt, $gb, $gs = ("$q" -split ',\s*') }
    }
    $phase = (Get-Content -LiteralPath $PhaseFile -Raw -ErrorAction SilentlyContinue)
    if (-not $phase) { $phase = 'idle' }
    $w.WriteLine([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:F3},{1},,{2:F0},{3:F0},{4},{5},{6},{7:F2}',
      $t, $phase.Trim(), ($BaseMhz * $avg / 100), ($BaseMhz * $max / 100), $gt, $gb, $gs, (($totalKib - $availKib) / 1MB)))
  }
}

function Stop-Sensors {
  if ($SensorJob) { Stop-Job $SensorJob -ErrorAction SilentlyContinue; Remove-Job $SensorJob -Force -ErrorAction SilentlyContinue }
  Remove-Item -LiteralPath $PhaseFile -Force -ErrorAction SilentlyContinue
}

function Set-Phase([string]$Name) { Set-Content -LiteralPath $PhaseFile -Value $Name -NoNewline }

# ------------------------------------------------------------------------------------------------- run
$Failed = @()
try {
  $first = $true
  foreach ($lang in ($Langs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    switch ($lang) {
      'cpp'    { $exe = $CppExe; $cmd = @() ; $missing = if (-not (Test-Path $CppExe)) { 'C++ program not built (no g++/make; run setup_windows.ps1 and install Rtools)' } }
      'python' { $exe = $Python; $cmd = @(Join-Path $Root 'Python\common_benchmark.py'); $missing = if (-not $Python) { 'Python not found' } }
      'r'      { $exe = $Rscript; $cmd = @(Join-Path $Root 'R\common_benchmark.R'); $missing = if (-not $Rscript) { 'Rscript not found' } }
      'js'     { $exe = $Node; $cmd = @('--expose-gc', (Join-Path $Root 'JavaScript\common_benchmark.mjs')); $missing = if (-not $Node) { 'Node.js not found' } }
      default  { Write-Host "unknown language: $lang (use cpp, python, r, js)"; exit 2 }
    }
    if ($missing) { Write-Host "$missing - skipping $lang"; $Failed += $lang; continue }
    Write-Host ''
    Write-Host "=================================================================== $lang"
    if (-not $first -and $ModeArg -ne '--verify' -and $Pause -gt 0) {
      Write-Host "Pausing $Pause s so the CPU can cool down (Windows has no readable CPU temperature)..."
      Start-Sleep -Seconds $Pause
    }
    $first = $false
    Set-Phase $lang
    & $exe @cmd @ProgArgs
    if ($LASTEXITCODE -ne 0) { $Failed += $lang }
    Set-Phase 'idle'
  }

  Start-Sleep -Seconds 2   # one more idle sensor sample
  Write-Host ''
  Write-Host "System:  $(Join-Path $Out "system_$Batch.csv")"
  Write-Host "Sensors: $Sensors"
  if ($ModeArg -eq '--verify' -and $Python) {
    & $Python (Join-Path $Root 'common\verify.py') --batch $Batch --dir $Out
  }
} finally {
  Stop-Sensors
}
# The highest GPU clock the sensor log saw (a GPU can boost above the clock the driver reports).
$gpuMax = (Import-Csv $Sensors | Where-Object { $_.gpu_sclk_mhz -match '^\d+$' } | ForEach-Object { [int]$_.gpu_sclk_mhz } |
           Measure-Object -Maximum).Maximum
if ($gpuMax) { Add-Content -LiteralPath (Join-Path $Out "system_$Batch.csv") -Value "gpu_clock_max_logged_mhz,`"$gpuMax`"" -Encoding ascii }
if ($Failed.Count) { Write-Host "Finished with errors in: $($Failed -join ' ')"; exit 1 }
Write-Host 'Done. Open analysis.ipynb to explore the results.'
