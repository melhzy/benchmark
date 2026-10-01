# windows_tools.ps1 -- shared helpers for setup_windows.ps1 and run_all.ps1 (Windows PowerShell 5.1+).
# Dot-source it:  . "$PSScriptRoot\common\windows_tools.ps1"
#
# Finds the tools each language needs (BENCH_* environment variables first, then PATH, then the usual
# install folders) and computes the machine id the same way as the four benchmark programs (SPEC.md).

$BenchRoot = Split-Path -Parent $PSScriptRoot
$DepsDir   = Join-Path $BenchRoot 'deps'

function Get-FirstExisting([string[]]$Paths) {
  foreach ($p in $Paths) { if ($p -and (Test-Path -LiteralPath $p)) { return (Resolve-Path -LiteralPath $p).Path } }
  return $null
}

function Get-VersionKey([string]$Name) {
  # The first number in a name as a [version]: "R-4.10.1" -> 4.10.1, "rtools45" -> 45.0.
  $v = [regex]::Match($Name, '\d+(\.\d+){0,3}').Value
  if (-not $v) { return [version]'0.0' }
  if ($v -notmatch '\.') { $v += '.0' }
  return [version]$v
}

function Get-NewestDir([string]$Pattern) {
  # Newest folder matching a wildcard, by the version number in its name.
  $dirs = @(Get-ChildItem -Path $Pattern -Directory -ErrorAction SilentlyContinue)
  if (-not $dirs.Count) { return $null }
  return ($dirs | Sort-Object -Property @{ Expression = { Get-VersionKey $_.Name } } | Select-Object -Last 1).FullName
}

function Get-CommandPath([string]$Name) {
  # A command on PATH, ignoring the Microsoft Store "app execution alias" stubs (python.exe that opens the Store).
  foreach ($c in @(Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)) {
    if ($c.Source -notmatch '\\WindowsApps\\') { return $c.Source }
  }
  return $null
}

function Find-Python {
  if ($env:BENCH_PYTHON) { return $env:BENCH_PYTHON }
  $p = Get-CommandPath 'python'
  if ($p) { return $p }
  $py = Get-CommandPath 'py'
  if ($py) {
    $exe = & $py -3 -c 'import sys; print(sys.executable)' 2>$null
    if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
  }
  return $null
}

function Find-Rscript {
  if ($env:BENCH_RSCRIPT) { return $env:BENCH_RSCRIPT }
  $p = Get-CommandPath 'Rscript'
  if ($p) { return $p }
  foreach ($key in 'HKLM:\SOFTWARE\R-core\R', 'HKCU:\SOFTWARE\R-core\R') {
    $dir = (Get-ItemProperty -Path $key -Name InstallPath -ErrorAction SilentlyContinue).InstallPath
    if ($dir -and (Test-Path (Join-Path $dir 'bin\Rscript.exe'))) { return (Join-Path $dir 'bin\Rscript.exe') }
  }
  $dir = Get-NewestDir "$env:ProgramFiles\R\R-*"
  if ($dir) { return Get-FirstExisting @("$dir\bin\Rscript.exe") }
  return $null
}

function Find-Node {
  $p = Get-CommandPath 'node'
  if ($p) { return $p }
  return Get-FirstExisting @("$env:ProgramFiles\nodejs\node.exe", "$env:LOCALAPPDATA\Programs\nodejs\node.exe")
}

# The C++ toolchain: MinGW-w64 g++ plus a make. Rtools (installed with R, https://cran.r-project.org/bin/windows/Rtools/)
# provides both; MSYS2 (UCRT64) works too. Returns @{ Gxx; Make; Path } (Path = folders to put first on PATH), or $null.
function Find-Toolchain {
  $gxx = Get-CommandPath 'g++'
  $make = Get-CommandPath 'make'
  if (-not $make) { $make = Get-CommandPath 'mingw32-make' }
  if ($gxx -and $make) { return @{ Gxx = $gxx; Make = $make; Path = @((Split-Path $gxx), (Split-Path $make)) } }
  $candidates = @()
  $rtools = Get-NewestDir 'C:\rtools4*'
  if ($rtools) { $candidates += , @("$rtools\x86_64-w64-mingw32.static.posix\bin", "$rtools\usr\bin") }
  $candidates += , @('C:\msys64\ucrt64\bin', 'C:\msys64\usr\bin')
  $candidates += , @('C:\msys64\mingw64\bin', 'C:\msys64\usr\bin')
  foreach ($c in $candidates) {
    $g = Get-FirstExisting @("$($c[0])\g++.exe")
    $m = Get-FirstExisting @("$($c[1])\make.exe", "$($c[0])\mingw32-make.exe")
    if ($g -and $m) { return @{ Gxx = $g; Make = $m; Path = @($c[0], $c[1]) } }
  }
  return $null
}

# OpenBLAS DLL (C++ links it, JavaScript loads it): BENCH_OPENBLAS, else deps\openblas (setup_windows.ps1), else common
# install folders. R uses whatever BLAS R itself ships with; NumPy brings its own OpenBLAS.
function Find-OpenBlas {
  if ($env:BENCH_OPENBLAS) { return $env:BENCH_OPENBLAS }
  return Get-FirstExisting @("$DepsDir\openblas\bin\libopenblas.dll", 'C:\OpenBLAS\bin\libopenblas.dll',
                             'C:\msys64\ucrt64\bin\libopenblas.dll', 'C:\msys64\mingw64\bin\libopenblas.dll')
}

# OpenCL SDK folder with include\CL\cl.h and lib\OpenCL.lib (for building the C++ program and R's OpenCL package).
function Find-OpenClSdk {
  if ($env:BENCH_OPENCL_SDK) { return $env:BENCH_OPENCL_SDK }
  foreach ($d in @("$DepsDir\opencl-sdk", $env:OCL)) {
    if ($d -and (Test-Path (Join-Path $d 'include\CL\cl.h')) -and (Test-Path (Join-Path $d 'lib\OpenCL.lib'))) {
      return (Resolve-Path $d).Path
    }
  }
  return $null
}

function Get-Slug([string]$Text) {
  return ($Text.ToLowerInvariant() -replace '[^a-z0-9]+', '-').Trim('-')
}

# Machine id (SPEC.md): BENCH_MACHINE, else the firmware (SMBIOS) vendor + product name, else the computer name.
# The vendor is left out when the product name already starts with it ("Alienware" + "Alienware m16 R1").
function Get-MachineId {
  if ($env:BENCH_MACHINE) { return (Get-Slug $env:BENCH_MACHINE) }
  $bios = Get-ItemProperty -Path 'HKLM:\HARDWARE\DESCRIPTION\System\BIOS' -ErrorAction SilentlyContinue
  $vendor = Get-Slug "$($bios.SystemManufacturer)"
  $product = Get-Slug "$($bios.SystemProductName)"
  $slug = if ($vendor -and ($product -eq $vendor -or $product.StartsWith("$vendor-"))) { $product }
          else { Get-Slug "$vendor $product" }
  if (-not $slug) { $slug = Get-Slug $env:COMPUTERNAME }
  return $slug
}
