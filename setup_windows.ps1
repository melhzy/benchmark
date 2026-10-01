<#
setup_windows.ps1 -- install what the benchmark needs on Windows, as far as it can be done without admin rights.

Usage (from the repository folder, in PowerShell or cmd):
  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
  powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -SkipPython -SkipR

What it does (each step is skipped when already done):
  1. downloads OpenBLAS (Windows x64 build) into deps\openblas    - used by C++ and JavaScript for matmul_blas
  2. downloads the Khronos OpenCL SDK into deps\opencl-sdk         - headers + import library for the GPU tests
  3. pip install -r requirements.txt into the Python found          - NumPy, pyopencl, pandas, matplotlib, jupyterlab
  4. npm install in JavaScript\                                     - webgpu, koffi
  5. builds R's OpenCL package from CRAN source with Rtools         - there is no Windows binary of it on CRAN

Install these yourself first (all have normal Windows installers):
  - R           https://cran.r-project.org/bin/windows/base/
  - Rtools      https://cran.r-project.org/bin/windows/Rtools/  (g++ and make for the C++ program and R's OpenCL package)
  - Python 3    https://www.python.org/downloads/windows/
  - Node.js 20+ https://nodejs.org/
  - a GPU driver with OpenCL (NVIDIA, AMD and Intel drivers include it)
deps\ is ignored by git. Run .\run_all.cmd -Check afterwards to see what was found.
#>
param(
  [switch]$SkipPython,
  [switch]$SkipNode,
  [switch]$SkipR
)
# 'Continue', not 'Stop': Windows PowerShell 5.1 would turn every stderr line of R, pip or npm into a fatal error.
$ErrorActionPreference = 'Continue'
. "$PSScriptRoot\common\windows_tools.ps1"

$OpenBlasVersion = '0.3.34'
$OpenClSdkVersion = 'v2026.05.29'
$Cran = 'https://cran.r-project.org'

function Step([string]$Text) { Write-Host ''; Write-Host "== $Text" }

function Get-Archive([string]$Url, [string]$Zip) {
  if (-not (Test-Path $Zip)) {
    Write-Host "  downloading $Url"
    & curl.exe -fsSL --retry 3 -o $Zip $Url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $Url" }
  }
}

function Expand-Zip([string]$Zip, [string]$Dest) {
  New-Item -ItemType Directory -Force $Dest | Out-Null
  & tar.exe -xf $Zip -C $Dest          # bsdtar, part of Windows 10/11; much faster than Expand-Archive
  if ($LASTEXITCODE -ne 0) { Expand-Archive -Path $Zip -DestinationPath $Dest -Force }
}

New-Item -ItemType Directory -Force $DepsDir | Out-Null

Step "OpenBLAS $OpenBlasVersion"
$blasDll = Join-Path $DepsDir 'openblas\bin\libopenblas.dll'
if (Test-Path $blasDll) {
  Write-Host "  already installed: $blasDll"
} else {
  $zip = Join-Path $DepsDir "OpenBLAS-$OpenBlasVersion-x64.zip"
  Get-Archive "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$OpenBlasVersion/OpenBLAS-$OpenBlasVersion-x64.zip" $zip
  Expand-Zip $zip (Join-Path $DepsDir 'openblas')
  Write-Host "  installed: $blasDll"
}

Step "Khronos OpenCL SDK $OpenClSdkVersion"
$sdk = Join-Path $DepsDir 'opencl-sdk'
if (Test-Path (Join-Path $sdk 'include\CL\cl.h')) {
  Write-Host "  already installed: $sdk"
} else {
  $name = "OpenCL-SDK-$OpenClSdkVersion-Win-x64"
  $zip = Join-Path $DepsDir "$name.zip"
  Get-Archive "https://github.com/KhronosGroup/OpenCL-SDK/releases/download/$OpenClSdkVersion/$name.zip" $zip
  Expand-Zip $zip $DepsDir
  if (Test-Path $sdk) { Remove-Item -Recurse -Force $sdk }
  Rename-Item (Join-Path $DepsDir $name) 'opencl-sdk'
  Write-Host "  installed: $sdk"
}

if (-not $SkipPython) {
  Step 'Python packages (requirements.txt)'
  $python = Find-Python
  if (-not $python) {
    Write-Warning 'Python not found - install Python 3, or set BENCH_PYTHON to python.exe'
  } else {
    Write-Host "  $python"
    & $python -m pip install --disable-pip-version-check -r (Join-Path $BenchRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Write-Warning 'pip install failed' }
  }
}

if (-not $SkipNode) {
  Step 'npm packages (JavaScript\package.json)'
  $node = Find-Node
  if (-not $node) {
    Write-Warning 'Node.js not found - install Node.js 20 or newer'
  } else {
    $npm = Join-Path (Split-Path $node) 'npm.cmd'
    Push-Location (Join-Path $BenchRoot 'JavaScript')
    try { & $npm install --no-fund --no-audit } finally { Pop-Location }
  }
}

function Test-ROpenCl([string]$Rscript) {
  # (R code in single quotes only: Windows PowerShell drops double quotes inside native-command arguments.)
  & $Rscript -e "quit(status = !requireNamespace('OpenCL', quietly = TRUE))" 2>$null
  return $LASTEXITCODE -eq 0
}

if (-not $SkipR) {
  Step "R package 'OpenCL' (built from source)"
  $rscript = Find-Rscript
  $tc = Find-Toolchain
  if (-not $rscript) {
    Write-Warning 'R not found - install R, or set BENCH_RSCRIPT to Rscript.exe'
  } elseif (Test-ROpenCl $rscript) {
    Write-Host '  already installed'
  } elseif (-not $tc) {
    Write-Warning 'Rtools not found - install Rtools (it matches your R version) to build the OpenCL package'
  } else {
    # The package's configure.win wants OCL (SDK root), OCLINC (compiler flag) and OCL64LIB (import library).
    $fwd = $sdk -replace '\\', '/'
    $env:OCL = $fwd
    $env:OCLINC = "-I$fwd/include"
    $env:OCL64LIB = "$fwd/lib/OpenCL.lib"
    $env:PATH = (($tc.Path + $env:PATH.Split(';')) -join ';')
    & $rscript -e "install.packages('OpenCL', repos = '$Cran', type = 'source')"
    if (Test-ROpenCl $rscript) { Write-Host '  installed' }
    else { Write-Warning "building R's OpenCL package failed (R's GPU tests will be skipped)" }
  }
}

Write-Host ''
Write-Host 'Done. Check the result with:  .\run_all.cmd -Check'
