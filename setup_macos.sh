#!/usr/bin/env bash
# setup_macos.sh -- install what the benchmark needs on macOS (Apple Silicon or Intel), without administrator rights
# beyond what Homebrew itself needs.
#
# Usage (from the repository folder):
#   ./setup_macos.sh                    everything
#   ./setup_macos.sh --skip-r           leave out a part: --skip-python, --skip-node, --skip-r
#
# What it does (each step is skipped when already done):
#   1. checks for the Xcode Command Line Tools (clang++, make, the OpenCL framework's headers)
#   2. brew install openblas clinfo python node    - OpenBLAS for C++ and JavaScript (matmul_blas); clinfo lists
#                                                     the OpenCL devices; Python and Node.js if missing or too old
#   3. creates .venv with Homebrew's Python and pip install -r requirements.txt into it (run_all.sh uses .venv)
#   4. npm install in JavaScript/                   - webgpu (Dawn on Metal), koffi
#   5. R's OpenCL package from CRAN (a macOS binary exists)
#
# Install these yourself first:
#   - Homebrew      https://brew.sh
#   - R for macOS   https://cran.r-project.org/bin/macosx/  (the CRAN installer; Homebrew's r formula works too)
# .venv/ is ignored by git. Run ./run_all.sh --check afterwards to see what was found.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_PYTHON=0 SKIP_NODE=0 SKIP_R=0
for a in "$@"; do
  case "$a" in
    --skip-python) SKIP_PYTHON=1 ;;
    --skip-node)   SKIP_NODE=1 ;;
    --skip-r)      SKIP_R=1 ;;
    -h|--help)     sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
  esac
done
[[ "$(uname -s)" == Darwin ]] || { echo "setup_macos.sh is for macOS (Linux: see README.md, Setup)" >&2; exit 2; }
step() { echo; echo "== $1"; }

step "Xcode Command Line Tools"
if xcode-select -p >/dev/null 2>&1 && command -v clang++ >/dev/null; then
  echo "  found: $(xcode-select -p)"
else
  echo "  missing: run  xcode-select --install  (a system dialog), then run this script again" >&2
  exit 1
fi

step "Homebrew packages"
command -v brew >/dev/null || { echo "  Homebrew not found: install it from https://brew.sh first" >&2; exit 1; }
want=(openblas clinfo)
(( SKIP_PYTHON )) || want+=(python)
if (( ! SKIP_NODE )); then
  v="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
  (( v >= 20 )) || want+=(node)
fi
missing=()
for f in "${want[@]}"; do brew list --versions "$f" >/dev/null 2>&1 || missing+=("$f"); done
if (( ${#missing[@]} )); then
  echo "  brew install ${missing[*]}"
  brew install "${missing[@]}"
else
  echo "  already installed: ${want[*]}"
fi

if (( ! SKIP_PYTHON )); then
  step "Python packages (.venv)"
  PY="$(brew --prefix)/bin/python3"
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "  creating .venv with $("$PY" --version)"
    "$PY" -m venv "$ROOT/.venv"
  fi
  "$ROOT/.venv/bin/python" -m pip install --quiet --upgrade pip
  "$ROOT/.venv/bin/python" -m pip install --quiet -r "$ROOT/requirements.txt"
  echo "  $("$ROOT/.venv/bin/python" --version): $("$ROOT/.venv/bin/python" -c 'import numpy, pyopencl; print("numpy", numpy.__version__, "pyopencl", pyopencl.VERSION_TEXT)')"
fi

if (( ! SKIP_NODE )); then
  step "npm packages (JavaScript/)"
  (cd "$ROOT/JavaScript" && npm install --no-fund --no-audit)
fi

if (( ! SKIP_R )); then
  step "R OpenCL package"
  if ! command -v Rscript >/dev/null; then
    echo "  Rscript not found: install R from https://cran.r-project.org/bin/macosx/ and run this script again" >&2
  elif Rscript -e 'quit(status = !requireNamespace("OpenCL", quietly = TRUE))' >/dev/null 2>&1; then
    echo "  already installed"
  else
    Rscript -e 'install.packages("OpenCL", repos = "https://cloud.r-project.org")'
  fi
fi

step "Done"
echo "  Next: ./run_all.sh --check, then ./run_all.sh --verify and ./run_all.sh"
