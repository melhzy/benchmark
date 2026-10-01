# Builds the C++ implementation of the common benchmark (see ../common/SPEC.md).
#   make                               build ./common_benchmark (Windows: common_benchmark.exe)
#   make OPENBLAS=/path/to/libopenblas.so.0   use a specific OpenBLAS library
#   make clean                         remove it
#
# Linux. OpenBLAS: $(OPENBLAS) if given, else $BENCH_OPENBLAS, else the first library found in the usual
# Debian/Ubuntu, Fedora and Arch locations. If none is found, the build continues without BLAS and
# matmul_blas reports "skipped".
# GPU support (OpenCL) is compiled in only when /usr/include/CL/cl.h exists
# (packages opencl-headers + ocl-icd-libopencl1). Without it the GPU tests report "skipped".
#
# Windows (MinGW-w64 g++ and make, e.g. from Rtools; run_all.ps1 finds them and runs this Makefile).
# OpenBLAS: $(OPENBLAS), else $BENCH_OPENBLAS, else ../deps/openblas/bin/libopenblas.dll (setup_windows.ps1);
# the DLL is copied next to the program so Windows finds it. OpenCL: headers and import library from
# $(OPENCL_SDK), else $BENCH_OPENCL_SDK, else ../deps/opencl-sdk; the driver's OpenCL.dll is used at run time.
#
# Re-running make after installing either one rebuilds with it.

CXX      := g++
CXXFLAGS := -O3 -march=native -ffp-contract=off -std=c++20 -pthread

ifeq ($(OS),Windows_NT)
  EXE := .exe
  # GCC on Windows cannot align the stack to 32 bytes, so AVX register spills must use unaligned moves
  # (otherwise -march=native code can crash); the assembler rewrites them.
  CXXFLAGS += -Wa,-muse-unaligned-vector-move
  OPENBLAS_CANDIDATES := ../deps/openblas/bin/libopenblas.dll
  ifeq ($(origin OPENCL_SDK),undefined)
    OPENCL_SDK := $(if $(BENCH_OPENCL_SDK),$(subst \,/,$(BENCH_OPENCL_SDK)),../deps/opencl-sdk)
  endif
  CL_HEADER := $(wildcard $(OPENCL_SDK)/include/CL/cl.h)
else
  EXE :=
  OPENBLAS_CANDIDATES := \
    /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0 \
    /usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
    /usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblas.so.0 \
    /usr/lib64/libopenblasp.so.0 \
    /usr/lib64/libopenblas.so.0 \
    /usr/lib/libopenblas.so.0 \
    /usr/lib/libopenblas.so
  CL_HEADER := $(wildcard /usr/include/CL/cl.h)
endif
TARGET := common_benchmark$(EXE)

ifeq ($(origin OPENBLAS),undefined)
  ifneq ($(BENCH_OPENBLAS),)
    OPENBLAS := $(subst \,/,$(BENCH_OPENBLAS))
  else
    OPENBLAS := $(firstword $(wildcard $(OPENBLAS_CANDIDATES)))
  endif
endif

EXTRA :=
ifneq ($(OPENBLAS),)
  ifeq ($(OS),Windows_NT)
    LDLIBS += $(OPENBLAS)
    EXTRA  += $(notdir $(OPENBLAS))
  else
    LDLIBS += $(OPENBLAS) -Wl,-rpath,$(dir $(OPENBLAS))
  endif
else
  CPPFLAGS += -DNO_BLAS
endif

ifneq ($(CL_HEADER),)
  GPU := opencl
  ifeq ($(OS),Windows_NT)
    CPPFLAGS += -I$(OPENCL_SDK)/include
    LDLIBS   += $(OPENCL_SDK)/lib/OpenCL.lib
  else
    LDLIBS += -l:libOpenCL.so.1
  endif
else
  GPU    := none
  CPPFLAGS += -DNO_OPENCL
endif
CPPFLAGS += -DBENCH_BUILD='"$(CXX) $(CXXFLAGS)"'

# The build configuration is kept in .build-config; when it changes (OpenBLAS or OpenCL appears,
# disappears or moves), the file is rewritten, which makes the binary rebuild.
CONFIG     := gpu=$(GPU) blas=$(if $(OPENBLAS),$(OPENBLAS),none)
OLD_CONFIG := $(shell cat .build-config 2>/dev/null)
ifneq ($(CONFIG),$(OLD_CONFIG))
  $(shell rm -f .build-gpu-*; echo '$(CONFIG)' > .build-config)
endif

all: $(TARGET) $(EXTRA)

$(TARGET): common_benchmark.cpp Makefile .build-config
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDLIBS)

# Windows: the OpenBLAS DLL next to the program (Windows looks for DLLs in the program's folder).
ifneq ($(EXTRA),)
$(EXTRA): $(OPENBLAS) .build-config
	cp -f $(OPENBLAS) $@
endif

clean:
	rm -f common_benchmark common_benchmark.exe *.dll .build-config .build-gpu-*

.PHONY: all clean
