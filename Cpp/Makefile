# Builds the C++ implementation of the common benchmark (see ../common/SPEC.md).
#   make                               build ./common_benchmark
#   make OPENBLAS=/path/to/libopenblas.so.0   use a specific OpenBLAS library
#   make clean                         remove it
#
# OpenBLAS: $(OPENBLAS) if given, else $BENCH_OPENBLAS, else the first library found in the usual
# Debian/Ubuntu, Fedora and Arch locations. If none is found, the build continues without BLAS and
# matmul_blas reports "skipped".
# GPU support (OpenCL) is compiled in only when /usr/include/CL/cl.h exists
# (packages opencl-headers + ocl-icd-libopencl1). Without it the GPU tests report "skipped".
# Re-running make after installing either one rebuilds with it.

CXX      := g++
CXXFLAGS := -O3 -march=native -ffp-contract=off -std=c++20 -pthread

OPENBLAS_CANDIDATES := \
  /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0 \
  /usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
  /usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblas.so.0 \
  /usr/lib64/libopenblasp.so.0 \
  /usr/lib64/libopenblas.so.0 \
  /usr/lib/libopenblas.so.0 \
  /usr/lib/libopenblas.so

ifeq ($(origin OPENBLAS),undefined)
  ifneq ($(BENCH_OPENBLAS),)
    OPENBLAS := $(BENCH_OPENBLAS)
  else
    OPENBLAS := $(firstword $(wildcard $(OPENBLAS_CANDIDATES)))
  endif
endif

ifneq ($(OPENBLAS),)
  LDLIBS += $(OPENBLAS) -Wl,-rpath,$(dir $(OPENBLAS))
else
  CPPFLAGS += -DNO_BLAS
endif

ifneq ($(wildcard /usr/include/CL/cl.h),)
  GPU    := opencl
  LDLIBS += -l:libOpenCL.so.1
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

common_benchmark: common_benchmark.cpp Makefile .build-config
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f common_benchmark .build-config .build-gpu-*

.PHONY: clean
