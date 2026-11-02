NVCC ?= nvcc
# Default: build for the GPU in this machine. Override e.g. `make ARCH=sm_86`.
ARCH ?= native
NVCCFLAGS ?= -O3 -lineinfo -std=c++17

.PHONY: all clean
all: build/mm

build/mm: src/mm.cu
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

clean:
	rm -rf build
