NVCC ?= nvcc
ARCH ?= sm_110
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -Xcompiler "-O3 -march=native"

all: bw

bw: bw.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f bw
