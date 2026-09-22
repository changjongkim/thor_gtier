NVCC ?= nvcc
ARCH ?= sm_110
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)

all: mmaptest

mmaptest: mmaptest.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f mmaptest
