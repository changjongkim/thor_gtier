NVCC ?= nvcc
ARCH ?= sm_110
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)
LDLIBS ?= -luring

all: gtier

gtier: gtier.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f gtier
