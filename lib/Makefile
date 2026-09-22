NVCC ?= nvcc
ARCH ?= sm_110
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fPIC
LDLIBS ?= -luring -lcufile
NVCCFLAGS += -I/usr/local/cuda-13.0/targets/sbsa-linux/include -L/usr/local/cuda-13.0/targets/sbsa-linux/lib

all: libgtier.a gtier_bench

libgtier.a: gtier.o
	ar rcs $@ $^

gtier.o: gtier.cu gtier.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

gtier_bench: bench.cu libgtier.a
	$(NVCC) $(NVCCFLAGS) -o $@ bench.cu libgtier.a $(LDLIBS)

clean:
	rm -f *.o libgtier.a gtier_bench
