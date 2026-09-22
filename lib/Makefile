NVCC ?= nvcc
ARCH ?= sm_110
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fPIC
LDLIBS ?= -luring

all: libgtier.a gtier_demo

libgtier.a: gtier.o
	ar rcs $@ $^

gtier.o: gtier.cu gtier.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

gtier_demo: demo.cu libgtier.a
	$(NVCC) $(NVCCFLAGS) -o $@ demo.cu libgtier.a $(LDLIBS)

clean:
	rm -f *.o libgtier.a gtier_demo
