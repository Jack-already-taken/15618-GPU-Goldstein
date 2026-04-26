# Host: gcc/g++ (OpenMP via -fopenmp / libgomp)
# CUDA: nvcc

IDIR = ../include
ODIR = obj

CC   = gcc
CXX  = g++
NVCC := $(shell command -v nvcc 2>/dev/null)

ifeq ($(strip $(NVCC)),)
$(error nvcc not found. Load CUDA first or add nvcc to PATH)
endif

# Prefer environment CUDA_HOME if already set; otherwise infer from nvcc
CUDA_HOME ?= $(abspath $(dir $(NVCC))/..)

CFLAGS   = -I$(IDIR) -std=gnu11 -O3 -Wall -fno-pie
CXXFLAGS = -I$(IDIR) -std=c++17 -O3 -Wall -fno-pie -Dcimg_display=0 -Dcimg_use_tiff
LDFLAGS  = -fopenmp -no-pie
LDLIBS   = -lm -ltiff -lstdc++ -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart

# nvcc flags for CUDA compilation
NVCCFLAGS = -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/g++-11 -I$(IDIR) -I$(CUDA_HOME)/include 

.DEFAULT_GOAL := main

_DEPS = pi.h util.h file.h extract.h grad.h tiff_io.h unwrap_cuda.h
DEPS  = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o util.o extract.o grad.o tiff_io.o unwrap_cuda.o unwrap_cuda_tiled.o
OBJ  = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR):
	mkdir -p $(ODIR)

$(ODIR)/%.o: %.c $(DEPS) | $(ODIR)
	$(CC) -fopenmp $(CFLAGS) -c -o $@ $<

$(ODIR)/tiff_io.o: tiff_io.cpp $(IDIR)/tiff_io.h $(IDIR)/CImg.h | $(ODIR)
	$(CXX) -fopenmp $(CXXFLAGS) -c -o $@ $<

$(ODIR)/unwrap_cuda.o: unwrap_cuda.cu $(IDIR)/unwrap_cuda.h | $(ODIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<
$(ODIR)/unwrap_cuda_tiled.o: unwrap_cuda_tiled.cu $(IDIR)/unwrap_cuda.h | $(ODIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

main: $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core main
