# Host: gcc/g++ (OpenMP via -fopenmp / libgomp).
# CUDA: nvc++ (NVHPC); nvcc is not required.
#
# After `module load nvhpc`, nvc++ is on PATH. CUDA toolkit layout is typically:
#   .../Linux_x86_64/<ver>/compilers/bin/nvc++
#   .../Linux_x86_64/<ver>/cuda/{include,lib64}
# If the module sets CUDA_HOME, that value is kept (see ?= below).

IDIR = ../include
CC   = gcc
CXX  = g++

NVCXX := $(shell command -v nvc++ 2>/dev/null)
ifeq ($(strip $(NVCXX)),)
$(error nvc++ not found. Load NVHPC first, e.g.: module load nvhpc)
endif

# Prefer module-provided CUDA_HOME; else infer from nvc++ install tree.
CUDA_HOME ?= $(abspath $(dir $(NVCXX))/../../cuda)

CFLAGS = -I$(IDIR) -std=gnu11 -O3 -Wall
CXXFLAGS = -I$(IDIR) -std=c++17 -O3 -Wall -Dcimg_display=0 -Dcimg_use_tiff

# nvc++ builds CUDA C++ translation units; -cuda enables CUDA compilation/linking.
NVCXX_CUDA_FLAGS = -cuda -O3 -std=c++17 -Wall -I$(IDIR) -I$(CUDA_HOME)/include

ODIR = obj

.DEFAULT_GOAL := main

_DEPS = pi.h util.h file.h extract.h grad.h tiff_io.h unwrap_cuda.h
DEPS  = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o util.o extract.o grad.o tiff_io.o unwrap_cuda.o
OBJ  = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS) | $(ODIR)
	$(CC) -fopenmp -c -o $@ $< $(CFLAGS)

$(ODIR)/tiff_io.o: tiff_io.cpp $(IDIR)/tiff_io.h $(IDIR)/CImg.h | $(ODIR)
	$(CXX) -fopenmp -c -o $@ $< $(CXXFLAGS)

$(ODIR)/unwrap_cuda.o: unwrap_cuda.cu $(IDIR)/unwrap_cuda.h | $(ODIR)
	$(NVCXX) $(NVCXX_CUDA_FLAGS) -c -o $@ $<

main: $(OBJ)
	$(CC) -fopenmp -o $@ $^ $(CFLAGS) -lm -ltiff -lstdc++ \
		-L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart

$(ODIR):
	mkdir -p $(ODIR)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core main
