IDIR = ../include
CC   = gcc
CXX  = g++
CFLAGS = -I$(IDIR) -std=gnu11 -O3 -Wall
CXXFLAGS = -I$(IDIR) -std=c++17 -O3 -Wall -Dcimg_display=0 -Dcimg_use_tiff

ODIR = obj

.DEFAULT_GOAL := main

_DEPS = pi.h util.h file.h extract.h grad.h tiff_io.h
DEPS  = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o util.o extract.o grad.o tiff_io.o
OBJ  = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS) | $(ODIR)
	$(CC) -fopenmp -c -o $@ $< $(CFLAGS)

$(ODIR)/tiff_io.o: tiff_io.cpp $(IDIR)/tiff_io.h $(IDIR)/CImg.h | $(ODIR)
	$(CXX) -fopenmp -c -o $@ $< $(CXXFLAGS)

main: $(OBJ)
	$(CC) -fopenmp -o $@ $^ $(CFLAGS) -lm -ltiff -lstdc++

$(ODIR):
	mkdir -p $(ODIR)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core main
