IDIR = ../include
CC   = gcc
CFLAGS = -I$(IDIR) -std=gnu11 -O2 -Wall

ODIR = obj

_DEPS = pi.h util.h file.h extract.h grad.h stb_image.h stb_image_write.h
DEPS  = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = main.o util.o extract.o grad.o
OBJ  = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS) | $(ODIR)
	$(CC) -fopenmp -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CC) -fopenmp -o $@ $^ $(CFLAGS) -lm

$(ODIR):
	mkdir -p $(ODIR)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core main
