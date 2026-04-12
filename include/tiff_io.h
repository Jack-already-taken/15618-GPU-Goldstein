/*
 * C++ implementation (tiff_io.cpp) using CImg + libtiff.
 * Exposes a C-callable API for main.c.
 */
#ifndef TIFF_IO_H
#define TIFF_IO_H

#ifdef __cplusplus
extern "C" {
#endif

/* Load first channel (or average channels) as float32. Returns malloc'd buffer row-major. */
float *tiff_io_load_float(const char *path, int *width, int *height);

/*
 * Save single-channel float32 TIFF (e.g. unwrapped phase in radians).
 * Returns 0 on success, non-zero on failure.
 */
int tiff_io_save_float(const char *path, const float *data, int width, int height);

/* Save single-channel 8-bit TIFF (0 / 255 visualization). */
int tiff_io_save_u8(const char *path, const unsigned char *data, int width, int height);

#ifdef __cplusplus
}
#endif

#endif /* TIFF_IO_H */
