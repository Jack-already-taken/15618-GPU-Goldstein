/*
 * Float / uint8 TIFF I/O via CImg (libtiff backend).
 * #define cimg_display 0 avoids any X11 dependency.
 */
#define cimg_display 0
#include "CImg.h"
#include "tiff_io.h"

#include <cstdlib>
#include <cstring>

using namespace cimg_library;

extern "C" float *tiff_io_load_float(const char *path, int *width, int *height)
{
    if (!path || !width || !height)
        return nullptr;

    CImg<float> img;
    try {
        img.load_tiff(path);
    } catch (...) {
        return nullptr;
    }

    const int w = static_cast<int>(img.width());
    const int h = static_cast<int>(img.height());
    if (w <= 0 || h <= 0)
        return nullptr;

    const int c = static_cast<int>(img.spectrum());
    const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
    float *const buf = static_cast<float *>(std::malloc(n * sizeof(float)));
    if (!buf)
        return nullptr;

    if (c <= 1) {
        cimg_forXY(img, x, y) { buf[static_cast<size_t>(y) * w + x] = img(x, y, 0, 0); }
    } else {
        const float invc = 1.0f / static_cast<float>(c);
        cimg_forXY(img, x, y) {
            float s = 0.0f;
            for (int k = 0; k < c; ++k)
                s += img(x, y, 0, k);
            buf[static_cast<size_t>(y) * w + x] = s * invc;
        }
    }

    *width = w;
    *height = h;
    return buf;
}

extern "C" int tiff_io_save_float(const char *path, const float *data, int width, int height)
{
    if (!path || !data || width <= 0 || height <= 0)
        return -1;

    CImg<float> img(static_cast<unsigned int>(width),
                    static_cast<unsigned int>(height),
                    1, 1, 0.0f);
    const int w = width;
    cimg_forXY(img, x, y) {
        img(x, y, 0, 0) = data[static_cast<size_t>(y) * w + x];
    }
    try {
        img.save_tiff(path);
    } catch (...) {
        return -2;
    }
    return 0;
}

extern "C" int tiff_io_save_u8(const char *path, const unsigned char *data, int width, int height)
{
    if (!path || !data || width <= 0 || height <= 0)
        return -1;

    CImg<unsigned char> img(static_cast<unsigned int>(width),
                             static_cast<unsigned int>(height),
                             1, 1, 0);
    const int w = width;
    cimg_forXY(img, x, y) {
        img(x, y, 0, 0) = data[static_cast<size_t>(y) * w + x];
    }
    try {
        img.save_tiff(path);
    } catch (...) {
        return -2;
    }
    return 0;
}
