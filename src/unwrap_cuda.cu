#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#include "unwrap_cuda.h"
#include "pi.h"

namespace {

enum : unsigned char { kPosRes = 0x01, kNegRes = 0x02, kBorder = 0x20, kBranchCut = 0x10 };

__device__ __forceinline__ float device_gradient(float p1, float p2)
{
    float r = p1 - p2;
    if (r > (float)PI)
        r -= (float)TWOPI;
    else if (r < -(float)PI)
        r += (float)TWOPI;
    return r;
}

__global__ void k_identify_residues(const float *phase, unsigned char *bitflags, int xsize,
                                    int ysize, int *d_num_res)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize - 1 || j >= ysize - 1)
        return;

    const int                k = j * xsize + i;
    constexpr unsigned char avoid = kBranchCut | kBorder;
    if ((bitflags[k] & avoid) || (bitflags[k + 1] & avoid)
        || (bitflags[k + 1 + xsize] & avoid) || (bitflags[k + xsize] & avoid))
        return;

    const float r = device_gradient(phase[k + 1], phase[k])
                    + device_gradient(phase[k + 1 + xsize], phase[k + 1])
                    + device_gradient(phase[k + xsize], phase[k + 1 + xsize])
                    + device_gradient(phase[k], phase[k + xsize]);

    const float thr = static_cast<float>(RESIDUE_THRESHOLD);
    if (r > thr)
        bitflags[k] |= kPosRes;
    else if (r < -thr)
        bitflags[k] |= kNegRes;
    if (r * r > thr * thr)
        atomicAdd(d_num_res, 1);
}

__global__ void k_noop(void) {}

static int cuda_fail(cudaError_t e, const char *msg)
{
    if (e == cudaSuccess)
        return 0;
    fprintf(stderr, "unwrap_cuda: %s: %s\n", msg, cudaGetErrorString(e));
    return (int)e;
}

static dim3 residue_grid(int xsize, int ysize)
{
    constexpr int bx = 16, by = 16;
    return dim3((xsize + bx - 2) / bx, (ysize + by - 2) / by);
}

} /* namespace */

extern "C" int unwrap_cuda_init(void)
{
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess)
        return (int)e;
    if (n <= 0)
        return (int)cudaErrorNoDevice;
    if ((e = cudaSetDevice(0)) != cudaSuccess)
        return (int)e;
    return cuda_fail(cudaDeviceSynchronize(), "init sync");
}

extern "C" int unwrap_cuda_device_bufs_alloc(int length, UnwrapCudaDeviceBufs *out)
{
    cudaError_t e;

    if (!out || length < 1)
        return -1;
    memset(out, 0, sizeof(*out));

    e = cudaMalloc((void **)&out->d_phase, (size_t)length * sizeof(float));
    if (e != cudaSuccess)
        return (int)e;
    e = cudaMalloc((void **)&out->d_bitflags, (size_t)length * sizeof(unsigned char));
    if (e != cudaSuccess) {
        unwrap_cuda_device_bufs_free(out);
        return (int)e;
    }
    e = cudaMalloc((void **)&out->d_soln, (size_t)length * sizeof(float));
    if (e != cudaSuccess) {
        unwrap_cuda_device_bufs_free(out);
        return (int)e;
    }
    e = cudaMalloc((void **)&out->d_residue_count, sizeof(int));
    if (e != cudaSuccess) {
        unwrap_cuda_device_bufs_free(out);
        return (int)e;
    }
    return 0;
}

extern "C" void unwrap_cuda_device_bufs_free(UnwrapCudaDeviceBufs *buf)
{
    if (!buf)
        return;
    cudaFree(buf->d_phase);
    cudaFree(buf->d_bitflags);
    cudaFree(buf->d_soln);
    cudaFree(buf->d_residue_count);
    memset(buf, 0, sizeof(*buf));
}

extern "C" int unwrap_cuda_launch_residue_identification(
    float *h_phase, unsigned char *h_bitflags, const UnwrapCudaDeviceBufs *dev, int xsize, int ysize,
    int length)
{
    if (!h_phase || !h_bitflags || !dev || !dev->d_phase || !dev->d_bitflags || !dev->d_residue_count
        || xsize < 2 || ysize < 2 || length != xsize * ysize)
        return -1;

    cudaError_t e;
    int         h_count = 0;

    if ((e = cudaMemcpy(dev->d_phase, h_phase, (size_t)length * sizeof(float), cudaMemcpyHostToDevice))
        != cudaSuccess)
        return cuda_fail(e, "H2D phase");
    if ((e = cudaMemcpy(dev->d_bitflags, h_bitflags, (size_t)length * sizeof(unsigned char),
                        cudaMemcpyHostToDevice))
        != cudaSuccess)
        return cuda_fail(e, "H2D bitflags");
    if ((e = cudaMemset(dev->d_residue_count, 0, sizeof(int))) != cudaSuccess)
        return cuda_fail(e, "memset count");

    dim3 block(16, 16);
    dim3 grid = residue_grid(xsize, ysize);
    k_identify_residues<<<grid, block>>>(dev->d_phase, dev->d_bitflags, xsize, ysize,
                                         dev->d_residue_count);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return cuda_fail(e, "k_identify_residues");
    if ((e = cudaDeviceSynchronize()) != cudaSuccess)
        return cuda_fail(e, "sync residues");

    if ((e = cudaMemcpy(h_bitflags, dev->d_bitflags, (size_t)length * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost))
        != cudaSuccess)
        return cuda_fail(e, "D2H bitflags");
    if ((e = cudaMemcpy(&h_count, dev->d_residue_count, sizeof(int), cudaMemcpyDeviceToHost))
        != cudaSuccess)
        return cuda_fail(e, "D2H count");

    return h_count;
}

extern "C" void unwrap_cuda_launch_residue_matching(unsigned char *h_bitflags,
                                                    const UnwrapCudaDeviceBufs *dev, int max_cut_len,
                                                    int num_res, int xsize, int ysize, int length)
{
    (void)xsize;
    (void)ysize;
    (void)max_cut_len;
    (void)num_res;
    if (length < 1 || !h_bitflags || !dev || !dev->d_bitflags)
        return;

    if (cudaMemcpy(dev->d_bitflags, h_bitflags, (size_t)length * sizeof(unsigned char),
                   cudaMemcpyHostToDevice)
        != cudaSuccess)
        return;

    constexpr int threads = 256;
    const int blocks = (length + threads - 1) / threads;
    k_noop<<<blocks, threads>>>();
    (void)cudaGetLastError();
    (void)cudaMemcpy(h_bitflags, dev->d_bitflags, (size_t)length * sizeof(unsigned char),
                     cudaMemcpyDeviceToHost);
    (void)cudaDeviceSynchronize();
}

extern "C" void unwrap_cuda_launch_unwrapping(float *h_phase, unsigned char *h_bitflags, float *h_soln,
                                              const UnwrapCudaDeviceBufs *dev, int xsize, int ysize,
                                              int length)
{
    (void)xsize;
    (void)ysize;
    if (length < 1 || !h_phase || !h_bitflags || !h_soln || !dev || !dev->d_phase || !dev->d_bitflags
        || !dev->d_soln)
        return;

    (void)cudaMemcpy(dev->d_phase, h_phase, (size_t)length * sizeof(float), cudaMemcpyHostToDevice);
    (void)cudaMemcpy(dev->d_bitflags, h_bitflags, (size_t)length * sizeof(unsigned char),
                     cudaMemcpyHostToDevice);
    (void)cudaMemcpy(dev->d_soln, h_soln, (size_t)length * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int threads = 256;
    k_noop<<<(length + threads - 1) / threads, threads>>>();
    (void)cudaGetLastError();
    (void)cudaMemcpy(h_soln, dev->d_soln, (size_t)length * sizeof(float), cudaMemcpyDeviceToHost);
    (void)cudaDeviceSynchronize();
}
