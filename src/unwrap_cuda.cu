#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#include "unwrap_cuda.h"
#include "pi.h"

namespace {

constexpr unsigned char kUnwrapped  = 0x40;

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

__device__ __forceinline__ bool claim_pixel(unsigned char *flags, int idx)
{
    unsigned int *word = (unsigned int*)(flags + (idx & ~3));
    unsigned int bit   = (unsigned int)kUnwrapped << ((idx & 3) * 8);
    unsigned int old   = atomicOr(word, bit);
    return !(old & bit);
}

// __global__ void k_identify_residues(const float *phase, unsigned char *bitflags, int xsize,
//                                     int ysize, int *d_num_res)
// {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     const int j = blockIdx.y * blockDim.y + threadIdx.y;
//     if (i >= xsize - 1 || j >= ysize - 1)
//         return;

//     const int                k = j * xsize + i;
//     constexpr unsigned char avoid = kBranchCut | kBorder;
//     if ((bitflags[k] & avoid) || (bitflags[k + 1] & avoid)
//         || (bitflags[k + 1 + xsize] & avoid) || (bitflags[k + xsize] & avoid))
//         return;

//     const float r = device_gradient(phase[k + 1], phase[k])
//                     + device_gradient(phase[k + 1 + xsize], phase[k + 1])
//                     + device_gradient(phase[k + xsize], phase[k + 1 + xsize])
//                     + device_gradient(phase[k], phase[k + xsize]);

//     const float thr = static_cast<float>(RESIDUE_THRESHOLD);
//     if (r > thr)
//         bitflags[k] |= kPosRes;
//     else if (r < -thr)
//         bitflags[k] |= kNegRes;
//     if (r * r > thr * thr)
//         atomicAdd(d_num_res, 1);
// }

#define TILE_W 16
#define TILE_H 16

__global__ void k_identify_residues(const float *phase, unsigned char *bitflags,
                                    int xsize, int ysize, int *d_num_res)
{
    __shared__ float s[TILE_H + 1][TILE_W + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i  = blockIdx.x * TILE_W + tx;
    const int j  = blockIdx.y * TILE_H + ty;

    /* Load (TILE_H+1) x (TILE_W+1) patch — every thread loads its own cell,
       edge threads also load the +1 halo column / row */
    if (i < xsize && j < ysize)
        s[ty][tx] = phase[j * xsize + i];

    if (tx == TILE_W - 1 && i + 1 < xsize && j < ysize)
        s[ty][tx + 1] = phase[j * xsize + (i + 1)];

    if (ty == TILE_H - 1 && j + 1 < ysize && i < xsize)
        s[ty + 1][tx] = phase[(j + 1) * xsize + i];

    if (tx == TILE_W - 1 && ty == TILE_H - 1 && i + 1 < xsize && j + 1 < ysize)
        s[ty + 1][tx + 1] = phase[(j + 1) * xsize + (i + 1)];

    __syncthreads();

    if (i >= xsize - 1 || j >= ysize - 1)
        return;

    const int k = j * xsize + i;
    constexpr unsigned char avoid = kBranchCut | kBorder;
    if ((bitflags[k] & avoid) || (bitflags[k + 1] & avoid)
        || (bitflags[k + 1 + xsize] & avoid) || (bitflags[k + xsize] & avoid))
        return;

    const float p00 = s[ty    ][tx    ];
    const float p10 = s[ty    ][tx + 1];
    const float p11 = s[ty + 1][tx + 1];
    const float p01 = s[ty + 1][tx    ];

    const float r = device_gradient(p10, p00)
                  + device_gradient(p11, p10)
                  + device_gradient(p01, p11)
                  + device_gradient(p00, p01);

    const float thr = static_cast<float>(RESIDUE_THRESHOLD);
    if (r > thr)        bitflags[k] |= kPosRes;
    else if (r < -thr)  bitflags[k] |= kNegRes;
    if (r * r > thr * thr)
        atomicAdd(d_num_res, 1);
}

/* -----------------------------------------------------------------------
 * Stage 3 kernel 1 (BFS approach): seed initialisation
 * One thread per pixel. Pixels that are not BORDER/BRANCH_CUT/UNWRAPPED
 * and have at least one valid non-AVOID neighbor are elected seeds.
 * For simplicity we elect every valid pixel and let the BFS handle it —
 * the frontier compaction via atomicAdd keeps this correct.
 * ----------------------------------------------------------------------- */

constexpr unsigned char kAvoid      = kBranchCut | kBorder;

__global__ void k_init_seeds(const float        *phase,
                              unsigned char      *bitflags,
                              float              *soln,
                              int                *frontier,
                              int                *frontier_count,
                              int                 xsize,
                              int                 ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize || j >= ysize) return;

    const int k = j * xsize + i;
    if (bitflags[k] & kAvoid) return;

    /* Elect this pixel as a seed: initialise soln and mark UNWRAPPED */
    soln[k] = phase[k];
    bitflags[k] |= kUnwrapped;
    const int slot = atomicAdd(frontier_count, 1);
    frontier[slot] = k;
}

/* -----------------------------------------------------------------------
 * Stage 3 kernel 2 (BFS approach): one frontier expansion round
 * One thread per pixel in the current frontier (d_in / n_in).
 * Unwrapped neighbors are pushed into d_out via atomicAdd on *n_out.
 * ----------------------------------------------------------------------- */
// __global__ void k_bfs_expand(const float        *phase,
//                               unsigned char      *bitflags,
//                               float              *soln,
//                               const float        *gradx,
//                               const float        *grady,
//                               const int          *d_in,
//                               int                 n_in,
//                               int                *d_out,
//                               int                *n_out,
//                               int                 xsize,
//                               int                 ysize)
// {
//     const int t  = blockIdx.x * blockDim.x + threadIdx.x;
//     if (t >= n_in) return;

//     const int kk    = d_in[t];
//     const int x     = kk % xsize;
//     const int y     = kk / xsize;
//     const float val = soln[kk];
//     atomicOr((unsigned int*)(bitflags) + (kk>>2),
//          (unsigned int)kUnwrapped << ((kk&3)*8));

//     /* left */
//     if (x - 1 >= 0) {
//         const int nb = kk - 1;
//         if (!(bitflags[nb] & (kAvoid | kUnwrapped))) {
//             if (!(atomicOr((unsigned int*)(bitflags) + (nb>>2),
//                            (unsigned int)kUnwrapped << ((nb&3)*8))
//                   & ((unsigned int)kUnwrapped << ((nb&3)*8)))) {
//                 soln[nb] = val + gradx[nb];
//                 d_out[atomicAdd(n_out, 1)] = nb;
//             }
//         }
//     }
//     /* right */
//     if (x + 1 < xsize) {
//         const int nb = kk + 1;
//         if (!(bitflags[nb] & (kAvoid | kUnwrapped))) {
//             if (!(atomicOr((unsigned int*)(bitflags) + (nb>>2),
//                            (unsigned int)kUnwrapped << ((nb&3)*8))
//                   & ((unsigned int)kUnwrapped << ((nb&3)*8)))) {
//                 soln[nb] = val - gradx[kk];
//                 d_out[atomicAdd(n_out, 1)] = nb;
//             }
//         }
//     }
//     /* up */
//     if (y - 1 >= 0) {
//         const int nb = kk - xsize;
//         if (!(bitflags[nb] & (kAvoid | kUnwrapped))) {
//             if (!(atomicOr((unsigned int*)(bitflags) + (nb>>2),
//                            (unsigned int)kUnwrapped << ((nb&3)*8))
//                   & ((unsigned int)kUnwrapped << ((nb&3)*8)))) {
//                 soln[nb] = val + grady[nb];
//                 d_out[atomicAdd(n_out, 1)] = nb;
//             }
//         }
//     }
//     /* down */
//     if (y + 1 < ysize) {
//         const int nb = kk + xsize;
//         if (!(bitflags[nb] & (kAvoid | kUnwrapped))) {
//             if (!(atomicOr((unsigned int*)(bitflags) + (nb>>2),
//                            (unsigned int)kUnwrapped << ((nb&3)*8))
//                   & ((unsigned int)kUnwrapped << ((nb&3)*8)))) {
//                 soln[nb] = val - grady[kk];
//                 d_out[atomicAdd(n_out, 1)] = nb;
//             }
//         }
//     }
// }

__global__ void k_bfs_expand(const float        *phase,
                              unsigned char      *bitflags,
                              float              *soln,
                              const float        *gradx,
                              const float        *grady,
                              const int          *d_in,
                              int                 n_in,
                              int                *d_out,
                              int                *n_out,
                              int                 xsize,
                              int                 ysize)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_in) return;

    const int kk  = d_in[t];
    const int x   = kk % xsize;
    const int y   = kk / xsize;
    const float val = soln[kk];

    /* left */
    if (x - 1 >= 0) {
        const int nb = kk - 1;
        if (!(bitflags[nb] & (kAvoid | kUnwrapped))) {
            if (claim_pixel(bitflags, nb)) {
                soln[nb] = val + gradx[nb];
                d_out[atomicAdd(n_out, 1)] = nb;
            }
        }
    }
    /* right */
    if (x + 1 < xsize) {
        const int nb = kk + 1;
        if (!(bitflags[nb] & (kAvoid | kUnwrapped))) {
            if (claim_pixel(bitflags, nb)) {
                soln[nb] = val - gradx[kk];
                d_out[atomicAdd(n_out, 1)] = nb;
            }
        }
    }
    /* up */
    if (y - 1 >= 0) {
        const int nb = kk - xsize;
        if (!(bitflags[nb] & (kAvoid | kUnwrapped))) {
            if (claim_pixel(bitflags, nb)) {
                soln[nb] = val + grady[nb];
                d_out[atomicAdd(n_out, 1)] = nb;
            }
        }
    }
    /* down */
    if (y + 1 < ysize) {
        const int nb = kk + xsize;
        if (!(bitflags[nb] & (kAvoid | kUnwrapped))) {
            if (claim_pixel(bitflags, nb)) {
                soln[nb] = val - grady[kk];
                d_out[atomicAdd(n_out, 1)] = nb;
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Stage 3 kernel 3: AVOID-band fill (branch cuts + border pixels)
 * Same logic as the CPU serial AVOID pass — run once after BFS converges.
 * One thread per pixel; only acts on pixels with kAvoid set.
 * ----------------------------------------------------------------------- */
__global__ void k_avoid_fill(const float   *phase,
                              unsigned char *bitflags,
                              float         *soln,
                              int            xsize,
                              int            ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= xsize || j < 1 || j >= ysize) return;

    const int k = j * xsize + i;
    if (!(bitflags[k] & kAvoid)) return;

    if (!(bitflags[k - 1] & kAvoid)) {
        soln[k] = soln[k - 1] + device_gradient(phase[k], phase[k - 1]);
    } else if (!(bitflags[k - xsize] & kAvoid)) {
        soln[k] = soln[k - xsize] + device_gradient(phase[k], phase[k - xsize]);
    }
}

__global__ void k_noop(void) {}

static int cuda_fail(cudaError_t e, const char *msg)
{
    if (e == cudaSuccess)
        return 0;
    fprintf(stderr, "unwrap_cuda: %s: %s\n", msg, cudaGetErrorString(e));
    return (int)e;
}

// static dim3 residue_grid(int xsize, int ysize)
// {
//     constexpr int bx = 16, by = 16;
//     return dim3((xsize + bx - 2) / bx, (ysize + by - 2) / by);
// }

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

    e = cudaMalloc((void **)&out->d_gradx, (size_t)length * sizeof(float));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_grady, (size_t)length * sizeof(float));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_frontier_a, (size_t)length * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_frontier_b, (size_t)length * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_frontier_count_a, sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_frontier_count_b, sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

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
    cudaFree(buf->d_gradx);
    cudaFree(buf->d_grady);
    cudaFree(buf->d_frontier_a);
    cudaFree(buf->d_frontier_b);
    cudaFree(buf->d_frontier_count_a);
    cudaFree(buf->d_frontier_count_b);
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

    // dim3 block(16, 16);
    // dim3 grid = residue_grid(xsize, ysize);

    dim3 block(TILE_W, TILE_H);
    dim3 grid((xsize + TILE_W - 1) / TILE_W,
          (ysize + TILE_H - 1) / TILE_H);

    /* --- CUDA event timing --- */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    k_identify_residues<<<grid, block>>>(dev->d_phase, dev->d_bitflags, xsize, ysize,
                                         dev->d_residue_count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    printf("  [GPU] residue kernel only: %.4f ms\n", kernel_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /* --- end timing --- */

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


extern "C" void unwrap_cuda_launch_unwrapping(
    float *h_phase, unsigned char *h_bitflags, float *h_soln,
    float *h_gradx, float *h_grady,
    const UnwrapCudaDeviceBufs *dev,
    int xsize, int ysize, int length)
{
    if (length < 1 || !h_phase || !h_bitflags || !h_soln || !h_gradx || !h_grady
        || !dev || !dev->d_phase || !dev->d_bitflags || !dev->d_soln
        || !dev->d_gradx || !dev->d_grady
        || !dev->d_frontier_a || !dev->d_frontier_b
        || !dev->d_frontier_count_a || !dev->d_frontier_count_b)
        return;

    cudaError_t e;

    /* Seed one pixel per connected component on CPU,
       matching UnwrapAroundCutsFrontier logic exactly */
    constexpr unsigned char kAvoidU = kBranchCut | kBorder;
    int h_frontier_count = 0;
    int *h_frontier_tmp = (int*)malloc((size_t)length * sizeof(int));

    // Find one seed per connected component
for (int k = 0; k < length; k++) {
    if (!(h_bitflags[k] & (kAvoidU | kUnwrapped))) {
        h_soln[k] = h_phase[k];
        h_bitflags[k] |= kUnwrapped;
        h_frontier_tmp[h_frontier_count++] = k;

        // flood-fill to mark rest of this component so outer loop skips them
        int *stk = (int*)malloc((size_t)length * sizeof(int));
        int top = 0;
        stk[top++] = k;
        while (top > 0) {
            int cur = stk[--top];
            int cx = cur % xsize, cy = cur / xsize;
            int nb;
            if (cx > 0)        { nb = cur-1;      if (!(h_bitflags[nb] & (kAvoidU|kUnwrapped))) { h_bitflags[nb] |= kUnwrapped; stk[top++] = nb; } }
            if (cx < xsize-1)  { nb = cur+1;      if (!(h_bitflags[nb] & (kAvoidU|kUnwrapped))) { h_bitflags[nb] |= kUnwrapped; stk[top++] = nb; } }
            if (cy > 0)        { nb = cur-xsize;  if (!(h_bitflags[nb] & (kAvoidU|kUnwrapped))) { h_bitflags[nb] |= kUnwrapped; stk[top++] = nb; } }
            if (cy < ysize-1)  { nb = cur+xsize;  if (!(h_bitflags[nb] & (kAvoidU|kUnwrapped))) { h_bitflags[nb] |= kUnwrapped; stk[top++] = nb; } }
        }
        free(stk);
    }
}

// Reset kUnwrapped everywhere, then re-mark only seeds
for (int k = 0; k < length; k++)
    h_bitflags[k] &= ~kUnwrapped;
for (int i = 0; i < h_frontier_count; i++)
    h_bitflags[h_frontier_tmp[i]] |= kUnwrapped;
    printf("  [GPU] frontier seed count: %d / %d pixels\n", h_frontier_count, length);

    /* H2D transfers — after CPU seeding so bitflags/soln are updated */
    cudaMemcpy(dev->d_phase,    h_phase,    (size_t)length * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_bitflags, h_bitflags, (size_t)length * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_soln,     h_soln,     (size_t)length * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_gradx,    h_gradx,    (size_t)length * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_grady,    h_grady,    (size_t)length * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_frontier_a, h_frontier_tmp,
               (size_t)h_frontier_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_frontier_count_a, &h_frontier_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dev->d_frontier_count_b, 0, sizeof(int));
    free(h_frontier_tmp);

    /* BFS ping-pong loop */
    int  *d_in  = dev->d_frontier_a,  *d_out = dev->d_frontier_b;
    int  *n_in  = dev->d_frontier_count_a, *n_out = dev->d_frontier_count_b;
    int   h_n_in = h_frontier_count;
    int   round  = 0;

    while (h_n_in > 0) {
        cudaMemset(n_out, 0, sizeof(int));

        const int threads = 256;
        const int blocks  = (h_n_in + threads - 1) / threads;
        k_bfs_expand<<<blocks, threads>>>(
            dev->d_phase, dev->d_bitflags, dev->d_soln,
            dev->d_gradx, dev->d_grady,
            d_in, h_n_in, d_out, n_out,
            xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_bfs_expand"); return;
        }
        cudaDeviceSynchronize();

        /* swap ping-pong buffers */
        int *tmp; tmp = d_in;  d_in  = d_out;  d_out = tmp;
                  tmp = n_in;  n_in  = n_out;   n_out = tmp;
        cudaMemcpy(&h_n_in, n_in, sizeof(int), cudaMemcpyDeviceToHost);
        ++round;
    }
    printf("  [GPU] BFS unwrap: %d rounds\n", round);

    /* AVOID-band fill */
    {
        dim3 block(16, 16);
        dim3 grid((xsize + 15) / 16, (ysize + 15) / 16);
        k_avoid_fill<<<grid, block>>>(dev->d_phase, dev->d_bitflags, dev->d_soln,
                                      xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_avoid_fill"); return;
        }
        cudaDeviceSynchronize();
    }

    /* D2H */
    cudaMemcpy(h_soln,     dev->d_soln,     (size_t)length * sizeof(float),         cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bitflags, dev->d_bitflags, (size_t)length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    {
        int truly_bad = 0, twopi_off = 0;
        float twopi = 2.0f * 3.14159265f;
        for (int k = 0; k < length; k++) {
            if (!(h_bitflags[k] & kUnwrapped)) continue;
            float diff = h_soln[k] - h_phase[k];
            float mod = fmodf(fabsf(diff), twopi);
            if (mod > 0.01f && mod < twopi - 0.01f)
                truly_bad++;
            else
                twopi_off++;
        }
        printf("  [GPU] truly_bad=%d  twopi_multiple_off=%d\n", truly_bad, twopi_off);
    }
}
