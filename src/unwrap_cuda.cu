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

// Encode upper 16 bits as row index, lower 16 bits as column index
__device__ __forceinline__ int  encode_rc(int r, int c) {
    return (r << 16) | (c & 0xFFFF);
}

// Decode upper 16 bits as row index, lower 16 bits as column index
__device__ __forceinline__ void decode_rc(int e, int &r, int &c) {
    r = (e >> 16) & 0xFFFF;
    c =  e        & 0xFFFF;
}

__global__ void k_map_residues(unsigned char *bitflags, int *pos_residues, int *neg_residues, int xsize, int ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize - 1 || j >= ysize - 1)
        return;

    const int k = j * xsize + i;
    const unsigned char b = bitflags[k];
    const int enc = encode_rc(j, i);

    if (b & kPosRes) {
        const int idx = atomicAdd(pos_residues, 1);
        pos_residues[1 + idx] = enc;
    } else if (b & kNegRes) {
        const int idx = atomicAdd(neg_residues, 1);
        neg_residues[1 + idx] = enc;
    }
}

__global__ void k_match_residues(const int* __restrict__ d_minority,
                                 const int* __restrict__ d_majority,
                                 int n_min, int n_maj,
                                 int* __restrict__ d_pairs,
                                 int xsize, int ysize)
{
    __shared__ int s_maj[POS_CHUNK];

    const int tid     = threadIdx.x;
    const int min_idx = blockIdx.x * blockDim.x + tid;

    int  my_enc  = 0, mr = 0, mc = 0;
    int  best_d2  = INT_MAX;
    int  best_enc = -1;
    const bool active = (min_idx < n_min);
    if (active) {
        my_enc = d_minority[1 + min_idx];
        decode_rc(my_enc, mr, mc);
    }

    for (int cs = 0; cs < n_maj; cs += POS_CHUNK) {
        const int clen = min(POS_CHUNK, n_maj - cs);

        /* cooperative load of one chunk of the majority array */
        for (int i = tid; i < clen; i += blockDim.x)
            s_maj[i] = d_majority[1 + cs + i];
        __syncthreads();

        if (active) {
            #pragma unroll 4
            for (int i = 0; i < clen; ++i) {
                int pr, pc;
                decode_rc(s_maj[i], pr, pc);
                const int dr = pr - mr;
                const int dc = pc - mc;
                const int d2 = dr*dr + dc*dc;
                const bool better = (d2 < best_d2);
                best_d2  = better ? d2       : best_d2;
                best_enc = better ? s_maj[i] : best_enc;
            }
        }
        __syncthreads();
    }

    if (active) {
        /* Only reachable when n_maj == 0 (no majority residues at all). */
        if (best_enc < 0)
            best_enc = nearest_edge_enc(mr, mc, xsize, ysize);

        d_pairs[2 * min_idx    ] = my_enc;
        d_pairs[2 * min_idx + 1] = best_enc;
    }
}

__global__ void k_fill_leftovers(const int* __restrict__ d_majority,
                                 int n_min, int n_leftover,
                                 int* __restrict__ d_pairs,
                                 int xsize, int ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_leftover) return;

    const int slot = n_min + i;
    const int enc  = d_majority[1 + slot];
    int r, c;
    decode_rc(enc, r, c);

    d_pairs[2 * slot    ] = enc;
    d_pairs[2 * slot + 1] = nearest_edge_enc(r, c, xsize, ysize);
}

__global__ void k_rasterize_cuts(const int* __restrict__ d_pairs,
                                 int n_pairs,
                                 unsigned char* __restrict__ bitflags,
                                 int xsize, int ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;

    const int a = d_pairs[2 * i    ];
    const int b = d_pairs[2 * i + 1];

    int r0, c0, r1, c1;
    decode_rc(a, r0, c0);
    decode_rc(b, r1, c1);

    /* Bresenham (integer-only, all octants) */
    int dr =  abs(r1 - r0), sr = (r0 < r1) ? 1 : -1;
    int dc = -abs(c1 - c0), sc = (c0 < c1) ? 1 : -1;
    int err = dr + dc;

    int r = r0, c = c0;
    for (;;) {
        if ((unsigned)r < (unsigned)ysize && (unsigned)c < (unsigned)xsize)
            stamp_branch_cut(bitflags, xsize, r, c);
        if (r == r1 && c == c1) break;
        const int e2 = 2 * err;
        if (e2 >= dc) { 
            err += dc; r += sr; 
        }
        if (e2 <= dr) { 
            err += dr; c += sc; 
        }
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
    (void)max_cut_len;
    (void)num_res;

    if (length < 1 || !h_bitflags || !dev || !dev->d_bitflags
        || !dev->d_pos_residues || !dev->d_neg_residues || !dev->d_pairs)
        return;
    if (xsize >= 65536 || ysize >= 65536) {
        fprintf(stderr, "residue_matching: image too large for 16-bit encoding\n");
        return;
    }

    cudaError_t e;

    /* ---- H2D bitflags ---------------------------------------------------- */
    if ((e = cudaMemcpy(dev->d_bitflags, h_bitflags, length,
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
        cuda_fail(e, "H2D bitflags"); return;
    }

    /* ---- Reset only the two counter slots (payload is overwritten) ------- */
    cudaMemsetAsync(dev->d_pos_residues, 0, sizeof(int));
    cudaMemsetAsync(dev->d_neg_residues, 0, sizeof(int));

    /* ---- Kernel 1: pack residues ----------------------------------------- */
    {
        dim3 block(16, 16);
        dim3 grid((xsize + 15) / 16, (ysize + 15) / 16);
        k_pack_residues<<<grid, block>>>(dev->d_bitflags,
                                         dev->d_pos_residues,
                                         dev->d_neg_residues,
                                         xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_pack_residues"); return;
        }
    }

    int h_n_pos = 0, h_n_neg = 0;
    cudaMemcpy(&h_n_pos, dev->d_pos_residues, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_n_neg, dev->d_neg_residues, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_n_pos == 0 && h_n_neg == 0) {
        /* No residues — nothing to do. */
        cudaMemcpy(h_bitflags, dev->d_bitflags, length, cudaMemcpyDeviceToHost);
        return;
    }

    /* Pick minority vs majority. */
    const int  n_min = (h_n_pos <= h_n_neg) ? h_n_pos : h_n_neg;
    const int  n_maj = (h_n_pos <= h_n_neg) ? h_n_neg : h_n_pos;
    int* const d_min = (h_n_pos <= h_n_neg) ? dev->d_pos_residues
                                            : dev->d_neg_residues;
    int* const d_maj = (h_n_pos <= h_n_neg) ? dev->d_neg_residues
                                            : dev->d_pos_residues;

    /* ---- Kernel 2: minority -> majority matching ------------------------- */
    /*  Writes into d_pairs[0 .. 2*n_min).                                   */
    if (n_min > 0) {
        const int threads = 128;
        const int blocks  = (n_min + threads - 1) / threads;
        k_match_residues<<<blocks, threads>>>(d_min, d_maj, n_min, n_maj,
                                              dev->d_pairs, xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_match_residues"); return;
        }
    }

    /* ---- Kernel 3: leftover majority -> edge ----------------------------- */
    /*  Writes into d_pairs[2*n_min .. 2*n_maj). Empty when n_min == n_maj.  */
    const int n_leftover = n_maj - n_min;
    if (n_leftover > 0) {
        const int threads = 128;
        const int blocks  = (n_leftover + threads - 1) / threads;
        k_fill_leftovers<<<blocks, threads>>>(d_maj, n_min, n_leftover,
                                              dev->d_pairs, xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_fill_leftovers"); return;
        }
    }

    /* ---- Kernel 4: one rasterize pass over the combined buffer ----------- */
    const int n_total = n_maj;  /* == n_min + n_leftover */
    if (n_total > 0) {
        const int threads = 128;
        const int blocks  = (n_total + threads - 1) / threads;
        k_rasterize_cuts<<<blocks, threads>>>(dev->d_pairs, n_total,
                                              dev->d_bitflags, xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_rasterize_cuts"); return;
        }
    }

    if ((e = cudaDeviceSynchronize()) != cudaSuccess) {
        cuda_fail(e, "sync after stage 2"); return;
    }

    /* ---- D2H bitflags ---------------------------------------------------- */
    if ((e = cudaMemcpy(h_bitflags, dev->d_bitflags, length,
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        cuda_fail(e, "D2H bitflags");
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
