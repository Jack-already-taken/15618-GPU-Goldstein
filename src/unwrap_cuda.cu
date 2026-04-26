#include <cstdio>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <cuda_runtime.h>

#include "unwrap_cuda.h"
#include "pi.h"

namespace {

enum : unsigned char { kPosRes = 0x01, kNegRes = 0x02, kBorder = 0x20, kBranchCut = 0x10 };

constexpr int POS_CHUNK = 1024;

__device__ __forceinline__ float device_gradient(float p1, float p2)
{
    float r = p1 - p2;
    if (r > (float)PI)
        r -= (float)TWOPI;
    else if (r < -(float)PI)
        r += (float)TWOPI;
    return r;
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
constexpr unsigned char kUnwrapped  = 0x40;
constexpr unsigned char kAvoid      = kBranchCut | kBorder;

__device__ __forceinline__ bool claim_pixel(unsigned char *flags, int idx)
{
    unsigned int *word = (unsigned int *)(flags + (idx & ~3));
    unsigned int bit = (unsigned int)kUnwrapped << ((idx & 3) * 8);
    unsigned int old = atomicOr(word, bit);
    return !(old & bit);
}

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
__global__ void k_bfs_expand(const float        *phase,
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

/* ------------------------------------------------------------------------- */
/*  (i, j) encoding helpers                                                  */
/* ------------------------------------------------------------------------- */

__device__ __forceinline__ int encode_ij(int i, int j)
{
    return (j << 16) | (i & 0xFFFF);
}

__device__ __forceinline__ void decode_ij(int enc, int &i, int &j)
{
    j = (enc >> 16) & 0xFFFF;
    i =  enc        & 0xFFFF;
}

/* Closed-form nearest image edge for pixel (i, j). */
__device__ __forceinline__ int nearest_edge_enc(int i, int j, int xsize, int ysize)
{
    const int dT = j,                 dB = (ysize - 1) - j;
    const int dL = i,                 dR = (xsize - 1) - i;
    int bi = i, bj = 0, bd = dT;                 /* top edge */
    if (dB < bd) { bd = dB; bi = i;         bj = ysize - 1; }
    if (dL < bd) { bd = dL; bi = 0;         bj = j;         }
    if (dR < bd) {          bi = xsize - 1; bj = j;         }
    return encode_ij(bi, bj);
}

/* Stamp kBranchCut into bitflags[j*xsize + i] via a 32-bit atomicOr on the
 * containing word (CUDA doesn't expose portable 8-bit atomics). cudaMalloc
 * returns 256-byte aligned pointers so the word access is always safe. */
__device__ __forceinline__ void stamp_branch_cut(unsigned char *bitflags,
                                                 int xsize, int i, int j)
{
    const int     idx      = j * xsize + i;
    unsigned int *word_ptr = reinterpret_cast<unsigned int*>(bitflags) + (idx >> 2);
    const unsigned int bit = ((unsigned int)kBranchCut) << ((idx & 3) * 8);
    atomicOr(word_ptr, bit);
}


__host__ __device__ __forceinline__ bool is_blocked_flag(unsigned char b)
{
    return (b & (kBranchCut | kBorder)) != 0;
}

__device__ __forceinline__ bool is_valid_unwrap_pixel(const unsigned char *bitflags,
                                                      int idx)
{
    return !is_blocked_flag(bitflags[idx]);
}

static inline float host_wrap_diff(float p1, float p2)
{
    float r = p1 - p2;
    if (r > (float)PI)
        r -= (float)TWOPI;
    else if (r < -(float)PI)
        r += (float)TWOPI;
    return r;
}

static inline int div_up_int(int a, int b)
{
    return (a + b - 1) / b;
}

/* ------------------------------------------------------------------------- */
/*  Kernel: pack residue coordinates out of bitflags                         */
/* ------------------------------------------------------------------------- */
/*  Residues only exist at (i, j) with i < xsize-1 and j < ysize-1 because   */
/*  k_identify_residues only writes in that subrectangle. We could scan the  */
/*  whole image and nothing bad would happen, but the early return matches   */
/*  the residue-valid region exactly and skips a pointless bitflags read.    */

__global__ void k_pack_residues(const unsigned char *bitflags,
                                int *pos_residues, int *neg_residues,
                                int xsize, int ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize - 1 || j >= ysize - 1)
        return;

    const unsigned char b = bitflags[j * xsize + i];
    const int enc = encode_ij(i, j);

    if (b & kPosRes) {
        const int idx = atomicAdd(&pos_residues[0], 1);
        pos_residues[1 + idx] = enc;
    } else if (b & kNegRes) {
        const int idx = atomicAdd(&neg_residues[0], 1);
        neg_residues[1 + idx] = enc;
    }
}

/* ------------------------------------------------------------------------- */
/*  Kernel: minority -> nearest-majority matching                            */
/* ------------------------------------------------------------------------- */
/*  One thread per minority residue. The block cooperatively streams the     */
/*  majority array through shared memory in POS_CHUNK tiles. Each thread     */
/*  writes its pair into d_pairs[2*min_idx .. 2*min_idx + 1].                */

__global__ void k_match_residues(const int *__restrict__ d_minority,
                                 const int *__restrict__ d_majority,
                                 int n_min, int n_maj,
                                 int *__restrict__ d_pairs,
                                 int xsize, int ysize)
{
    __shared__ int s_maj[POS_CHUNK];

    const int tid     = threadIdx.x;
    const int min_idx = blockIdx.x * blockDim.x + tid;

    int        my_enc   = 0, mi = 0, mj = 0;
    int        best_d2  = INT_MAX;
    int        best_enc = -1;
    const bool active   = (min_idx < n_min);
    if (active) {
        my_enc = d_minority[1 + min_idx];
        decode_ij(my_enc, mi, mj);
    }

    for (int cs = 0; cs < n_maj; cs += POS_CHUNK) {
        const int clen = min(POS_CHUNK, n_maj - cs);

        /* cooperative load of one chunk of the majority array */
        for (int k = tid; k < clen; k += blockDim.x)
            s_maj[k] = d_majority[1 + cs + k];
        __syncthreads();

        if (active) {
            #pragma unroll 4
            for (int k = 0; k < clen; ++k) {
                int pi, pj;
                decode_ij(s_maj[k], pi, pj);
                const int di = pi - mi;
                const int dj = pj - mj;
                const int d2 = di*di + dj*dj;
                const bool better = (d2 < best_d2);
                best_d2  = better ? d2       : best_d2;
                best_enc = better ? s_maj[k] : best_enc;
            }
        }
        __syncthreads();
    }

    if (active) {
        /* Only reachable when n_maj == 0 (no majority residues at all). */
        if (best_enc < 0)
            best_enc = nearest_edge_enc(mi, mj, xsize, ysize);

        d_pairs[2 * min_idx    ] = my_enc;
        d_pairs[2 * min_idx + 1] = best_enc;
    }
}

/* ------------------------------------------------------------------------- */
/*  Kernel: leftover majority -> nearest image edge (closed form)            */
/* ------------------------------------------------------------------------- */
/*  Writes into the tail of the same d_pairs buffer the matcher used, so    */
/*  one rasterize pass over d_pairs[0 .. n_maj) covers everything.          */

__global__ void k_fill_leftovers(const int *__restrict__ d_majority,
                                 int n_min, int n_leftover,
                                 int *__restrict__ d_pairs,
                                 int xsize, int ysize)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_leftover) return;

    const int slot = n_min + t;
    const int enc  = d_majority[1 + slot];
    int i, j;
    decode_ij(enc, i, j);

    d_pairs[2 * slot    ] = enc;
    d_pairs[2 * slot + 1] = nearest_edge_enc(i, j, xsize, ysize);
}

/* ------------------------------------------------------------------------- */
/*  Kernel: rasterize each pair as a Bresenham line of kBranchCut bits       */
/* ------------------------------------------------------------------------- */
/*  Standard all-octants integer Bresenham. di is kept positive and dj      */
/*  negative so the error test is a two-way compare against one positive    */
/*  and one negative value, which is the canonical branchless form.         */

__global__ void k_rasterize_cuts(const int *__restrict__ d_pairs,
                                 int n_pairs,
                                 unsigned char *__restrict__ bitflags,
                                 int xsize, int ysize)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_pairs) return;

    const int a = d_pairs[2 * t    ];
    const int b = d_pairs[2 * t + 1];

    int i0, j0, i1, j1;
    decode_ij(a, i0, j0);
    decode_ij(b, i1, j1);

    int di =  abs(i1 - i0), si = (i0 < i1) ? 1 : -1;
    int dj = -abs(j1 - j0), sj = (j0 < j1) ? 1 : -1;
    int err = di + dj;

    int i = i0, j = j0;
    for (;;) {
        if ((unsigned)i < (unsigned)xsize && (unsigned)j < (unsigned)ysize)
            stamp_branch_cut(bitflags, xsize, i, j);
        if (i == i1 && j == j1) break;
        const int e2 = 2 * err;
        if (e2 >= dj) { err += dj; i += si; }
        if (e2 <= di) { err += di; j += sj; }
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

static inline int residue_capacity(int length)
{
    return length + 1;
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
    const size_t bitflag_bytes = ((size_t)length + 3u) & ~(size_t)3u;
    e = cudaMalloc((void **)&out->d_bitflags, bitflag_bytes);
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

    /* Stage 2 scratch buffers: counters live at [0], residue data starts at [1]. */
    const int cap = residue_capacity(length);

    e = cudaMalloc((void **)&out->d_pos_residues, (size_t)(cap + 1) * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_neg_residues, (size_t)(cap + 1) * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_pairs, 2 * (size_t)cap * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

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
    cudaFree(buf->d_pos_residues);
    cudaFree(buf->d_neg_residues);
    cudaFree(buf->d_pairs);
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
                                                    const UnwrapCudaDeviceBufs *dev,
                                                    int max_cut_len, int num_res,
                                                    int xsize, int ysize, int length)
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

    /* ---- Reset only the counter slots ------------------------------------ */
    cudaMemsetAsync(dev->d_pos_residues, 0, sizeof(int));
    cudaMemsetAsync(dev->d_neg_residues, 0, sizeof(int));

    /* ---- Kernel 1: pack residues ----------------------------------------- */
    {
        dim3 block(16, 16);
        dim3 grid = residue_grid(xsize, ysize);
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

    /* Capacity sanity check. If this fires, the residue density of the      */
    /* input exceeded the allocator's assumed upper bound; the atomicAdd in  */
    /* k_pack_residues has already overrun the buffer and device memory is   */
    /* corrupt. Bail loudly rather than producing silently wrong output.     */
    const int cap = residue_capacity(length);
    if (h_n_pos > cap || h_n_neg > cap) {
        fprintf(stderr, "residue_matching: residue count exceeds capacity "
                        "(pos=%d neg=%d cap=%d) — raise residue_capacity()\n",
                h_n_pos, h_n_neg, cap);
        return;
    }

    /* Pick minority vs majority. */
    const int  n_min = (h_n_pos <= h_n_neg) ? h_n_pos : h_n_neg;
    const int  n_maj = (h_n_pos <= h_n_neg) ? h_n_neg : h_n_pos;
    int *const d_min = (h_n_pos <= h_n_neg) ? dev->d_pos_residues
                                            : dev->d_neg_residues;
    int *const d_maj = (h_n_pos <= h_n_neg) ? dev->d_neg_residues
                                            : dev->d_pos_residues;

    /* ---- Kernel 2: minority -> majority matching ------------------------- */
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
