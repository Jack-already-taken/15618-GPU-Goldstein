#include <cstdio>
#include <cstring>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "unwrap_cuda.h"
#include "pi.h"

namespace {

constexpr unsigned char kUnwrapped  = 0x40;

enum : unsigned char { kPosRes = 0x01, kNegRes = 0x02, kBorder = 0x20, kBranchCut = 0x10 };
constexpr unsigned char kAvoid = kBranchCut | kBorder;

static int cuda_fail(cudaError_t e, const char *msg)
{
    if (e == cudaSuccess)
        return 0;
    fprintf(stderr, "unwrap_cuda: %s: %s\n", msg, cudaGetErrorString(e));
    return (int)e;
}

static void print_cuda_interval(const char *name, cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("  [GPU][timing] %-34s %.4f ms\n", name, ms);
}


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

/* -----------------------------------------------------------------------
 * Editable CUDA launch/block parameters, separated by pipeline stage.
 * Tune these independently for correctness/performance experiments.
 *
 * Stage 1 uses a 2D tile and shared-memory halo for residue detection.
 * Stage 2 uses 1D thread blocks for residue packing and cluster growth.
 * Stage 3 uses 1D thread blocks for BFS expansion and a 2D tile for the
 * final AVOID-band fill pass.
 * ----------------------------------------------------------------------- */
#ifndef STAGE1_RESIDUE_TILE_W
#define STAGE1_RESIDUE_TILE_W 16
#endif
#ifndef STAGE1_RESIDUE_TILE_H
#define STAGE1_RESIDUE_TILE_H 16
#endif

#ifndef STAGE2_PACK_THREADS
#define STAGE2_PACK_THREADS 256
#endif
#ifndef STAGE2_GROW_THREADS
#define STAGE2_GROW_THREADS 128
#endif
#ifndef STAGE2_CLUSTER_MAX_ACTIVE
#define STAGE2_CLUSTER_MAX_ACTIVE 64
#endif
#ifndef STAGE2_CLUSTER_MAX_WINDOW
#define STAGE2_CLUSTER_MAX_WINDOW 4096
#endif
#ifndef STAGE2_POS_CHUNK
#define STAGE2_POS_CHUNK 1024
#endif

#ifndef STAGE2_MATCH_WINDOW
/* 0 = original full-scan nearest-opposite-residue matching.
   >0 = approximate bounded-window matching (faster but topology-changing). */
#define STAGE2_MATCH_WINDOW 0
#endif

#ifndef STAGE2_USE_FIXED_BINS
/* 1 = fixed-size spatial binning for Stage 2 majority-residue lookup.
   This avoids O(N_min*N_maj) global scans. Each bin has a fixed capacity,
   so overflowed residues are dropped from the bin lookup and counted. */
#define STAGE2_USE_FIXED_BINS 1
#endif
#ifndef STAGE2_BIN_GRID_X
#define STAGE2_BIN_GRID_X 128
#endif
#ifndef STAGE2_BIN_GRID_Y
#define STAGE2_BIN_GRID_Y 128
#endif
#ifndef STAGE2_BIN_CAP
#define STAGE2_BIN_CAP 32
#endif
#ifndef STAGE2_BIN_SEARCH_RADIUS
#define STAGE2_BIN_SEARCH_RADIUS 2
#endif
#ifndef STAGE2_BIN_FALLBACK_FULL_SCAN
/* If local bins have no candidate, fall back to the old full scan for that
   residue. Set to 0 for maximum speed, 1 for safer matching. */
#define STAGE2_BIN_FALLBACK_FULL_SCAN 0
#endif

#ifndef STAGE3_FAST_GPU_BFS
/* 0 = exact CPU-order frontier fallback.
   1 = experimental fast GPU BFS. */
#define STAGE3_FAST_GPU_BFS 0
#endif

constexpr int POS_CHUNK = STAGE2_POS_CHUNK;

#ifndef STAGE3_BFS_THREADS
#define STAGE3_BFS_THREADS 256
#endif
#ifndef STAGE3_MAX_BFS_ROUNDS
#define STAGE3_MAX_BFS_ROUNDS 20000
#endif
#ifndef STAGE3_AVOID_TILE_W
#define STAGE3_AVOID_TILE_W 16
#endif
#ifndef STAGE3_AVOID_TILE_H
#define STAGE3_AVOID_TILE_H 16
#endif

__global__ void k_identify_residues(const float *phase, unsigned char *bitflags,
                                    int xsize, int ysize, int *d_num_res)
{
    __shared__ float s[STAGE1_RESIDUE_TILE_H + 1][STAGE1_RESIDUE_TILE_W + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i  = blockIdx.x * STAGE1_RESIDUE_TILE_W + tx;
    const int j  = blockIdx.y * STAGE1_RESIDUE_TILE_H + ty;

    /* Load (TILE_H+1) x (TILE_W+1) patch — every thread loads its own cell,
       edge threads also load the +1 halo column / row */
    if (i < xsize && j < ysize)
        s[ty][tx] = phase[j * xsize + i];

    if (tx == STAGE1_RESIDUE_TILE_W - 1 && i + 1 < xsize && j < ysize)
        s[ty][tx + 1] = phase[j * xsize + (i + 1)];

    if (ty == STAGE1_RESIDUE_TILE_H - 1 && j + 1 < ysize && i < xsize)
        s[ty + 1][tx] = phase[(j + 1) * xsize + i];

    if (tx == STAGE1_RESIDUE_TILE_W - 1 && ty == STAGE1_RESIDUE_TILE_H - 1 && i + 1 < xsize && j + 1 < ysize)
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


static inline float host_gradient(float p1, float p2)
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

#if STAGE2_MATCH_WINDOW > 0
    /* Performance mode: approximate local matching.
       The residue arrays are packed from a 2D grid, so nearby indices are often
       spatially nearby.  Instead of scanning all majority residues, map the
       minority index proportionally into the majority list and search a bounded
       window around that position.  This reduces matching from O(N^2) to
       O(N*W).  It intentionally does not preserve exact CPU Goldstein topology;
       use the relaxed RMS metrics in main.c to judge physical correctness. */
    int begin = 0;
    int end   = n_maj;
    if (active && n_maj > 0) {
        const long long center_ll = ((long long)min_idx * (long long)n_maj) / max(n_min, 1);
        const int center = (int)center_ll;
        begin = center - STAGE2_MATCH_WINDOW;
        end   = center + STAGE2_MATCH_WINDOW + 1;
        if (begin < 0) begin = 0;
        if (end > n_maj) end = n_maj;
        /* Very small fallback: ensure at least one candidate. */
        if (begin >= end) { begin = 0; end = n_maj; }
    }
#else
    const int begin = 0;
    const int end   = n_maj;
#endif

    for (int cs = begin; cs < end; cs += POS_CHUNK) {
        const int clen = min(POS_CHUNK, end - cs);

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
        if (best_enc < 0)
            best_enc = nearest_edge_enc(mi, mj, xsize, ysize);

        d_pairs[2 * min_idx    ] = my_enc;
        d_pairs[2 * min_idx + 1] = best_enc;
    }
}


/* ------------------------------------------------------------------------- */
/*  Stage 2 fixed-bin spatial lookup                                         */
/* ------------------------------------------------------------------------- */
__device__ __forceinline__ int stage2_bin_id_from_xy(int i, int j, int xsize, int ysize)
{
    int bx = (int)(((long long)i * STAGE2_BIN_GRID_X) / max(xsize, 1));
    int by = (int)(((long long)j * STAGE2_BIN_GRID_Y) / max(ysize, 1));
    if (bx < 0) bx = 0;
    if (by < 0) by = 0;
    if (bx >= STAGE2_BIN_GRID_X) bx = STAGE2_BIN_GRID_X - 1;
    if (by >= STAGE2_BIN_GRID_Y) by = STAGE2_BIN_GRID_Y - 1;
    return by * STAGE2_BIN_GRID_X + bx;
}

__global__ void k_bin_majority_residues(const int *__restrict__ d_majority,
                                        int n_maj,
                                        int *__restrict__ d_bin_counts,
                                        int *__restrict__ d_bin_items,
                                        int *__restrict__ d_overflow,
                                        int xsize, int ysize)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_maj) return;

    const int enc = d_majority[1 + t];
    int i, j;
    decode_ij(enc, i, j);
    const int bid = stage2_bin_id_from_xy(i, j, xsize, ysize);
    const int slot = atomicAdd(&d_bin_counts[bid], 1);
    if (slot < STAGE2_BIN_CAP) {
        d_bin_items[bid * STAGE2_BIN_CAP + slot] = enc;
    } else {
        atomicAdd(d_overflow, 1);
    }
}

__global__ void k_match_residues_fixed_bins(const int *__restrict__ d_minority,
                                            const int *__restrict__ d_majority,
                                            int n_min, int n_maj,
                                            const int *__restrict__ d_bin_counts,
                                            const int *__restrict__ d_bin_items,
                                            int *__restrict__ d_pairs,
                                            int xsize, int ysize)
{
    const int min_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (min_idx >= n_min) return;

    const int my_enc = d_minority[1 + min_idx];
    int mi, mj;
    decode_ij(my_enc, mi, mj);

    const int my_bid = stage2_bin_id_from_xy(mi, mj, xsize, ysize);
    const int my_bx = my_bid % STAGE2_BIN_GRID_X;
    const int my_by = my_bid / STAGE2_BIN_GRID_X;

    int best_d2 = INT_MAX;
    int best_enc = -1;

    for (int dy = -STAGE2_BIN_SEARCH_RADIUS; dy <= STAGE2_BIN_SEARCH_RADIUS; ++dy) {
        const int by = my_by + dy;
        if ((unsigned)by >= (unsigned)STAGE2_BIN_GRID_Y) continue;
        for (int dx = -STAGE2_BIN_SEARCH_RADIUS; dx <= STAGE2_BIN_SEARCH_RADIUS; ++dx) {
            const int bx = my_bx + dx;
            if ((unsigned)bx >= (unsigned)STAGE2_BIN_GRID_X) continue;
            const int bid = by * STAGE2_BIN_GRID_X + bx;
            int count = d_bin_counts[bid];
            if (count > STAGE2_BIN_CAP) count = STAGE2_BIN_CAP;

            for (int k = 0; k < count; ++k) {
                const int enc = d_bin_items[bid * STAGE2_BIN_CAP + k];
                int pi, pj;
                decode_ij(enc, pi, pj);
                const int di = pi - mi;
                const int dj = pj - mj;
                const int d2 = di * di + dj * dj;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_enc = enc;
                }
            }
        }
    }

#if STAGE2_BIN_FALLBACK_FULL_SCAN
    if (best_enc < 0) {
        for (int m = 0; m < n_maj; ++m) {
            const int enc = d_majority[1 + m];
            int pi, pj;
            decode_ij(enc, pi, pj);
            const int di = pi - mi;
            const int dj = pj - mj;
            const int d2 = di * di + dj * dj;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_enc = enc;
            }
        }
    }
#endif

    if (best_enc < 0)
        best_enc = nearest_edge_enc(mi, mj, xsize, ysize);

    d_pairs[2 * min_idx    ] = my_enc;
    d_pairs[2 * min_idx + 1] = best_enc;
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

/* ------------------------------------------------------------------------- */
/*  Stage 2 verification: check residue/cut consistency on device           */
/* ------------------------------------------------------------------------- */
/* stats layout:
 *   [0] total residue pixels
 *   [1] residue pixels not marked as branch cut
 *   [2] positive residues not marked as branch cut
 *   [3] negative residues not marked as branch cut
 *   [4] total branch-cut pixels
 *   [5] branch-cut pixels touching image border
 */

/* ------------------------------------------------------------------------- */
/*  Kernel: force every residue pixel to be part of the branch-cut mask       */
/* ------------------------------------------------------------------------- */
__global__ void k_mark_residues_as_branch_cuts(unsigned char *__restrict__ bitflags,
                                               int length)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= length) return;
    if (bitflags[k] & (kPosRes | kNegRes))
        bitflags[k] |= kBranchCut;
}

__global__ void k_verify_stage2_branchcuts(const unsigned char *__restrict__ bitflags,
                                           int length, int xsize, int ysize,
                                           int *__restrict__ stats)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= length) return;

    const unsigned char b = bitflags[k];
    const bool is_pos = (b & kPosRes) != 0;
    const bool is_neg = (b & kNegRes) != 0;
    const bool is_res = is_pos || is_neg;
    const bool is_cut = (b & kBranchCut) != 0;

    if (is_res) {
        atomicAdd(&stats[0], 1);
        if (!is_cut) {
            atomicAdd(&stats[1], 1);
            if (is_pos) atomicAdd(&stats[2], 1);
            if (is_neg) atomicAdd(&stats[3], 1);
        }
    }

    if (is_cut) {
        atomicAdd(&stats[4], 1);
        const int x = k % xsize;
        const int y = k / xsize;
        if (x == 0 || x == xsize - 1 || y == 0 || y == ysize - 1)
            atomicAdd(&stats[5], 1);
    }
}

static void verify_stage2_branchcuts_device(const UnwrapCudaDeviceBufs *dev,
                                            int length, int xsize, int ysize)
{
    if (!dev || !dev->d_bitflags || length <= 0 || xsize <= 0 || ysize <= 0)
        return;

    int *d_stats = nullptr;
    int h_stats[6] = {0, 0, 0, 0, 0, 0};
    cudaError_t e = cudaMalloc((void **)&d_stats, sizeof(h_stats));
    if (e != cudaSuccess) {
        cuda_fail(e, "cudaMalloc stage2 verify stats");
        return;
    }

    cudaMemset(d_stats, 0, sizeof(h_stats));
    const int threads = STAGE2_GROW_THREADS;
    const int blocks = (length + threads - 1) / threads;
    k_verify_stage2_branchcuts<<<blocks, threads>>>(dev->d_bitflags, length,
                                                    xsize, ysize, d_stats);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cuda_fail(e, "k_verify_stage2_branchcuts");
        cudaFree(d_stats);
        return;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cuda_fail(e, "sync stage2 verify");
        cudaFree(d_stats);
        return;
    }
    e = cudaMemcpy(h_stats, d_stats, sizeof(h_stats), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cuda_fail(e, "D2H stage2 verify stats");
        cudaFree(d_stats);
        return;
    }
    cudaFree(d_stats);

    const double miss_pct = h_stats[0] ? 100.0 * (double)h_stats[1] / (double)h_stats[0] : 0.0;
    const double border_pct = h_stats[4] ? 100.0 * (double)h_stats[5] / (double)h_stats[4] : 0.0;

    printf("  [GPU][Stage2 verify] residues=%d, residue_not_cut=%d (%.2f%%), "
           "pos_not_cut=%d, neg_not_cut=%d\n",
           h_stats[0], h_stats[1], miss_pct, h_stats[2], h_stats[3]);
    printf("  [GPU][Stage2 verify] branch_cut_pixels=%d, border_cut_pixels=%d (%.2f%% of cuts)\n",
           h_stats[4], h_stats[5], border_pct);

    if (h_stats[1] > 0) {
        printf("  [GPU][Stage2 verify] WARNING: some residues are not on branch cuts; "
               "Stage 3 may unwrap through residue cells.\n");
    }
}


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

static dim3 residue_grid(int xsize, int ysize)
{
    constexpr int bx = STAGE1_RESIDUE_TILE_W;
    constexpr int by = STAGE1_RESIDUE_TILE_H;
    return dim3((xsize + bx - 2) / bx, (ysize + by - 2) / by);
}

static inline int residue_capacity(int length)
{
    return length / 5 + 4;
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
    e = cudaMalloc((void **)&out->d_bitflags, (((size_t)length + 3u) & ~((size_t)3u)) * sizeof(unsigned char));
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

    /* Stage 2 original scratch buffers. Counters live at [0], residue data starts at [1]. */
    e = cudaMalloc((void **)&out->d_pos_residues, (size_t)length * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_neg_residues, (size_t)length * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_pairs, (size_t)length * sizeof(int));
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

    dim3 block(STAGE1_RESIDUE_TILE_W, STAGE1_RESIDUE_TILE_H);
    dim3 grid((xsize + STAGE1_RESIDUE_TILE_W - 1) / STAGE1_RESIDUE_TILE_W,
          (ysize + STAGE1_RESIDUE_TILE_H - 1) / STAGE1_RESIDUE_TILE_H);

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
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    printf("  [GPU][Stage2 timing] begin\n");

    /* ---- H2D bitflags ---------------------------------------------------- */
    cudaEventRecord(t0, 0);
    e = cudaMemcpy(dev->d_bitflags, h_bitflags, (size_t)length,
                   cudaMemcpyHostToDevice);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 H2D bitflags", t0, t1);
    if (e != cudaSuccess) {
        cuda_fail(e, "H2D bitflags");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }

    /* ---- Reset only the counter slots ------------------------------------ */
    cudaEventRecord(t0, 0);
    cudaMemsetAsync(dev->d_pos_residues, 0, sizeof(int));
    cudaMemsetAsync(dev->d_neg_residues, 0, sizeof(int));
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 reset residue counters", t0, t1);

    /* ---- Kernel 1: pack residues ----------------------------------------- */
    {
        dim3 block(16, 16);
        dim3 grid = residue_grid(xsize, ysize);

        cudaEventRecord(t0, 0);
        k_pack_residues<<<grid, block>>>(dev->d_bitflags,
                                         dev->d_pos_residues,
                                         dev->d_neg_residues,
                                         xsize, ysize);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        print_cuda_interval("Stage2 k_pack_residues", t0, t1);

        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_pack_residues");
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            return;
        }
    }

    int h_n_pos = 0, h_n_neg = 0;
    cudaEventRecord(t0, 0);
    e = cudaMemcpy(&h_n_pos, dev->d_pos_residues, sizeof(int), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess)
        e = cudaMemcpy(&h_n_neg, dev->d_neg_residues, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 D2H residue counts", t0, t1);
    if (e != cudaSuccess) {
        cuda_fail(e, "D2H residue counts");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }

    printf("  [GPU][Stage2] packed residues: pos=%d neg=%d\n", h_n_pos, h_n_neg);

    if (h_n_pos == 0 && h_n_neg == 0) {
        cudaEventRecord(t0, 0);
        e = cudaMemcpy(h_bitflags, dev->d_bitflags, (size_t)length,
                       cudaMemcpyDeviceToHost);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        print_cuda_interval("Stage2 D2H bitflags", t0, t1);
        if (e != cudaSuccess)
            cuda_fail(e, "D2H bitflags");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }

    const int cap = residue_capacity(length);
    if (h_n_pos > cap || h_n_neg > cap) {
        fprintf(stderr, "residue_matching: residue count exceeds capacity "
                        "(pos=%d neg=%d cap=%d) — raise residue_capacity()\n",
                h_n_pos, h_n_neg, cap);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
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
    int *d_bin_counts = NULL;
    int *d_bin_items = NULL;
    int *d_bin_overflow = NULL;

#if STAGE2_USE_FIXED_BINS
    const int n_bins = STAGE2_BIN_GRID_X * STAGE2_BIN_GRID_Y;
    cudaEventRecord(t0, 0);
    e = cudaMalloc((void **)&d_bin_counts, (size_t)n_bins * sizeof(int));
    if (e == cudaSuccess)
        e = cudaMalloc((void **)&d_bin_items,
                       (size_t)n_bins * (size_t)STAGE2_BIN_CAP * sizeof(int));
    if (e == cudaSuccess)
        e = cudaMalloc((void **)&d_bin_overflow, sizeof(int));
    if (e == cudaSuccess)
        e = cudaMemsetAsync(d_bin_counts, 0, (size_t)n_bins * sizeof(int));
    if (e == cudaSuccess)
        e = cudaMemsetAsync(d_bin_overflow, 0, sizeof(int));
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 fixed-bin alloc/reset", t0, t1);
    if (e != cudaSuccess) {
        cuda_fail(e, "Stage2 fixed-bin alloc/reset");
        cudaFree(d_bin_counts); cudaFree(d_bin_items); cudaFree(d_bin_overflow);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }

    if (n_maj > 0) {
        const int threads = STAGE2_GROW_THREADS;
        const int blocks = (n_maj + threads - 1) / threads;
        cudaEventRecord(t0, 0);
        k_bin_majority_residues<<<blocks, threads>>>(d_maj, n_maj,
                                                     d_bin_counts, d_bin_items,
                                                     d_bin_overflow,
                                                     xsize, ysize);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        print_cuda_interval("Stage2 k_bin_majority", t0, t1);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_bin_majority_residues");
            cudaFree(d_bin_counts); cudaFree(d_bin_items); cudaFree(d_bin_overflow);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            return;
        }
    }

    int h_bin_overflow = 0;
    cudaEventRecord(t0, 0);
    e = cudaMemcpy(&h_bin_overflow, d_bin_overflow, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 D2H bin overflow", t0, t1);
    if (e == cudaSuccess) {
        printf("  [GPU][Stage2] fixed bins: grid=%dx%d cap=%d search_radius=%d overflow=%d\n",
               STAGE2_BIN_GRID_X, STAGE2_BIN_GRID_Y, STAGE2_BIN_CAP,
               STAGE2_BIN_SEARCH_RADIUS, h_bin_overflow);
    } else {
        cuda_fail(e, "D2H bin overflow");
    }
#endif

    if (n_min > 0) {
        const int threads = STAGE2_GROW_THREADS;
        const int blocks  = (n_min + threads - 1) / threads;

        cudaEventRecord(t0, 0);
#if STAGE2_USE_FIXED_BINS
        k_match_residues_fixed_bins<<<blocks, threads>>>(d_min, d_maj, n_min, n_maj,
                                                         d_bin_counts, d_bin_items,
                                                         dev->d_pairs, xsize, ysize);
#else
        k_match_residues<<<blocks, threads>>>(d_min, d_maj, n_min, n_maj,
                                              dev->d_pairs, xsize, ysize);
#endif
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
#if STAGE2_USE_FIXED_BINS
        print_cuda_interval("Stage2 k_match_fixed_bins", t0, t1);
#else
        print_cuda_interval("Stage2 k_match_residues", t0, t1);
#endif

        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "Stage2 matching");
            cudaFree(d_bin_counts); cudaFree(d_bin_items); cudaFree(d_bin_overflow);
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            return;
        }
    }

#if STAGE2_USE_FIXED_BINS
    cudaFree(d_bin_counts);
    cudaFree(d_bin_items);
    cudaFree(d_bin_overflow);
#endif

    /* ---- Kernel 3: leftover majority -> edge ----------------------------- */
    const int n_leftover = n_maj - n_min;
    if (n_leftover > 0) {
        const int threads = STAGE2_GROW_THREADS;
        const int blocks  = (n_leftover + threads - 1) / threads;

        cudaEventRecord(t0, 0);
        k_fill_leftovers<<<blocks, threads>>>(d_maj, n_min, n_leftover,
                                              dev->d_pairs, xsize, ysize);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        print_cuda_interval("Stage2 k_fill_leftovers", t0, t1);

        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_fill_leftovers");
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            return;
        }
    } else {
        printf("  [GPU][timing] %-34s %.4f ms\n", "Stage2 k_fill_leftovers", 0.0f);
    }

    /* ---- Kernel 4: one rasterize pass over the combined buffer ----------- */
    const int n_total = n_maj;  /* == n_min + n_leftover */
    if (n_total > 0) {
        const int threads = STAGE2_GROW_THREADS;
        const int blocks  = (n_total + threads - 1) / threads;

        cudaEventRecord(t0, 0);
        k_rasterize_cuts<<<blocks, threads>>>(dev->d_pairs, n_total,
                                              dev->d_bitflags, xsize, ysize);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        print_cuda_interval("Stage2 k_rasterize_cuts", t0, t1);

        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_rasterize_cuts");
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            return;
        }
    }

    /* Safety/correctness fix: Goldstein cuts must include residue endpoints. */
    {
        const int threads = STAGE2_GROW_THREADS;
        const int blocks  = (length + threads - 1) / threads;

        cudaEventRecord(t0, 0);
        k_mark_residues_as_branch_cuts<<<blocks, threads>>>(dev->d_bitflags, length);
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        print_cuda_interval("Stage2 k_mark_residues_as_cuts", t0, t1);

        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_mark_residues_as_branch_cuts");
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            return;
        }
    }

    /* ---- Stage 2 diagnostics: verify that residues are covered by cuts. -- */
    cudaEventRecord(t0, 0);
    verify_stage2_branchcuts_device(dev, length, xsize, ysize);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 verify diagnostics total", t0, t1);

    /* ---- D2H bitflags ---------------------------------------------------- */
    cudaEventRecord(t0, 0);
    e = cudaMemcpy(h_bitflags, dev->d_bitflags, (size_t)length,
                   cudaMemcpyDeviceToHost);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 D2H bitflags", t0, t1);
    if (e != cudaSuccess)
        cuda_fail(e, "D2H bitflags");

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}
extern "C" void unwrap_cuda_launch_unwrapping(
    float *h_phase, unsigned char *h_bitflags, float *h_soln,
    float *h_gradx, float *h_grady,
    const UnwrapCudaDeviceBufs *dev,
    int xsize, int ysize, int length)
{
    if (length < 1 || !h_phase || !h_bitflags || !h_soln || !h_gradx || !h_grady
        || xsize < 1 || ysize < 1 || length != xsize * ysize)
        return;

#if STAGE3_FAST_GPU_BFS
    if (!dev || !dev->d_phase || !dev->d_bitflags || !dev->d_soln
        || !dev->d_gradx || !dev->d_grady
        || !dev->d_frontier_a || !dev->d_frontier_b
        || !dev->d_frontier_count_a || !dev->d_frontier_count_b) {
        fprintf(stderr, "unwrap_cuda: Stage3 fast GPU BFS missing device buffers; falling back to CPU exact\n");
    } else {
        cudaError_t e;
        constexpr unsigned char kAvoidU = kBranchCut | kBorder;
        cudaEvent_t s3_t0, s3_t1;
        cudaEventCreate(&s3_t0);
        cudaEventCreate(&s3_t1);
        clock_t seed_cpu_t0 = clock();

        printf("  [GPU][Stage3 timing] begin\n");

        /* CPU seed selection: one seed per connected component.  This is cheap
           compared with full CPU unwrapping and avoids the old single-source
           failure when cuts split the image. */
        int h_frontier_count = 0;
        int *h_frontier_tmp = (int*)malloc((size_t)length * sizeof(int));
        int *stack = (int*)malloc((size_t)length * sizeof(int));
        if (!h_frontier_tmp || !stack) {
            fprintf(stderr, "unwrap_cuda: Stage3 seed malloc failed; falling back to CPU exact\n");
            free(h_frontier_tmp);
            free(stack);
        } else {
            for (int k = 0; k < length; k++)
                h_bitflags[k] &= ~kUnwrapped;

            for (int k = 0; k < length; k++) {
                if (!(h_bitflags[k] & (kAvoidU | kUnwrapped))) {
                    h_soln[k] = h_phase[k];
                    h_bitflags[k] |= kUnwrapped;
                    h_frontier_tmp[h_frontier_count++] = k;

                    int top = 0;
                    stack[top++] = k;
                    while (top > 0) {
                        const int cur = stack[--top];
                        const int cx = cur % xsize;
                        const int cy = cur / xsize;
                        int nb;
                        if (cx > 0) {
                            nb = cur - 1;
                            if (!(h_bitflags[nb] & (kAvoidU | kUnwrapped))) {
                                h_bitflags[nb] |= kUnwrapped;
                                stack[top++] = nb;
                            }
                        }
                        if (cx + 1 < xsize) {
                            nb = cur + 1;
                            if (!(h_bitflags[nb] & (kAvoidU | kUnwrapped))) {
                                h_bitflags[nb] |= kUnwrapped;
                                stack[top++] = nb;
                            }
                        }
                        if (cy > 0) {
                            nb = cur - xsize;
                            if (!(h_bitflags[nb] & (kAvoidU | kUnwrapped))) {
                                h_bitflags[nb] |= kUnwrapped;
                                stack[top++] = nb;
                            }
                        }
                        if (cy + 1 < ysize) {
                            nb = cur + xsize;
                            if (!(h_bitflags[nb] & (kAvoidU | kUnwrapped))) {
                                h_bitflags[nb] |= kUnwrapped;
                                stack[top++] = nb;
                            }
                        }
                    }
                }
            }

            /* Reset all temporary component marks, then mark only seeds. */
            for (int k = 0; k < length; k++)
                h_bitflags[k] &= ~kUnwrapped;
            for (int i = 0; i < h_frontier_count; i++)
                h_bitflags[h_frontier_tmp[i]] |= kUnwrapped;

            {
                double seed_ms = 1000.0 * (double)(clock() - seed_cpu_t0) / (double)CLOCKS_PER_SEC;
                printf("  [GPU][timing] %-34s %.4f ms\n", "Stage3 CPU component seed scan", seed_ms);
            }

            cudaEventRecord(s3_t0, 0);
            if ((e = cudaMemcpy(dev->d_phase, h_phase,
                                (size_t)length * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
                cuda_fail(e, "Stage3 H2D phase");
            } else if ((e = cudaMemcpy(dev->d_bitflags, h_bitflags,
                                       (size_t)length * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess) {
                cuda_fail(e, "Stage3 H2D bitflags");
            } else if ((e = cudaMemcpy(dev->d_soln, h_soln,
                                       (size_t)length * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
                cuda_fail(e, "Stage3 H2D soln");
            } else if ((e = cudaMemcpy(dev->d_gradx, h_gradx,
                                       (size_t)length * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
                cuda_fail(e, "Stage3 H2D gradx");
            } else if ((e = cudaMemcpy(dev->d_grady, h_grady,
                                       (size_t)length * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
                cuda_fail(e, "Stage3 H2D grady");
            } else if ((e = cudaMemcpy(dev->d_frontier_a, h_frontier_tmp,
                                       (size_t)h_frontier_count * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) {
                cuda_fail(e, "Stage3 H2D seeds");
            } else if ((e = cudaMemcpy(dev->d_frontier_count_a, &h_frontier_count,
                                       sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) {
                cuda_fail(e, "Stage3 H2D seed count");
            } else {
                cudaEventRecord(s3_t1, 0);
                cudaEventSynchronize(s3_t1);
                print_cuda_interval("Stage3 H2D inputs + seeds", s3_t0, s3_t1);

                cudaEventRecord(s3_t0, 0);
                cudaMemset(dev->d_frontier_count_b, 0, sizeof(int));

                int *d_in = dev->d_frontier_a;
                int *d_out = dev->d_frontier_b;
                int *n_in = dev->d_frontier_count_a;
                int *n_out = dev->d_frontier_count_b;
                int h_n_in = h_frontier_count;
                int round = 0;

                while (h_n_in > 0 && round < STAGE3_MAX_BFS_ROUNDS) {
                    cudaMemset(n_out, 0, sizeof(int));
                    const int threads = STAGE3_BFS_THREADS;
                    const int blocks = (h_n_in + threads - 1) / threads;
                    k_bfs_expand<<<blocks, threads>>>(
                        dev->d_phase, dev->d_bitflags, dev->d_soln,
                        dev->d_gradx, dev->d_grady,
                        d_in, h_n_in, d_out, n_out,
                        xsize, ysize);
                    if ((e = cudaGetLastError()) != cudaSuccess) {
                        cuda_fail(e, "k_bfs_expand");
                        break;
                    }
                    if ((e = cudaDeviceSynchronize()) != cudaSuccess) {
                        cuda_fail(e, "sync k_bfs_expand");
                        break;
                    }
                    int *tmp;
                    tmp = d_in; d_in = d_out; d_out = tmp;
                    tmp = n_in; n_in = n_out; n_out = tmp;
                    cudaMemcpy(&h_n_in, n_in, sizeof(int), cudaMemcpyDeviceToHost);
                    ++round;
                }

                cudaEventRecord(s3_t1, 0);
                cudaEventSynchronize(s3_t1);
                print_cuda_interval("Stage3 BFS expand loop total", s3_t0, s3_t1);

                if (h_n_in > 0) {
                    fprintf(stderr,
                            "unwrap_cuda: Stage3 BFS stopped after STAGE3_MAX_BFS_ROUNDS=%d "
                            "with frontier=%d still active; output may be incomplete.\n",
                            STAGE3_MAX_BFS_ROUNDS, h_n_in);
                }

                dim3 block(STAGE3_AVOID_TILE_W, STAGE3_AVOID_TILE_H);
                dim3 grid((xsize + STAGE3_AVOID_TILE_W - 1) / STAGE3_AVOID_TILE_W,
                          (ysize + STAGE3_AVOID_TILE_H - 1) / STAGE3_AVOID_TILE_H);
                cudaEventRecord(s3_t0, 0);
                k_avoid_fill<<<grid, block>>>(dev->d_phase, dev->d_bitflags, dev->d_soln,
                                              xsize, ysize);
                if ((e = cudaGetLastError()) != cudaSuccess)
                    cuda_fail(e, "k_avoid_fill");
                cudaDeviceSynchronize();
                cudaEventRecord(s3_t1, 0);
                cudaEventSynchronize(s3_t1);
                print_cuda_interval("Stage3 k_avoid_fill", s3_t0, s3_t1);

                cudaEventRecord(s3_t0, 0);
                cudaMemcpy(h_soln, dev->d_soln,
                           (size_t)length * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_bitflags, dev->d_bitflags,
                           (size_t)length * sizeof(unsigned char), cudaMemcpyDeviceToHost);
                cudaEventRecord(s3_t1, 0);
                cudaEventSynchronize(s3_t1);
                print_cuda_interval("Stage3 D2H soln + bitflags", s3_t0, s3_t1);

                printf("  [GPU] Stage3 fast BFS: seeds=%d, rounds=%d, max_rounds=%d\n",
                       h_frontier_count, round, STAGE3_MAX_BFS_ROUNDS);
                cudaEventDestroy(s3_t0);
                cudaEventDestroy(s3_t1);
                free(h_frontier_tmp);
                free(stack);
                return;
            }
            cudaEventDestroy(s3_t0);
            cudaEventDestroy(s3_t1);
            free(h_frontier_tmp);
            free(stack);
        }
    }
#endif

    /* Exact CPU fallback.  Compile with -DSTAGE3_FAST_GPU_BFS=0 to force this
       path when exact CPU-order replay is needed for debugging. */
    printf("  [GPU][Stage3 timing] begin (exact CPU-order fallback)\n");
    clock_t s3_exact_total_t0 = clock();
    clock_t s3_exact_t0 = clock();

    int *list = (int*)malloc((size_t)2 * (size_t)(xsize + ysize) * sizeof(int));
    if (!list) {
        fprintf(stderr, "unwrap_cuda: Stage3 host frontier malloc failed\n");
        return;
    }

    int num_pieces = 0;
    int max_frontier = 0;

    for (int k = 0; k < length; k++) {
        if (!(h_bitflags[k] & (kBranchCut | kBorder | kUnwrapped))) {
            ++num_pieces;
            h_soln[k] = h_phase[k];

            int flag = 1;
            int base_in = 0;
            int base_out = xsize + ysize;
            int top_in = base_in;
            int top_out = base_out;

            list[top_in++] = k;

            while (flag) {
                if (top_in - base_in > max_frontier)
                    max_frontier = top_in - base_in;

                for (int l = base_in; l < top_in; l++) {
                    const int kk = list[l];
                    const int x = kk % xsize;
                    const int y = kk / xsize;

                    h_bitflags[kk] |= kUnwrapped;
                    const float value = h_soln[kk];

                    int index;
                    index = kk - 1;
                    if (x - 1 >= 0 && !(h_bitflags[index] & (kBranchCut | kUnwrapped | kBorder))) {
                        h_bitflags[index] |= kUnwrapped;
                        h_soln[index] = value + h_gradx[index];
                        list[top_out++] = index;
                    }
                    index = kk + 1;
                    if (x + 1 < xsize && !(h_bitflags[index] & (kBranchCut | kUnwrapped | kBorder))) {
                        h_bitflags[index] |= kUnwrapped;
                        h_soln[index] = value - h_gradx[kk];
                        list[top_out++] = index;
                    }
                    index = kk - xsize;
                    if (y - 1 >= 0 && !(h_bitflags[index] & (kBranchCut | kUnwrapped | kBorder))) {
                        h_bitflags[index] |= kUnwrapped;
                        h_soln[index] = value + h_grady[index];
                        list[top_out++] = index;
                    }
                    index = kk + xsize;
                    if (y + 1 < ysize && !(h_bitflags[index] & (kBranchCut | kUnwrapped | kBorder))) {
                        h_bitflags[index] |= kUnwrapped;
                        h_soln[index] = value - h_grady[kk];
                        list[top_out++] = index;
                    }
                }

                if (base_out == top_out) flag = 0;
                else {
                    int tmp;
                    tmp = base_in;  base_in = base_out;  base_out = tmp;
                    tmp = top_in;   top_in = top_out;    top_out = tmp;
                    top_out = base_out;
                }
            }
        }
    }

    {
        double ms = 1000.0 * (double)(clock() - s3_exact_t0) / (double)CLOCKS_PER_SEC;
        printf("  [GPU][timing] %-34s %.4f ms\n", "Stage3 exact frontier unwrap", ms);
    }
    s3_exact_t0 = clock();

    for (int j = 1; j < ysize; j++) {
        for (int i = 1; i < xsize; i++) {
            const int k = j * xsize + i;
            if (h_bitflags[k] & (kBranchCut | kBorder)) {
                if (!(h_bitflags[k - 1] & (kBranchCut | kBorder)))
                    h_soln[k] = h_soln[k - 1] + host_gradient(h_phase[k], h_phase[k - 1]);
                else if (!(h_bitflags[k - xsize] & (kBranchCut | kBorder)))
                    h_soln[k] = h_soln[k - xsize] + host_gradient(h_phase[k], h_phase[k - xsize]);
            }
        }
    }

    {
        double ms = 1000.0 * (double)(clock() - s3_exact_t0) / (double)CLOCKS_PER_SEC;
        printf("  [GPU][timing] %-34s %.4f ms\n", "Stage3 exact avoid-fill", ms);
        double total_ms = 1000.0 * (double)(clock() - s3_exact_total_t0) / (double)CLOCKS_PER_SEC;
        printf("  [GPU][timing] %-34s %.4f ms\n", "Stage3 exact total", total_ms);
    }

    printf("  [GPU] Stage3 exact CPU fallback: pieces=%d, max_frontier=%d\n",
           num_pieces, max_frontier);
    free(list);
}
