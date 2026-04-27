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


// __device__ __forceinline__ float device_gradient(float p1, float p2)
// {
//     float r = p1 - p2;
//     if (r > (float)PI)
//         r -= (float)TWOPI;
//     else if (r < -(float)PI)
//         r += (float)TWOPI;
//     return r;
// }

__device__ __forceinline__ float device_gradient(float p1, float p2)
{
    const float PI_F    = (float)PI;
    const float TWOPI_F = (float)TWOPI;

    float r = p1 - p2;

    // corr = +1 if r < -pi, -1 if r > pi, 0 otherwise
    float corr = (float)(r < -PI_F) - (float)(r > PI_F);

    return __fmaf_rn(TWOPI_F, corr, r);
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
 * Stage 3 uses tile-independent local unwrap, tile-graph height stitching,
 * and a 2D tile for the final AVOID-band fill pass.
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
#define STAGE2_BIN_GRID_X 64
#endif
#ifndef STAGE2_BIN_GRID_Y
#define STAGE2_BIN_GRID_Y 64
#endif
#ifndef STAGE2_BIN_CAP
#define STAGE2_BIN_CAP 128
#endif
#ifndef STAGE2_BIN_SEARCH_RADIUS
#define STAGE2_BIN_SEARCH_RADIUS 2
#endif
#ifndef STAGE2_BIN_FALLBACK_FULL_SCAN
/* If local bins have no candidate, fall back to the old full scan for that
   residue. Set to 0 for maximum speed, 1 for safer matching. */
#define STAGE2_BIN_FALLBACK_FULL_SCAN 0
#endif

constexpr int POS_CHUNK = STAGE2_POS_CHUNK;

#ifndef STAGE3_AVOID_TILE_W
#define STAGE3_AVOID_TILE_W 16
#endif
#ifndef STAGE3_AVOID_TILE_H
#define STAGE3_AVOID_TILE_H 16
#endif

#ifndef STAGE3_TILE_W
#define STAGE3_TILE_W 32
#endif
#ifndef STAGE3_TILE_H
#define STAGE3_TILE_H 32
#endif
#ifndef STAGE3_TILE_LOCAL_MAX_RESTARTS
#define STAGE3_TILE_LOCAL_MAX_RESTARTS 1
#endif
#ifndef STAGE3_TILE_RELAX_MAX_ITERS
#define STAGE3_TILE_RELAX_MAX_ITERS 4096
#endif

#ifndef STAGE3_TILE_WAVEFRONT_THREADS
/* Threads per tile CTA for the SIMD/SIMT local wavefront unwrap.
   256 works well for 32x32 tiles: each thread owns about 4 pixels. */
#define STAGE3_TILE_WAVEFRONT_THREADS 256
#endif
#ifndef STAGE3_TILE_WAVEFRONT_MAX_ROUNDS
/* Upper bound for one connected-component wave expansion inside a tile.
   Using 4x tile diameter keeps the wavefront robust around local branch-cut
   obstacles while still bounding per-tile work. */
#define STAGE3_TILE_WAVEFRONT_MAX_ROUNDS ((STAGE3_TILE_W + STAGE3_TILE_H) * 4)
#endif

#ifndef STAGE3_STITCH_MIN_SUPPORT
/* Minimum same-k boundary votes required before accepting a tile-to-tile height edge. */
#define STAGE3_STITCH_MIN_SUPPORT 4
#endif
#ifndef STAGE3_STITCH_REQUIRE_MAJORITY
/* 1 = require chosen k to explain at least half of all valid boundary samples. */
#define STAGE3_STITCH_REQUIRE_MAJORITY 1
#endif

__global__ void k_identify_residues_pack_grad(const float *phase,
                                                    unsigned char *bitflags,
                                                    float *gradx,
                                                    float *grady,
                                                    int xsize,
                                                    int ysize,
                                                    int residue_capacity,
                                                    int *pos_residues,
                                                    int *neg_residues,
                                                    int *d_num_res)
{
    __shared__ float s[STAGE1_RESIDUE_TILE_H + 1][STAGE1_RESIDUE_TILE_W + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i  = blockIdx.x * STAGE1_RESIDUE_TILE_W + tx;
    const int j  = blockIdx.y * STAGE1_RESIDUE_TILE_H + ty;

    if (i < xsize && j < ysize)
        s[ty][tx] = phase[j * xsize + i];

    if (tx == STAGE1_RESIDUE_TILE_W - 1 && i + 1 < xsize && j < ysize)
        s[ty][tx + 1] = phase[j * xsize + (i + 1)];

    if (ty == STAGE1_RESIDUE_TILE_H - 1 && j + 1 < ysize && i < xsize)
        s[ty + 1][tx] = phase[(j + 1) * xsize + i];

    if (tx == STAGE1_RESIDUE_TILE_W - 1 && ty == STAGE1_RESIDUE_TILE_H - 1 &&
        i + 1 < xsize && j + 1 < ysize)
        s[ty + 1][tx + 1] = phase[(j + 1) * xsize + (i + 1)];

    __syncthreads();

    if (i >= xsize || j >= ysize)
        return;

    const int k = j * xsize + i;

    /* Stage 3 gradient generation is fused here so Stage 3 no longer launches
       k_compute_unwrap_gradients nor reloads phase/bitflags from host.  The
       sign convention matches the old k_compute_unwrap_gradients kernel. */
    const float p00 = s[ty][tx];
    if (i + 1 < xsize) {
        const float px = s[ty][tx + 1];
        gradx[k] = device_gradient(p00, px);
    } else {
        gradx[k] = 0.0f;
    }
    if (j + 1 < ysize) {
        const float py = s[ty + 1][tx];
        grady[k] = device_gradient(p00, py);
    } else {
        grady[k] = 0.0f;
    }

    if (i >= xsize - 1 || j >= ysize - 1)
        return;

    constexpr unsigned char avoid = kBranchCut | kBorder;
    const unsigned char b00 = bitflags[k];
    if ((b00 & avoid) || (bitflags[k + 1] & avoid) ||
        (bitflags[k + 1 + xsize] & avoid) || (bitflags[k + xsize] & avoid))
        return;

    const float p10 = s[ty    ][tx + 1];
    const float p11 = s[ty + 1][tx + 1];
    const float p01 = s[ty + 1][tx    ];

    const float r = device_gradient(p10, p00)
                  + device_gradient(p11, p10)
                  + device_gradient(p01, p11)
                  + device_gradient(p00, p01);

    const float thr = static_cast<float>(RESIDUE_THRESHOLD);
    unsigned char out = b00;

    if (r > thr) {
        out = (unsigned char)(out | kPosRes | kBranchCut);
        const int idx = atomicAdd(&pos_residues[0], 1);
        if (idx < residue_capacity)
            pos_residues[1 + idx] = ((j << 16) | (i & 0xFFFF));
        atomicAdd(d_num_res, 1);
    } else if (r < -thr) {
        out = (unsigned char)(out | kNegRes | kBranchCut);
        const int idx = atomicAdd(&neg_residues[0], 1);
        if (idx < residue_capacity)
            neg_residues[1 + idx] = ((j << 16) | (i & 0xFFFF));
        atomicAdd(d_num_res, 1);
    }

    bitflags[k] = out;
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

static inline int stage2_nbins_value(void)
{
    return STAGE2_BIN_GRID_X * STAGE2_BIN_GRID_Y;
}

static inline int stage2_bin_items_value(void)
{
    return STAGE2_BIN_GRID_X * STAGE2_BIN_GRID_Y * STAGE2_BIN_CAP;
}

static inline int stage3_max_tiles_from_length(int length)
{
    const int min_tile = (STAGE3_TILE_W < STAGE3_TILE_H) ? STAGE3_TILE_W : STAGE3_TILE_H;
    return div_up_int(length, min_tile);
}

static inline int stage3_max_edges_from_length(int length)
{
    return 2 * stage3_max_tiles_from_length(length);
}

/* ------------------------------------------------------------------------- */
/*  Kernel: pack residue coordinates out of bitflags                         */
/* ------------------------------------------------------------------------- */
/*  Residues only exist at (i, j) with i < xsize-1 and j < ysize-1 because   */
/*  k_identify_residues only writes in that subrectangle. We could scan the  */
/*  whole image and nothing bad would happen, but the early return matches   */
/*  the residue-valid region exactly and skips a pointless bitflags read.    */

__global__ void k_pack_residues_and_mark_cuts(unsigned char *bitflags,
                                             int *pos_residues, int *neg_residues,
                                             int xsize, int ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize - 1 || j >= ysize - 1)
        return;

    const int idx_img = j * xsize + i;
    const unsigned char b = bitflags[idx_img];
    const int enc = encode_ij(i, j);

    if (b & kPosRes) {
        bitflags[idx_img] = (unsigned char)(b | kBranchCut);
        const int idx = atomicAdd(&pos_residues[0], 1);
        pos_residues[1 + idx] = enc;
    } else if (b & kNegRes) {
        bitflags[idx_img] = (unsigned char)(b | kBranchCut);
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


__device__ __forceinline__ int clamped_residue_count(const int *a, int cap)
{
    int n = a[0];
    if (n < 0) n = 0;
    if (n > cap) n = cap;
    return n;
}

__device__ __forceinline__ void select_min_majority_arrays(
    const int *pos_residues,
    const int *neg_residues,
    int cap,
    const int **d_min,
    const int **d_maj,
    int *n_min,
    int *n_maj)
{
    const int n_pos = clamped_residue_count(pos_residues, cap);
    const int n_neg = clamped_residue_count(neg_residues, cap);
    if (n_pos <= n_neg) {
        *d_min = pos_residues;
        *d_maj = neg_residues;
        *n_min = n_pos;
        *n_maj = n_neg;
    } else {
        *d_min = neg_residues;
        *d_maj = pos_residues;
        *n_min = n_neg;
        *n_maj = n_pos;
    }
}

__global__ void k_bin_majority_residues_auto(const int *__restrict__ pos_residues,
                                             const int *__restrict__ neg_residues,
                                             int cap,
                                             int *__restrict__ d_bin_counts,
                                             int *__restrict__ d_bin_items,
                                             int *__restrict__ d_overflow,
                                             int xsize,
                                             int ysize)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int *d_min;
    const int *d_maj;
    int n_min, n_maj;
    select_min_majority_arrays(pos_residues, neg_residues, cap,
                               &d_min, &d_maj, &n_min, &n_maj);
    (void)d_min;
    (void)n_min;
    if (t >= n_maj) return;

    const int enc = d_maj[1 + t];
    int i, j;
    decode_ij(enc, i, j);
    const int bid = stage2_bin_id_from_xy(i, j, xsize, ysize);
    const int slot = atomicAdd(&d_bin_counts[bid], 1);
    if (slot < STAGE2_BIN_CAP)
        d_bin_items[bid * STAGE2_BIN_CAP + slot] = enc;
    else
        atomicAdd(d_overflow, 1);
}

__global__ void k_match_residues_fixed_bins_auto(const int *__restrict__ pos_residues,
                                                 const int *__restrict__ neg_residues,
                                                 int cap,
                                                 const int *__restrict__ d_bin_counts,
                                                 const int *__restrict__ d_bin_items,
                                                 int *__restrict__ d_pairs,
                                                 int xsize,
                                                 int ysize)
{
    const int min_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int *d_min;
    const int *d_maj;
    int n_min, n_maj;
    select_min_majority_arrays(pos_residues, neg_residues, cap,
                               &d_min, &d_maj, &n_min, &n_maj);
    if (min_idx >= n_min) return;

    const int my_enc = d_min[1 + min_idx];
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
            const int enc = d_maj[1 + m];
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

__global__ void k_match_residues_auto_scan(const int *__restrict__ pos_residues,
                                           const int *__restrict__ neg_residues,
                                           int cap,
                                           int *__restrict__ d_pairs,
                                           int xsize,
                                           int ysize)
{
    const int min_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int *d_min;
    const int *d_maj;
    int n_min, n_maj;
    select_min_majority_arrays(pos_residues, neg_residues, cap,
                               &d_min, &d_maj, &n_min, &n_maj);
    if (min_idx >= n_min) return;

    const int my_enc = d_min[1 + min_idx];
    int mi, mj;
    decode_ij(my_enc, mi, mj);

    int best_d2 = INT_MAX;
    int best_enc = -1;
    for (int m = 0; m < n_maj; ++m) {
        const int enc = d_maj[1 + m];
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

    if (best_enc < 0)
        best_enc = nearest_edge_enc(mi, mj, xsize, ysize);

    d_pairs[2 * min_idx    ] = my_enc;
    d_pairs[2 * min_idx + 1] = best_enc;
}

__global__ void k_fill_leftovers_auto(const int *__restrict__ pos_residues,
                                      const int *__restrict__ neg_residues,
                                      int cap,
                                      int *__restrict__ d_pairs,
                                      int xsize,
                                      int ysize)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int *d_min;
    const int *d_maj;
    int n_min, n_maj;
    select_min_majority_arrays(pos_residues, neg_residues, cap,
                               &d_min, &d_maj, &n_min, &n_maj);
    (void)d_min;

    const int n_leftover = n_maj - n_min;
    if (t >= n_leftover) return;

    const int slot = n_min + t;
    const int enc  = d_maj[1 + slot];
    int i, j;
    decode_ij(enc, i, j);

    d_pairs[2 * slot    ] = enc;
    d_pairs[2 * slot + 1] = nearest_edge_enc(i, j, xsize, ysize);
}

__global__ void k_rasterize_cuts_auto(const int *__restrict__ pos_residues,
                                      const int *__restrict__ neg_residues,
                                      int cap,
                                      const int *__restrict__ d_pairs,
                                      unsigned char *__restrict__ bitflags,
                                      int xsize,
                                      int ysize)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_pos = clamped_residue_count(pos_residues, cap);
    const int n_neg = clamped_residue_count(neg_residues, cap);
    const int n_pairs = (n_pos <= n_neg) ? n_neg : n_pos;
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

/* Residue endpoint marking is fused into k_pack_residues_and_mark_cuts. */

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
    if (!dev || !dev->d_bitflags || !dev->d_stage2_verify_stats ||
        length <= 0 || xsize <= 0 || ysize <= 0)
        return;

    int *const d_stats = dev->d_stage2_verify_stats;
    int h_stats[6] = {0, 0, 0, 0, 0, 0};

    cudaError_t e = cudaMemset(d_stats, 0, sizeof(h_stats));
    if (e != cudaSuccess) {
        cuda_fail(e, "memset stage2 verify stats");
        return;
    }

    const int threads = STAGE2_GROW_THREADS;
    const int blocks = (length + threads - 1) / threads;
    k_verify_stage2_branchcuts<<<blocks, threads>>>(dev->d_bitflags, length,
                                                    xsize, ysize, d_stats);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        cuda_fail(e, "k_verify_stage2_branchcuts");
        return;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        cuda_fail(e, "sync stage2 verify");
        return;
    }
    e = cudaMemcpy(h_stats, d_stats, sizeof(h_stats), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cuda_fail(e, "D2H stage2 verify stats");
        return;
    }

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


/* -----------------------------------------------------------------------
 * Stage 3 tile-independent local unwrap + robust height stitching.
 *
 * Each CUDA block owns one spatial tile.  Inside the tile, one worker performs
 * repeated four-direction sweeps.  This avoids global pixel propagation and
 * exposes tile-level parallelism.  A second phase stitches tiles by propagating
 * one additive offset per tile through valid tile-boundary contacts.
 *
 * This is not CPU-order exact.  It is intended as the fast tile-independent path:
 *   local tiled four-direction unwrap -> robust tile-offset propagation -> apply.
 * ----------------------------------------------------------------------- */


__global__ void k_compute_unwrap_gradients(const float *phase,
                                           float *gradx,
                                           float *grady,
                                           int xsize,
                                           int ysize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= xsize || y >= ysize) return;

    const int k = y * xsize + x;
    gradx[k] = (x + 1 < xsize) ? device_gradient(phase[k], phase[k + 1]) : 0.0f;
    grady[k] = (y + 1 < ysize) ? device_gradient(phase[k], phase[k + xsize]) : 0.0f;
}

__global__ void k_tile_local_fourdir_wavefront(const float *phase,
                                                unsigned char *bitflags,
                                                float *soln,
                                                const float *gradx,
                                                const float *grady,
                                                int xsize,
                                                int ysize)
{
    constexpr int TILE_PIXELS = STAGE3_TILE_W * STAGE3_TILE_H;
    enum : unsigned char {
        TILE_UNKNOWN = 0,
        TILE_DONE    = 1,
        TILE_CUT     = 2
    };

    __shared__ float s_sol[TILE_PIXELS];
    __shared__ unsigned char s_state[TILE_PIXELS];
    __shared__ int s_seed_pack;
    __shared__ int s_changed;

    const int tile_x = blockIdx.x;
    const int tile_y = blockIdx.y;
    const int x0 = tile_x * STAGE3_TILE_W;
    const int y0 = tile_y * STAGE3_TILE_H;
    const int x1 = min(x0 + STAGE3_TILE_W, xsize);
    const int y1 = min(y0 + STAGE3_TILE_H, ysize);
    const int local_w = x1 - x0;
    const int local_h = y1 - y0;
    const int local_count = local_w * local_h;
    const int tid = threadIdx.x;

    if (local_count <= 0) return;

    /* Load tile-valid state into shared memory.  We do not need shared phase:
       all propagation uses precomputed wrapped gradients plus shared local soln. */
    for (int li = tid; li < local_count; li += blockDim.x) {
        const int lx = li % local_w;
        const int ly = li / local_w;
        const int g = (y0 + ly) * xsize + (x0 + lx);

        s_sol[li] = phase[g];
        s_state[li] = (bitflags[g] & kAvoid) ? TILE_CUT : TILE_UNKNOWN;
    }
    __syncthreads();

    int restarts = 0;
    while (restarts < STAGE3_TILE_LOCAL_MAX_RESTARTS) {
        if (tid == 0) s_seed_pack = INT_MAX;
        __syncthreads();

        /* Pick a deterministic seed for the next connected component:
           the unvisited valid pixel closest to the tile center.  Each tile still
           has an arbitrary local 2*pi height; the later tile graph fixes that. */
        const int cx = local_w / 2;
        const int cy = local_h / 2;
        for (int li = tid; li < local_count; li += blockDim.x) {
            if (s_state[li] != TILE_UNKNOWN) continue;
            const int lx = li % local_w;
            const int ly = li / local_w;
            const int dx = lx - cx;
            const int dy = ly - cy;
            const int score = dx * dx + dy * dy;
            const int pack = score * TILE_PIXELS + li;
            atomicMin(&s_seed_pack, pack);
        }
        __syncthreads();

        if (s_seed_pack == INT_MAX) break;

        const int seed = s_seed_pack % TILE_PIXELS;
        if (tid == 0) {
            const int sx = seed % local_w;
            const int sy = seed / local_w;
            const int sg = (y0 + sy) * xsize + (x0 + sx);
            s_sol[seed] = phase[sg];
            s_state[seed] = TILE_DONE;
        }
        __syncthreads();

        /* SIMT wavefront floodfill.  Each round all unknown pixels try to attach
           to an already-unwrapped 4-neighbor.  This replaces the old one-thread
           CPU-style tile sweep while preserving local four-direction semantics.

           Sign convention follows k_compute_unwrap_gradients():
             gradx[p] = wrap(phase[p] - phase[p+1])
             grady[p] = wrap(phase[p] - phase[p+xsize])
           Therefore:
             from left  L -> P: sol[P] = sol[L] - gradx[L]
             from right R -> P: sol[P] = sol[R] + gradx[P]
             from up    U -> P: sol[P] = sol[U] - grady[U]
             from down  D -> P: sol[P] = sol[D] + grady[P]
        */
        for (int round = 0; round < STAGE3_TILE_WAVEFRONT_MAX_ROUNDS; ++round) {
            if (tid == 0) s_changed = 0;
            __syncthreads();

            for (int li = tid; li < local_count; li += blockDim.x) {
                if (s_state[li] != TILE_UNKNOWN) continue;

                const int lx = li % local_w;
                const int ly = li / local_w;
                const int g = (y0 + ly) * xsize + (x0 + lx);

                float v = 0.0f;
                bool can_unwrap = false;

                /* Fixed priority makes the result deterministic enough for
                   debugging.  Other priorities are possible. */
                if (lx > 0 && s_state[li - 1] == TILE_DONE) {
                    v = s_sol[li - 1] - gradx[g - 1];
                    can_unwrap = true;
                } else if (lx + 1 < local_w && s_state[li + 1] == TILE_DONE) {
                    v = s_sol[li + 1] + gradx[g];
                    can_unwrap = true;
                } else if (ly > 0 && s_state[li - local_w] == TILE_DONE) {
                    v = s_sol[li - local_w] - grady[g - xsize];
                    can_unwrap = true;
                } else if (ly + 1 < local_h && s_state[li + local_w] == TILE_DONE) {
                    v = s_sol[li + local_w] + grady[g];
                    can_unwrap = true;
                }

                if (can_unwrap) {
                    s_sol[li] = v;
                    __threadfence_block();
                    s_state[li] = TILE_DONE;
                    atomicExch(&s_changed, 1);
                }
            }
            __syncthreads();
            if (!s_changed) break;
        }

        ++restarts;
        __syncthreads();
    }

    /* Commit local tile result.  Only pixels reached by the wavefront become
       kUnwrapped and participate in tile-boundary height voting. */
    for (int li = tid; li < local_count; li += blockDim.x) {
        const int lx = li % local_w;
        const int ly = li / local_w;
        const int g = (y0 + ly) * xsize + (x0 + lx);
        if (s_state[li] == TILE_DONE) {
            soln[g] = s_sol[li];
            bitflags[g] |= kUnwrapped;
        }
    }
}


/* -----------------------------------------------------------------------
 * Robust tile-height matching helpers.
 *
 * Each accepted edge stores one relative integer 2*pi relation:
 *
 *   offset[neighbor] - offset[current] = edge_k * 2*pi
 *
 * The boundary relation is estimated from all valid same-boundary samples,
 * not from the first valid pair.  This avoids most one-pixel seam errors.
 * ----------------------------------------------------------------------- */

__device__ __forceinline__ bool tile_stitch_valid(const unsigned char *bitflags, int idx)
{
    return !(bitflags[idx] & kAvoid) && (bitflags[idx] & kUnwrapped);
}

__device__ __forceinline__ void majority_vote_add(int k, int &candidate, int &balance)
{
    if (balance == 0) {
        candidate = k;
        balance = 1;
    } else if (k == candidate) {
        ++balance;
    } else {
        --balance;
    }
}

/* Returns offset[right_tile] - offset[left_tile] in integer multiples of 2*pi. */
__device__ __forceinline__ bool robust_boundary_right_delta_k(
    const unsigned char *bitflags,
    const float *soln,
    const float *gradx,
    int xsize,
    int ysize,
    int tx,
    int ty,
    int *delta_k_out)
{
    const int ax = min((tx + 1) * STAGE3_TILE_W, xsize) - 1;
    const int bx = ax + 1;
    if (ax < 0 || bx >= xsize) return false;

    const int y0 = ty * STAGE3_TILE_H;
    const int y1 = min(y0 + STAGE3_TILE_H, ysize);
    const float inv_twopi = 1.0f / (float)TWOPI;

    int candidate = 0;
    int balance = 0;
    int total = 0;

    for (int y = y0; y < y1; ++y) {
        const int a = y * xsize + ax;  /* left tile boundary pixel */
        const int b = y * xsize + bx;  /* right tile boundary pixel */
        if (!tile_stitch_valid(bitflags, a) || !tile_stitch_valid(bitflags, b)) continue;

        /* soln[b] + offB should equal soln[a] + offA - gradx[a].
           Therefore offB - offA = soln[a] - gradx[a] - soln[b]. */
        const float raw = (soln[a] - gradx[a]) - soln[b];
        const int k = __float2int_rn(raw * inv_twopi);
        majority_vote_add(k, candidate, balance);
        ++total;
    }

    if (total <= 0) return false;

    int support = 0;
    for (int y = y0; y < y1; ++y) {
        const int a = y * xsize + ax;
        const int b = y * xsize + bx;
        if (!tile_stitch_valid(bitflags, a) || !tile_stitch_valid(bitflags, b)) continue;

        const float raw = (soln[a] - gradx[a]) - soln[b];
        const int k = __float2int_rn(raw * inv_twopi);
        if (k == candidate) ++support;
    }

    if (support < STAGE3_STITCH_MIN_SUPPORT) return false;
#if STAGE3_STITCH_REQUIRE_MAJORITY
    if (support * 2 < total) return false;
#endif

    *delta_k_out = candidate;
    return true;
}

/* Returns offset[bottom_tile] - offset[top_tile] in integer multiples of 2*pi. */
__device__ __forceinline__ bool robust_boundary_down_delta_k(
    const unsigned char *bitflags,
    const float *soln,
    const float *grady,
    int xsize,
    int ysize,
    int tx,
    int ty,
    int *delta_k_out)
{
    const int ay = min((ty + 1) * STAGE3_TILE_H, ysize) - 1;
    const int by = ay + 1;
    if (ay < 0 || by >= ysize) return false;

    const int x0 = tx * STAGE3_TILE_W;
    const int x1 = min(x0 + STAGE3_TILE_W, xsize);
    const float inv_twopi = 1.0f / (float)TWOPI;

    int candidate = 0;
    int balance = 0;
    int total = 0;

    for (int x = x0; x < x1; ++x) {
        const int a = ay * xsize + x;  /* top tile boundary pixel */
        const int b = by * xsize + x;  /* bottom tile boundary pixel */
        if (!tile_stitch_valid(bitflags, a) || !tile_stitch_valid(bitflags, b)) continue;

        /* soln[b] + offB should equal soln[a] + offA - grady[a].
           Therefore offB - offA = soln[a] - grady[a] - soln[b]. */
        const float raw = (soln[a] - grady[a]) - soln[b];
        const int k = __float2int_rn(raw * inv_twopi);
        majority_vote_add(k, candidate, balance);
        ++total;
    }

    if (total <= 0) return false;

    int support = 0;
    for (int x = x0; x < x1; ++x) {
        const int a = ay * xsize + x;
        const int b = by * xsize + x;
        if (!tile_stitch_valid(bitflags, a) || !tile_stitch_valid(bitflags, b)) continue;

        const float raw = (soln[a] - grady[a]) - soln[b];
        const int k = __float2int_rn(raw * inv_twopi);
        if (k == candidate) ++support;
    }

    if (support < STAGE3_STITCH_MIN_SUPPORT) return false;
#if STAGE3_STITCH_REQUIRE_MAJORITY
    if (support * 2 < total) return false;
#endif

    *delta_k_out = candidate;
    return true;
}

__global__ void k_clear_unwrapped_flags(unsigned char *bitflags, int length)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= length) return;
    bitflags[k] &= (unsigned char)(~kUnwrapped);
}

/* edge index layout:
 *   e = 2 * tile_id + 0: right edge, tile -> tile+1
 *   e = 2 * tile_id + 1: down edge,  tile -> tile+tiles_x
 */
__global__ void k_build_tile_edge_constraints(const unsigned char *bitflags,
                                              const float *soln,
                                              const float *gradx,
                                              const float *grady,
                                              int *tile_has_valid,
                                              int *edge_valid,
                                              int *edge_delta_k,
                                              int tiles_x,
                                              int tiles_y,
                                              int xsize,
                                              int ysize)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int ntiles = tiles_x * tiles_y;
    if (tid >= ntiles) return;

    const int tx = tid % tiles_x;
    const int ty = tid / tiles_x;
    const int e_right = 2 * tid;
    const int e_down  = 2 * tid + 1;

    edge_valid[e_right] = 0;
    edge_valid[e_down]  = 0;
    edge_delta_k[e_right] = 0;
    edge_delta_k[e_down]  = 0;

    int has_valid = 0;
    const int x0 = tx * STAGE3_TILE_W;
    const int y0 = ty * STAGE3_TILE_H;
    const int x1 = min(x0 + STAGE3_TILE_W, xsize);
    const int y1 = min(y0 + STAGE3_TILE_H, ysize);

    for (int y = y0; y < y1 && !has_valid; ++y) {
        for (int x = x0; x < x1; ++x) {
            const int k = y * xsize + x;
            if (tile_stitch_valid(bitflags, k)) {
                has_valid = 1;
                break;
            }
        }
    }
    tile_has_valid[tid] = has_valid;

    int dk = 0;
    if (tx + 1 < tiles_x &&
        robust_boundary_right_delta_k(bitflags, soln, gradx, xsize, ysize, tx, ty, &dk)) {
        edge_valid[e_right] = 1;
        edge_delta_k[e_right] = dk;
    }

    if (ty + 1 < tiles_y &&
        robust_boundary_down_delta_k(bitflags, soln, grady, xsize, ysize, tx, ty, &dk)) {
        edge_valid[e_down] = 1;
        edge_delta_k[e_down] = dk;
    }
}

__global__ void k_apply_tile_offsets_k(float *soln,
                                       const unsigned char *bitflags,
                                       const int *tile_known,
                                       const int *tile_offset_k,
                                       int tiles_x,
                                       int xsize,
                                       int ysize)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= xsize || y >= ysize) return;

    const int k = y * xsize + x;
    if (bitflags[k] & kAvoid) return;

    const int tx = x / STAGE3_TILE_W;
    const int ty = y / STAGE3_TILE_H;
    const int tid = ty * tiles_x + tx;

    if (tile_known[tid])
        soln[k] += (float)tile_offset_k[tid] * (float)TWOPI;
}

/* -----------------------------------------------------------------------
 * Stage 3 kernel 3: AVOID-band fill (branch cuts + border pixels)
 * Same logic as the CPU serial AVOID pass — run once after tile stitching.
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

    out->stage2_nbins = stage2_nbins_value();
    out->stage2_bin_items = stage2_bin_items_value();
    out->stage3_tile_capacity = stage3_max_tiles_from_length(length);
    out->stage3_edge_capacity = stage3_max_edges_from_length(length);

    e = cudaMalloc((void **)&out->d_phase, (size_t)length * sizeof(float));
    if (e != cudaSuccess)
        return (int)e;
    e = cudaMalloc((void **)&out->d_bitflags, (((size_t)length + 3u) & ~((size_t)3u)) * sizeof(unsigned char));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_soln, (size_t)length * sizeof(float));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_residue_count, sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    /* Stage 2 residue lists and pair buffer. Counters live at [0], residue data starts at [1]. */
    e = cudaMalloc((void **)&out->d_pos_residues, (size_t)length * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_neg_residues, (size_t)length * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_pairs, (size_t)length * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    /* Stage 2 fixed-bin scratch. These are reset, not reallocated, per launch. */
    e = cudaMalloc((void **)&out->d_bin_counts, (size_t)out->stage2_nbins * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_bin_items, (size_t)out->stage2_bin_items * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_bin_overflow, sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_stage2_verify_stats, 6 * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    /* Stage 3 image buffers. */
    e = cudaMalloc((void **)&out->d_gradx, (size_t)length * sizeof(float));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_grady, (size_t)length * sizeof(float));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    /* Stage 3 tile-graph device scratch. Capacity is a safe upper bound derived from length only. */
    e = cudaMalloc((void **)&out->d_tile_has_valid, (size_t)out->stage3_tile_capacity * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_edge_valid, (size_t)out->stage3_edge_capacity * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_edge_delta_k, (size_t)out->stage3_edge_capacity * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_tile_known, (size_t)out->stage3_tile_capacity * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }
    e = cudaMalloc((void **)&out->d_tile_offset_k, (size_t)out->stage3_tile_capacity * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    /* Stage 3 tile-graph host scratch. These replace per-launch malloc/calloc. */
    out->h_tile_has_valid = (int*)calloc((size_t)out->stage3_tile_capacity, sizeof(int));
    out->h_edge_valid     = (int*)calloc((size_t)out->stage3_edge_capacity, sizeof(int));
    out->h_edge_delta_k   = (int*)calloc((size_t)out->stage3_edge_capacity, sizeof(int));
    out->h_tile_known     = (int*)calloc((size_t)out->stage3_tile_capacity, sizeof(int));
    out->h_tile_offset_k  = (int*)calloc((size_t)out->stage3_tile_capacity, sizeof(int));
    out->h_queue          = (int*)malloc((size_t)out->stage3_tile_capacity * sizeof(int));
    if (!out->h_tile_has_valid || !out->h_edge_valid || !out->h_edge_delta_k ||
        !out->h_tile_known || !out->h_tile_offset_k || !out->h_queue) {
        unwrap_cuda_device_bufs_free(out);
        return -2;
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
    cudaFree(buf->d_pos_residues);
    cudaFree(buf->d_neg_residues);
    cudaFree(buf->d_pairs);
    cudaFree(buf->d_bin_counts);
    cudaFree(buf->d_bin_items);
    cudaFree(buf->d_bin_overflow);
    cudaFree(buf->d_stage2_verify_stats);
    cudaFree(buf->d_gradx);
    cudaFree(buf->d_grady);
    cudaFree(buf->d_tile_has_valid);
    cudaFree(buf->d_edge_valid);
    cudaFree(buf->d_edge_delta_k);
    cudaFree(buf->d_tile_known);
    cudaFree(buf->d_tile_offset_k);

    free(buf->h_tile_has_valid);
    free(buf->h_edge_valid);
    free(buf->h_edge_delta_k);
    free(buf->h_tile_known);
    free(buf->h_tile_offset_k);
    free(buf->h_queue);

    memset(buf, 0, sizeof(*buf));
}

extern "C" int unwrap_cuda_launch_residue_identification(
    float *h_phase, unsigned char *h_bitflags, const UnwrapCudaDeviceBufs *dev, int xsize, int ysize,
    int length)
{
    if (!h_phase || !h_bitflags || !dev || !dev->d_phase || !dev->d_bitflags ||
        !dev->d_residue_count || !dev->d_pos_residues || !dev->d_neg_residues ||
        !dev->d_gradx || !dev->d_grady || xsize < 2 || ysize < 2 || length != xsize * ysize)
        return -1;

    cudaError_t e;
    const int cap = residue_capacity(length);

    /* Only H2D entry point for the CUDA pipeline.  After this launch, bitflags,
       packed residues, and gradients stay resident on device until final D2H. */
    if ((e = cudaMemcpy(dev->d_phase, h_phase, (size_t)length * sizeof(float), cudaMemcpyHostToDevice))
        != cudaSuccess)
        return cuda_fail(e, "Stage1 H2D phase");
    if ((e = cudaMemcpy(dev->d_bitflags, h_bitflags,
                        (size_t)length * sizeof(unsigned char),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return cuda_fail(e, "Stage1 H2D bitflags");

    if ((e = cudaMemset(dev->d_residue_count, 0, sizeof(int))) != cudaSuccess)
        return cuda_fail(e, "Stage1 memset total residue count");
    if ((e = cudaMemset(dev->d_pos_residues, 0, sizeof(int))) != cudaSuccess)
        return cuda_fail(e, "Stage1 memset pos residue count");
    if ((e = cudaMemset(dev->d_neg_residues, 0, sizeof(int))) != cudaSuccess)
        return cuda_fail(e, "Stage1 memset neg residue count");

    dim3 block(STAGE1_RESIDUE_TILE_W, STAGE1_RESIDUE_TILE_H);
    dim3 grid((xsize + STAGE1_RESIDUE_TILE_W - 1) / STAGE1_RESIDUE_TILE_W,
              (ysize + STAGE1_RESIDUE_TILE_H - 1) / STAGE1_RESIDUE_TILE_H);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    k_identify_residues_pack_grad<<<grid, block>>>(dev->d_phase, dev->d_bitflags,
                                                   dev->d_gradx, dev->d_grady,
                                                   xsize, ysize, cap,
                                                   dev->d_pos_residues,
                                                   dev->d_neg_residues,
                                                   dev->d_residue_count);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    printf("  [GPU] Stage1 fused residue+pack+grad: %.4f ms\n", kernel_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if ((e = cudaGetLastError()) != cudaSuccess)
        return cuda_fail(e, "k_identify_residues_pack_grad");
    if ((e = cudaDeviceSynchronize()) != cudaSuccess)
        return cuda_fail(e, "sync fused residues");

    /* No D2H bitflags/count here. Stage 2 reads the device counters directly.
       Return 0 as an intentional host-side placeholder. */
    return 0;
}

extern "C" void unwrap_cuda_launch_residue_matching(unsigned char *h_bitflags,
                                                    const UnwrapCudaDeviceBufs *dev,
                                                    int max_cut_len, int num_res,
                                                    int xsize, int ysize, int length)
{
    (void)h_bitflags;
    (void)max_cut_len;
    (void)num_res;

    if (length < 1 || !dev || !dev->d_bitflags || !dev->d_pos_residues ||
        !dev->d_neg_residues || !dev->d_pairs)
        return;
    if (xsize >= 65536 || ysize >= 65536) {
        fprintf(stderr, "residue_matching: image too large for 16-bit encoding\n");
        return;
    }

    cudaError_t e;
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    const int cap = residue_capacity(length);
    const int threads = STAGE2_GROW_THREADS;
    const int blocks = (cap + threads - 1) / threads;

    printf("  [GPU][Stage2 timing] begin device-resident/no-verify path\n");

#if STAGE2_USE_FIXED_BINS
    const int n_bins = STAGE2_BIN_GRID_X * STAGE2_BIN_GRID_Y;
    if (!dev->d_bin_counts || !dev->d_bin_items || !dev->d_bin_overflow ||
        dev->stage2_nbins < n_bins || dev->stage2_bin_items < n_bins * STAGE2_BIN_CAP) {
        fprintf(stderr, "residue_matching: fixed-bin scratch buffers were not allocated correctly\n");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }

    cudaEventRecord(t0, 0);
    e = cudaMemsetAsync(dev->d_bin_counts, 0, (size_t)n_bins * sizeof(int));
    if (e == cudaSuccess)
        e = cudaMemsetAsync(dev->d_bin_overflow, 0, sizeof(int));
    k_bin_majority_residues_auto<<<blocks, threads>>>(dev->d_pos_residues,
                                                      dev->d_neg_residues,
                                                      cap,
                                                      dev->d_bin_counts,
                                                      dev->d_bin_items,
                                                      dev->d_bin_overflow,
                                                      xsize, ysize);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 reset+bin_majority_auto", t0, t1);
    if (e != cudaSuccess) {
        cuda_fail(e, "Stage2 fixed-bin reset");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }
    if ((e = cudaGetLastError()) != cudaSuccess) {
        cuda_fail(e, "k_bin_majority_residues_auto");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }

    cudaEventRecord(t0, 0);
    k_match_residues_fixed_bins_auto<<<blocks, threads>>>(dev->d_pos_residues,
                                                          dev->d_neg_residues,
                                                          cap,
                                                          dev->d_bin_counts,
                                                          dev->d_bin_items,
                                                          dev->d_pairs,
                                                          xsize, ysize);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 k_match_fixed_bins_auto", t0, t1);
    if ((e = cudaGetLastError()) != cudaSuccess) {
        cuda_fail(e, "k_match_residues_fixed_bins_auto");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }
#else
    cudaEventRecord(t0, 0);
    k_match_residues_auto_scan<<<blocks, threads>>>(dev->d_pos_residues,
                                                    dev->d_neg_residues,
                                                    cap,
                                                    dev->d_pairs,
                                                    xsize, ysize);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 k_match_residues_auto_scan", t0, t1);
    if ((e = cudaGetLastError()) != cudaSuccess) {
        cuda_fail(e, "k_match_residues_auto_scan");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }
#endif

    cudaEventRecord(t0, 0);
    k_fill_leftovers_auto<<<blocks, threads>>>(dev->d_pos_residues,
                                               dev->d_neg_residues,
                                               cap,
                                               dev->d_pairs,
                                               xsize, ysize);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 k_fill_leftovers_auto", t0, t1);
    if ((e = cudaGetLastError()) != cudaSuccess) {
        cuda_fail(e, "k_fill_leftovers_auto");
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return;
    }

    cudaEventRecord(t0, 0);
    k_rasterize_cuts_auto<<<blocks, threads>>>(dev->d_pos_residues,
                                               dev->d_neg_residues,
                                               cap,
                                               dev->d_pairs,
                                               dev->d_bitflags,
                                               xsize, ysize);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage2 k_rasterize_cuts_auto", t0, t1);
    if ((e = cudaGetLastError()) != cudaSuccess)
        cuda_fail(e, "k_rasterize_cuts_auto");

    /* No D2H bitflags/counts and no verification diagnostics in max-perf mode. */
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

extern "C" void unwrap_cuda_launch_unwrapping(
    float *h_phase,
    unsigned char *h_bitflags,
    float *h_soln,
    float *h_gradx,
    float *h_grady,
    const UnwrapCudaDeviceBufs *dev,
    int xsize,
    int ysize,
    int length)
{
    (void)h_gradx;
    (void)h_grady;

    if (!h_phase || !h_bitflags || !h_soln
        || !dev || !dev->d_phase || !dev->d_bitflags || !dev->d_soln
        || !dev->d_gradx || !dev->d_grady
        || !dev->d_tile_has_valid || !dev->d_edge_valid || !dev->d_edge_delta_k
        || !dev->d_tile_known || !dev->d_tile_offset_k
        || !dev->h_tile_has_valid || !dev->h_edge_valid || !dev->h_edge_delta_k
        || !dev->h_tile_known || !dev->h_tile_offset_k || !dev->h_queue
        || xsize <= 0 || ysize <= 0 || length != xsize * ysize) {
        fprintf(stderr, "unwrap_cuda: invalid Stage3 arguments or missing preallocated scratch buffers\n");
        return;
    }

    printf("  [GPU][Stage3 timing] begin (tile-independent wavefront unwrap + tile-graph height stitching)\n");
    fflush(stdout);

    cudaError_t e = cudaSuccess;
    cudaEvent_t t0 = nullptr, t1 = nullptr, total0 = nullptr, total1 = nullptr;

    const int tiles_x = div_up_int(xsize, STAGE3_TILE_W);
    const int tiles_y = div_up_int(ysize, STAGE3_TILE_H);
    const int ntiles = tiles_x * tiles_y;
    const int edge_count = 2 * ntiles;
    const int tile_threads = 256;
    const int tile_blocks = div_up_int(ntiles, tile_threads);

    int *const d_tile_has_valid = dev->d_tile_has_valid;
    int *const d_edge_valid = dev->d_edge_valid;
    int *const d_edge_delta_k = dev->d_edge_delta_k;
    int *const d_tile_known = dev->d_tile_known;
    int *const d_tile_offset_k = dev->d_tile_offset_k;

    int *const h_tile_has_valid = dev->h_tile_has_valid;
    int *const h_edge_valid = dev->h_edge_valid;
    int *const h_edge_delta_k = dev->h_edge_delta_k;
    int *const h_tile_known = dev->h_tile_known;
    int *const h_tile_offset_k = dev->h_tile_offset_k;
    int *const h_queue = dev->h_queue;

    int valid_edges = 0;
    int visited_tiles = 0;
    int components = 0;
    int conflicts = 0;
    int max_abs_offset_k = 0;

    if (ntiles > dev->stage3_tile_capacity || edge_count > dev->stage3_edge_capacity) {
        fprintf(stderr,
                "unwrap_cuda: Stage3 tile scratch capacity too small "
                "(need tiles=%d edges=%d, have tiles=%d edges=%d)\n",
                ntiles, edge_count, dev->stage3_tile_capacity, dev->stage3_edge_capacity);
        return;
    }

    e = cudaEventCreate(&t0);
    if (cuda_fail(e, "Stage3 create event t0")) goto cleanup;
    e = cudaEventCreate(&t1);
    if (cuda_fail(e, "Stage3 create event t1")) goto cleanup;
    e = cudaEventCreate(&total0);
    if (cuda_fail(e, "Stage3 create event total0")) goto cleanup;
    e = cudaEventCreate(&total1);
    if (cuda_fail(e, "Stage3 create event total1")) goto cleanup;
    e = cudaEventRecord(total0, 0);
    if (cuda_fail(e, "Stage3 record total0")) goto cleanup;

    /* Per-launch host scratch reset.  The arrays themselves are allocated once
       by unwrap_cuda_device_bufs_alloc and freed by unwrap_cuda_device_bufs_free. */
    memset(h_tile_known, 0, (size_t)ntiles * sizeof(int));
    memset(h_tile_offset_k, 0, (size_t)ntiles * sizeof(int));

    cudaEventRecord(t0, 0);
    /* Device-resident handoff from Stage 2: do not recopy phase or bitflags.
       Gradients were already produced by the fused Stage 1 kernel. */
    (void)h_phase;
    (void)h_bitflags;
    e = cudaMemset(dev->d_soln, 0, (size_t)length * sizeof(float));
    if (cuda_fail(e, "Stage3 memset soln")) goto cleanup;
    {
        const int threads = 256;
        const int blocks = div_up_int(length, threads);
        k_clear_unwrapped_flags<<<blocks, threads>>>(dev->d_bitflags, length);
        e = cudaGetLastError();
        if (cuda_fail(e, "k_clear_unwrapped_flags")) goto cleanup;
        e = cudaDeviceSynchronize();
        if (cuda_fail(e, "sync Stage3 resident setup")) goto cleanup;
    }
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage3 resident setup", t0, t1);

    cudaEventRecord(t0, 0);
    {
        dim3 grid(tiles_x, tiles_y);
        k_tile_local_fourdir_wavefront<<<grid, STAGE3_TILE_WAVEFRONT_THREADS>>>(
            dev->d_phase, dev->d_bitflags, dev->d_soln,
            dev->d_gradx, dev->d_grady, xsize, ysize);
        e = cudaGetLastError();
        if (cuda_fail(e, "k_tile_local_fourdir_wavefront")) goto cleanup;
        e = cudaDeviceSynchronize();
        if (cuda_fail(e, "sync k_tile_local_fourdir_wavefront")) goto cleanup;
    }
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage3 tile-local wavefront", t0, t1);

    cudaEventRecord(t0, 0);
    k_build_tile_edge_constraints<<<tile_blocks, tile_threads>>>(
        dev->d_bitflags, dev->d_soln, dev->d_gradx, dev->d_grady,
        d_tile_has_valid, d_edge_valid, d_edge_delta_k,
        tiles_x, tiles_y, xsize, ysize);
    e = cudaGetLastError();
    if (cuda_fail(e, "k_build_tile_edge_constraints")) goto cleanup;
    e = cudaDeviceSynchronize();
    if (cuda_fail(e, "sync k_build_tile_edge_constraints")) goto cleanup;
    e = cudaMemcpy(h_tile_has_valid, d_tile_has_valid,
                   (size_t)ntiles * sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_fail(e, "Stage3 D2H tile_has_valid")) goto cleanup;
    e = cudaMemcpy(h_edge_valid, d_edge_valid,
                   (size_t)edge_count * sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_fail(e, "Stage3 D2H edge_valid")) goto cleanup;
    e = cudaMemcpy(h_edge_delta_k, d_edge_delta_k,
                   (size_t)edge_count * sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_fail(e, "Stage3 D2H edge_delta_k")) goto cleanup;
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage3 build+D2H tile edges", t0, t1);

    cudaEventRecord(t0, 0);

    /* CPU solve on the tiny tile graph.  This is not pixel propagation; it only solves
       one integer 2*pi height per tile. */
    for (int tid = 0; tid < ntiles; ++tid) {
        if (!h_edge_valid[2 * tid] && !h_edge_valid[2 * tid + 1])
            continue;
        valid_edges += h_edge_valid[2 * tid] + h_edge_valid[2 * tid + 1];
    }

    for (int root = 0; root < ntiles; ++root) {
        if (!h_tile_has_valid[root] || h_tile_known[root])
            continue;

        ++components;
        h_tile_known[root] = 1;
        h_tile_offset_k[root] = 0;

        int qh = 0;
        int qt = 0;
        h_queue[qt++] = root;

        while (qh < qt) {
            const int cur = h_queue[qh++];
            const int tx = cur % tiles_x;
            const int ty = cur / tiles_x;
            const int cur_k = h_tile_offset_k[cur];

            int nb, eidx, expected;

            if (tx + 1 < tiles_x) {
                nb = cur + 1;
                eidx = 2 * cur;
                if (h_edge_valid[eidx]) {
                    expected = cur_k + h_edge_delta_k[eidx];
                    if (!h_tile_known[nb]) {
                        h_tile_known[nb] = 1;
                        h_tile_offset_k[nb] = expected;
                        h_queue[qt++] = nb;
                    } else if (h_tile_offset_k[nb] != expected) {
                        ++conflicts;
                    }
                }
            }

            if (tx > 0) {
                nb = cur - 1;
                eidx = 2 * nb;
                if (h_edge_valid[eidx]) {
                    expected = cur_k - h_edge_delta_k[eidx];
                    if (!h_tile_known[nb]) {
                        h_tile_known[nb] = 1;
                        h_tile_offset_k[nb] = expected;
                        h_queue[qt++] = nb;
                    } else if (h_tile_offset_k[nb] != expected) {
                        ++conflicts;
                    }
                }
            }

            if (ty + 1 < tiles_y) {
                nb = cur + tiles_x;
                eidx = 2 * cur + 1;
                if (h_edge_valid[eidx]) {
                    expected = cur_k + h_edge_delta_k[eidx];
                    if (!h_tile_known[nb]) {
                        h_tile_known[nb] = 1;
                        h_tile_offset_k[nb] = expected;
                        h_queue[qt++] = nb;
                    } else if (h_tile_offset_k[nb] != expected) {
                        ++conflicts;
                    }
                }
            }

            if (ty > 0) {
                nb = cur - tiles_x;
                eidx = 2 * nb + 1;
                if (h_edge_valid[eidx]) {
                    expected = cur_k - h_edge_delta_k[eidx];
                    if (!h_tile_known[nb]) {
                        h_tile_known[nb] = 1;
                        h_tile_offset_k[nb] = expected;
                        h_queue[qt++] = nb;
                    } else if (h_tile_offset_k[nb] != expected) {
                        ++conflicts;
                    }
                }
            }
        }
    }

    for (int tid = 0; tid < ntiles; ++tid) {
        if (h_tile_known[tid]) {
            ++visited_tiles;
            const int a = abs(h_tile_offset_k[tid]);
            if (a > max_abs_offset_k) max_abs_offset_k = a;
        }
    }

    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage3 CPU tile graph solve", t0, t1);

    cudaEventRecord(t0, 0);
    e = cudaMemcpy(d_tile_known, h_tile_known,
                   (size_t)ntiles * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_fail(e, "Stage3 H2D tile_known")) goto cleanup;
    e = cudaMemcpy(d_tile_offset_k, h_tile_offset_k,
                   (size_t)ntiles * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_fail(e, "Stage3 H2D tile_offset_k")) goto cleanup;

    {
        dim3 block(STAGE3_AVOID_TILE_W, STAGE3_AVOID_TILE_H);
        dim3 grid(div_up_int(xsize, STAGE3_AVOID_TILE_W), div_up_int(ysize, STAGE3_AVOID_TILE_H));

        k_apply_tile_offsets_k<<<grid, block>>>(dev->d_soln, dev->d_bitflags,
                                                d_tile_known, d_tile_offset_k,
                                                tiles_x, xsize, ysize);
        e = cudaGetLastError();
        if (cuda_fail(e, "k_apply_tile_offsets_k")) goto cleanup;

        k_avoid_fill<<<grid, block>>>(dev->d_phase, dev->d_bitflags, dev->d_soln, xsize, ysize);
        e = cudaGetLastError();
        if (cuda_fail(e, "k_avoid_fill")) goto cleanup;
        e = cudaDeviceSynchronize();
        if (cuda_fail(e, "sync apply tile offsets/avoid")) goto cleanup;
    }

    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage3 apply offsets + avoid", t0, t1);

    cudaEventRecord(t0, 0);
    e = cudaMemcpy(h_soln, dev->d_soln, (size_t)length * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_fail(e, "Stage3 final D2H soln")) goto cleanup;
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    print_cuda_interval("Stage3 final D2H soln", t0, t1);

    cudaEventRecord(total1, 0);
    cudaEventSynchronize(total1);
    print_cuda_interval("Stage3 tile total", total0, total1);
    printf("  [GPU] Stage3 wavefront+tilegraph-stitch: tile=%dx%d threads=%d tiles=%dx%d valid_edges=%d visited_tiles=%d/%d components=%d conflicts=%d max_abs_offset_k=%d min_support=%d majority=%d\n",
           STAGE3_TILE_W, STAGE3_TILE_H, STAGE3_TILE_WAVEFRONT_THREADS, tiles_x, tiles_y,
           valid_edges, visited_tiles, ntiles, components, conflicts, max_abs_offset_k,
           STAGE3_STITCH_MIN_SUPPORT, STAGE3_STITCH_REQUIRE_MAJORITY);

cleanup:
    if (t0) cudaEventDestroy(t0);
    if (t1) cudaEventDestroy(t1);
    if (total0) cudaEventDestroy(total0);
    if (total1) cudaEventDestroy(total1);
}
