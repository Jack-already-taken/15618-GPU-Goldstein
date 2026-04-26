#include <cstdio>
#include <cstring>
#include <climits>
#include <cmath>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <utility>
#include <vector>
#include <queue>
#include <algorithm>
#include <cuda_runtime.h>

#include "unwrap_cuda.h"
#include "pi.h"

namespace {

enum : unsigned char {
    kPosRes    = 0x01,
    kNegRes    = 0x02,
    kBranchCut = 0x10,
    kBorder    = 0x20,
};

/* How many majority residues each block streams through shared memory in one
 * pass of k_match_residues. 1024 ints = 4 KB, well under any GPU's shared-mem
 * limit, and large enough to amortize the __syncthreads at chunk boundaries. */
constexpr int POS_CHUNK = 1024;


/* ------------------------------------------------------------------------- */
/*  Stage-3 unwrapping modes                                                 */
/* ------------------------------------------------------------------------- */
/*  Baseline: correctness-first serial region grower that runs in a single   */
/*  CUDA thread. It is slow, but it respects branch cuts and finishes every  */
/*  cut-free connected component.                                            */
/*                                                                           */
/*  Proposed block-wise version: unwrap one tile per thread block in         */
/*  parallel, then stitch neighboring tiles with integer 2*pi offsets along  */
/*  tile seams. This is much more parallel, but the simple implementation    */
/*  below assumes a single dominant connected component per tile.            */
/*  That makes it a useful prototype / optimization path, but not the        */
/*  default correctness path.                                                */

enum : int {
    /* Baseline = paper-style tiled flood-fill:
     * host orchestrates a tile frontier, GPU parallelizes within each tile. */
    kStage3ModeBaselineSerial = 0,
    /* Proposed = unwrap all tiles locally at once, then seam-stitch them. */
    kStage3ModeBlockwiseParallel = 1,
};

enum : int {
    kTaskStatusReached  = 1 << 0,
    kTaskStatusComplete = 1 << 1,
    kTaskStatusProgress = 1 << 2,
};

#ifndef UNWRAP_STAGE3_MODE
#define UNWRAP_STAGE3_MODE kStage3ModeBlockwiseParallel
#endif

constexpr int UNWRAP_TILE_W = 16;
constexpr int UNWRAP_TILE_H = 16;
constexpr int UNWRAP_TILE_PIXELS = UNWRAP_TILE_W * UNWRAP_TILE_H;

enum : int {
    kTileSeedAny = 0,
    kTileSeedFromLeft = 1,
    kTileSeedFromRight = 2,
    kTileSeedFromTop = 3,
    kTileSeedFromBottom = 4,
};

struct TileTask {
    int tile_x;
    int tile_y;
    int seed_mode;
    int _pad;
};

/* ------------------------------------------------------------------------- */
/*  Phase wrap + residue identification (unchanged logic; kept for context)  */
/* ------------------------------------------------------------------------- */

__device__ __forceinline__ float device_gradient(float p1, float p2)
{
    const float r = p1 - p2;

    // branchless single-wrap correction:
    // corr = +1 when r < -PI
    // corr = -1 when r > +PI
    // corr =  0 otherwise
    const float corr = (float)((r < -(float)PI) - (r > (float)PI));
    return __fmaf_rn(corr, (float)TWOPI, r);
}

__global__ void k_identify_residues(const float *phase, unsigned char *bitflags,
                                    int xsize, int ysize, int *d_num_res)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize - 1 || j >= ysize - 1)
        return;

    const int                k     = j * xsize + i;
    constexpr unsigned char  avoid = kBranchCut | kBorder;
    if ((bitflags[k]             & avoid) || (bitflags[k + 1]         & avoid)
        || (bitflags[k + 1 + xsize] & avoid) || (bitflags[k + xsize]   & avoid))
        return;

    const float r = device_gradient(phase[k + 1],         phase[k])
                  + device_gradient(phase[k + 1 + xsize], phase[k + 1])
                  + device_gradient(phase[k + xsize],     phase[k + 1 + xsize])
                  + device_gradient(phase[k],             phase[k + xsize]);

    const float thr = static_cast<float>(RESIDUE_THRESHOLD);
    if (r > thr)
        bitflags[k] |= kPosRes;
    else if (r < -thr)
        bitflags[k] |= kNegRes;
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


/* ------------------------------------------------------------------------- */
/*  Stage 3A: baseline tiled flood-fill (paper-style)                        */
/* ------------------------------------------------------------------------- */
/*  This baseline follows the paper's tiled integration idea more closely:   */
/*  the host expands a frontier of tiles, and each CUDA block unwraps one     */
/*  tile in shared memory using directional sweeps seeded from already-      */
/*  solved neighboring tiles. That makes the baseline non-serial: it is       */
/*  still synchronized tile-by-tile at the frontier level, but pixels inside  */
/*  each tile are processed cooperatively on the GPU.                         */

__global__ void k_init_unwrap_state(const float *__restrict__ phase,
                                    const unsigned char *__restrict__ bitflags,
                                    float *__restrict__ soln,
                                    int xsize, int ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize || j >= ysize)
        return;

    const int idx = j * xsize + i;
    soln[idx] = is_blocked_flag(bitflags[idx]) ? phase[idx] : nanf("");
}

/* AVOID-band fill: mirror the final loop of UnwrapAroundCutsFrontier in main.c.
 * For every blocked (branch-cut / border) pixel, pull a one-step-unwrapped value
 * from a non-blocked solved neighbor. We probe all four directions rather than
 * only left/top so the cleanup does not bias the branch-cut halo in one raster
 * direction. Like the C cleanup, this is a final cosmetic/consistency pass only;
 * branch-cut pixels are never used as traversal sources during the tiled flood-fill. */
__global__ void k_fill_avoid_band(float *__restrict__ soln,
                                  const float *__restrict__ phase,
                                  const unsigned char *__restrict__ bitflags,
                                  int xsize, int ysize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 0 || i >= xsize || j < 0 || j >= ysize)
        return;

    const int k = j * xsize + i;
    if (!is_blocked_flag(bitflags[k]))
        return;

    int nidx = -1;
    if (i > 0 && !is_blocked_flag(bitflags[k - 1]) && isfinite(soln[k - 1]))
        nidx = k - 1;
    else if (j > 0 && !is_blocked_flag(bitflags[k - xsize]) && isfinite(soln[k - xsize]))
        nidx = k - xsize;
    else if (i + 1 < xsize && !is_blocked_flag(bitflags[k + 1]) && isfinite(soln[k + 1]))
        nidx = k + 1;
    else if (j + 1 < ysize && !is_blocked_flag(bitflags[k + xsize]) && isfinite(soln[k + xsize]))
        nidx = k + xsize;

    if (nidx >= 0)
        soln[k] = soln[nidx] + device_gradient(phase[k], phase[nidx]);
}

/* Baseline seeding: a frontier task may only seed from the side implied by
 * its entry orientation. This avoids mixing incompatible 2*pi references from
 * multiple already-solved neighbors inside one tile. */
__device__ __forceinline__ void seed_from_left_neighbor(
    float s_phase[][UNWRAP_TILE_W],
    unsigned char s_flags[][UNWRAP_TILE_W],
    float s_soln[][UNWRAP_TILE_W],
    const float *__restrict__ phase,
    const unsigned char *__restrict__ bitflags,
    const float *__restrict__ solved,
    int tx, int ty, int gx, int gy, int xsize, int ysize,
    int *s_has_seed, int *s_progress, int *s_reached)
{
    if (tx != 0 || gx <= 0 || gy >= ysize)
        return;
    if (is_blocked_flag(s_flags[ty][tx]) || isfinite(s_soln[ty][tx]))
        return;

    const int nidx = gy * xsize + (gx - 1);
    if (!is_valid_unwrap_pixel(bitflags, nidx) || !isfinite(solved[nidx]))
        return;

    s_soln[ty][tx] = solved[nidx] + device_gradient(s_phase[ty][tx], phase[nidx]);
    atomicExch(s_has_seed, 1);
    atomicExch(s_progress, 1);
    atomicExch(s_reached, 1);
}

__device__ __forceinline__ void seed_from_right_neighbor(
    float s_phase[][UNWRAP_TILE_W],
    unsigned char s_flags[][UNWRAP_TILE_W],
    float s_soln[][UNWRAP_TILE_W],
    const float *__restrict__ phase,
    const unsigned char *__restrict__ bitflags,
    const float *__restrict__ solved,
    int tx, int ty, int gx, int gy, int xsize, int ysize,
    int *s_has_seed, int *s_progress, int *s_reached)
{
    if (tx != UNWRAP_TILE_W - 1 || gx + 1 >= xsize || gy >= ysize)
        return;
    if (is_blocked_flag(s_flags[ty][tx]) || isfinite(s_soln[ty][tx]))
        return;

    const int nidx = gy * xsize + (gx + 1);
    if (!is_valid_unwrap_pixel(bitflags, nidx) || !isfinite(solved[nidx]))
        return;

    s_soln[ty][tx] = solved[nidx] + device_gradient(s_phase[ty][tx], phase[nidx]);
    atomicExch(s_has_seed, 1);
    atomicExch(s_progress, 1);
    atomicExch(s_reached, 1);
}

__device__ __forceinline__ void seed_from_top_neighbor(
    float s_phase[][UNWRAP_TILE_W],
    unsigned char s_flags[][UNWRAP_TILE_W],
    float s_soln[][UNWRAP_TILE_W],
    const float *__restrict__ phase,
    const unsigned char *__restrict__ bitflags,
    const float *__restrict__ solved,
    int tx, int ty, int gx, int gy, int xsize, int ysize,
    int *s_has_seed, int *s_progress, int *s_reached)
{
    if (ty != 0 || gy <= 0 || gx >= xsize)
        return;
    if (is_blocked_flag(s_flags[ty][tx]) || isfinite(s_soln[ty][tx]))
        return;

    const int nidx = (gy - 1) * xsize + gx;
    if (!is_valid_unwrap_pixel(bitflags, nidx) || !isfinite(solved[nidx]))
        return;

    s_soln[ty][tx] = solved[nidx] + device_gradient(s_phase[ty][tx], phase[nidx]);
    atomicExch(s_has_seed, 1);
    atomicExch(s_progress, 1);
    atomicExch(s_reached, 1);
}

__device__ __forceinline__ void seed_from_bottom_neighbor(
    float s_phase[][UNWRAP_TILE_W],
    unsigned char s_flags[][UNWRAP_TILE_W],
    float s_soln[][UNWRAP_TILE_W],
    const float *__restrict__ phase,
    const unsigned char *__restrict__ bitflags,
    const float *__restrict__ solved,
    int tx, int ty, int gx, int gy, int xsize, int ysize,
    int *s_has_seed, int *s_progress, int *s_reached)
{
    if (ty != UNWRAP_TILE_H - 1 || gy + 1 >= ysize || gx >= xsize)
        return;
    if (is_blocked_flag(s_flags[ty][tx]) || isfinite(s_soln[ty][tx]))
        return;

    const int nidx = (gy + 1) * xsize + gx;
    if (!is_valid_unwrap_pixel(bitflags, nidx) || !isfinite(solved[nidx]))
        return;

    s_soln[ty][tx] = solved[nidx] + device_gradient(s_phase[ty][tx], phase[nidx]);
    atomicExch(s_has_seed, 1);
    atomicExch(s_progress, 1);
    atomicExch(s_reached, 1);
}

/* Paper-faithful directional sweeps (Algorithm 5 in the paper).
 *
 * Each sweep walks SERIALLY along one axis. Parallelism comes from running
 * many independent rows (or columns) concurrently, NOT from a wavefront
 * within a single row.  Concretely:
 *
 *   - Row sweeps (left/right): one thread per row does the full walk;
 *     we use the threads with tx == 0 and ty in 0..H-1.
 *   - Column sweeps (up/down): one thread per column; ty == 0 and tx in 0..W-1.
 *
 * Each walking thread maintains a running anchor: the last finite, non-cut
 * pixel it saw. NaN pixels with a live anchor get unwrapped; cuts kill the
 * anchor (segments past a cut need their own anchor, picked up from the
 * boundary seeding or from a perpendicular sweep that ran earlier).
 *
 * Caller MUST place __syncthreads() between sweeps so the perpendicular
 * sweep sees this sweep's writes to s_soln.
 */
__device__ inline void sweep_right(float s_phase[][UNWRAP_TILE_W],
                                   unsigned char s_flags[][UNWRAP_TILE_W],
                                   float s_soln[][UNWRAP_TILE_W],
                                   int tx, int ty,
                                   int *s_progress, int *s_reached)
{
    if (tx != 0) return;
    bool have_anchor = false;
    float prev_unwr = 0.0f;
    float prev_wrap = 0.0f;
    #pragma unroll
    for (int x = 0; x < UNWRAP_TILE_W; ++x) {
        if (is_blocked_flag(s_flags[ty][x])) {
            have_anchor = false;
            continue;
        }
        if (isfinite(s_soln[ty][x])) {
            prev_unwr = s_soln[ty][x];
            prev_wrap = s_phase[ty][x];
            have_anchor = true;
            continue;
        }
        if (!have_anchor) continue;
        const float wv = s_phase[ty][x];
        const float uv = prev_unwr + device_gradient(wv, prev_wrap);
        s_soln[ty][x] = uv;
        prev_unwr = uv;
        prev_wrap = wv;
        atomicExch(s_progress, 1);
        atomicExch(s_reached, 1);
    }
}

__device__ inline void sweep_left(float s_phase[][UNWRAP_TILE_W],
                                  unsigned char s_flags[][UNWRAP_TILE_W],
                                  float s_soln[][UNWRAP_TILE_W],
                                  int tx, int ty,
                                  int *s_progress, int *s_reached)
{
    if (tx != 0) return;
    bool have_anchor = false;
    float prev_unwr = 0.0f;
    float prev_wrap = 0.0f;
    #pragma unroll
    for (int x = UNWRAP_TILE_W - 1; x >= 0; --x) {
        if (is_blocked_flag(s_flags[ty][x])) {
            have_anchor = false;
            continue;
        }
        if (isfinite(s_soln[ty][x])) {
            prev_unwr = s_soln[ty][x];
            prev_wrap = s_phase[ty][x];
            have_anchor = true;
            continue;
        }
        if (!have_anchor) continue;
        const float wv = s_phase[ty][x];
        const float uv = prev_unwr + device_gradient(wv, prev_wrap);
        s_soln[ty][x] = uv;
        prev_unwr = uv;
        prev_wrap = wv;
        atomicExch(s_progress, 1);
        atomicExch(s_reached, 1);
    }
}

__device__ inline void sweep_down(float s_phase[][UNWRAP_TILE_W],
                                  unsigned char s_flags[][UNWRAP_TILE_W],
                                  float s_soln[][UNWRAP_TILE_W],
                                  int tx, int ty,
                                  int *s_progress, int *s_reached)
{
    if (ty != 0) return;
    bool have_anchor = false;
    float prev_unwr = 0.0f;
    float prev_wrap = 0.0f;
    #pragma unroll
    for (int y = 0; y < UNWRAP_TILE_H; ++y) {
        if (is_blocked_flag(s_flags[y][tx])) {
            have_anchor = false;
            continue;
        }
        if (isfinite(s_soln[y][tx])) {
            prev_unwr = s_soln[y][tx];
            prev_wrap = s_phase[y][tx];
            have_anchor = true;
            continue;
        }
        if (!have_anchor) continue;
        const float wv = s_phase[y][tx];
        const float uv = prev_unwr + device_gradient(wv, prev_wrap);
        s_soln[y][tx] = uv;
        prev_unwr = uv;
        prev_wrap = wv;
        atomicExch(s_progress, 1);
        atomicExch(s_reached, 1);
    }
}

__device__ inline void sweep_up(float s_phase[][UNWRAP_TILE_W],
                                unsigned char s_flags[][UNWRAP_TILE_W],
                                float s_soln[][UNWRAP_TILE_W],
                                int tx, int ty,
                                int *s_progress, int *s_reached)
{
    if (ty != 0) return;
    bool have_anchor = false;
    float prev_unwr = 0.0f;
    float prev_wrap = 0.0f;
    #pragma unroll
    for (int y = UNWRAP_TILE_H - 1; y >= 0; --y) {
        if (is_blocked_flag(s_flags[y][tx])) {
            have_anchor = false;
            continue;
        }
        if (isfinite(s_soln[y][tx])) {
            prev_unwr = s_soln[y][tx];
            prev_wrap = s_phase[y][tx];
            have_anchor = true;
            continue;
        }
        if (!have_anchor) continue;
        const float wv = s_phase[y][tx];
        const float uv = prev_unwr + device_gradient(wv, prev_wrap);
        s_soln[y][tx] = uv;
        prev_unwr = uv;
        prev_wrap = wv;
        atomicExch(s_progress, 1);
        atomicExch(s_reached, 1);
    }
}

__global__ void k_unwrap_frontier_tiles_baseline(
    const float *__restrict__ phase,
    const unsigned char *__restrict__ bitflags,
    float *__restrict__ soln,
    const TileTask *__restrict__ tasks,
    int task_count,
    int *__restrict__ task_status,
    int xsize, int ysize)
{
    const int task_id = blockIdx.x;
    if (task_id >= task_count)
        return;

    __shared__ float s_phase[UNWRAP_TILE_H][UNWRAP_TILE_W];
    __shared__ float s_soln [UNWRAP_TILE_H][UNWRAP_TILE_W];
    __shared__ unsigned char s_flags[UNWRAP_TILE_H][UNWRAP_TILE_W];
    __shared__ int s_seed_flat;
    __shared__ int s_has_seed;
    __shared__ int s_progress;
    __shared__ int s_reached;
    __shared__ int s_incomplete;

    const TileTask task = tasks[task_id];
    const int base_x = task.tile_x * UNWRAP_TILE_W;
    const int base_y = task.tile_y * UNWRAP_TILE_H;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int flat = ty * UNWRAP_TILE_W + tx;
    const int gx = base_x + tx;
    const int gy = base_y + ty;
    const bool in_bounds = (gx < xsize) && (gy < ysize);

    /* Load tile state from global. Out-of-bounds threads pretend to be a
     * blocked pixel so the sweeps treat them as cut-equivalent. */
    float ph = 0.0f;
    unsigned char fl = kBranchCut | kBorder;
    float prior = nanf("");
    if (in_bounds) {
        const int idx = gy * xsize + gx;
        ph = phase[idx];
        fl = bitflags[idx];
        prior = soln[idx];
    }

    s_phase[ty][tx] = ph;
    s_flags[ty][tx] = fl;
    /* Initial s_soln: blocked pixels carry their wrapped phase as a
     * placeholder (the AVOID-band pass cleans them up later); unblocked
     * pixels start at NaN unless the global buffer already has a finite
     * value for them (i.e., they were solved in a prior wave). */
    if (is_blocked_flag(fl))
        s_soln[ty][tx] = ph;
    else
        s_soln[ty][tx] = isfinite(prior) ? prior : nanf("");

    if (flat == 0) {
        s_seed_flat  = UNWRAP_TILE_PIXELS;
        s_has_seed   = 0;
        s_progress   = 0;
        s_reached    = 0;
        s_incomplete = 0;
    }
    __syncthreads();

    /* Any unblocked pixel that already carries a finite value (from a prior
     * wave) counts as "reached" and serves as an in-tile anchor. */
    if (in_bounds && !is_blocked_flag(fl) && isfinite(s_soln[ty][tx])) {
        atomicExch(&s_reached, 1);
        atomicExch(&s_has_seed, 1);
    }
    __syncthreads();

    /* Seed from exactly one side, matching the tile's entry orientation. */
    if (in_bounds && !is_blocked_flag(fl)) {
        switch (task.seed_mode) {
            case kTileSeedFromLeft:
                seed_from_left_neighbor(s_phase, s_flags, s_soln,
                                        phase, bitflags, soln,
                                        tx, ty, gx, gy, xsize, ysize,
                                        &s_has_seed, &s_progress, &s_reached);
                break;
            case kTileSeedFromRight:
                seed_from_right_neighbor(s_phase, s_flags, s_soln,
                                         phase, bitflags, soln,
                                         tx, ty, gx, gy, xsize, ysize,
                                         &s_has_seed, &s_progress, &s_reached);
                break;
            case kTileSeedFromTop:
                seed_from_top_neighbor(s_phase, s_flags, s_soln,
                                       phase, bitflags, soln,
                                       tx, ty, gx, gy, xsize, ysize,
                                       &s_has_seed, &s_progress, &s_reached);
                break;
            case kTileSeedFromBottom:
                seed_from_bottom_neighbor(s_phase, s_flags, s_soln,
                                          phase, bitflags, soln,
                                          tx, ty, gx, gy, xsize, ysize,
                                          &s_has_seed, &s_progress, &s_reached);
                break;
            case kTileSeedAny:
            default:
                if (s_has_seed == 0 && !isfinite(s_soln[ty][tx]))
                    atomicMin(&s_seed_flat, flat);
                break;
        }
    }
    __syncthreads();

    /* First reference tile: choose one arbitrary valid seed only if the tile
     * does not already inherit a solved anchor from an earlier revisit. */
    if (task.seed_mode == kTileSeedAny
        && s_has_seed == 0
        && s_seed_flat < UNWRAP_TILE_PIXELS
        && flat == s_seed_flat) {
        s_soln[ty][tx] = s_phase[ty][tx];
        atomicExch(&s_has_seed, 1);
        atomicExch(&s_progress, 1);
        atomicExch(&s_reached, 1);
    }
    __syncthreads();

    /* No anchor anywhere in the tile -- nothing to do this wave. */
    if (s_has_seed == 0 && s_reached == 0) {
        if (flat == 0)
            task_status[task_id] = 0;
        return;
    }

    /* Orientation-aware four-pass local solve. The entry side determines the
     * primary propagation direction; the orthogonal sweeps complete reachable
     * pixels without imposing a global raster bias. */
    switch (task.seed_mode) {
        case kTileSeedFromLeft:
            sweep_right(s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_down (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_up   (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_left (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            break;
        case kTileSeedFromRight:
            sweep_left (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_down (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_up   (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_right(s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            break;
        case kTileSeedFromTop:
            sweep_down (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_right(s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_left (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_up   (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            break;
        case kTileSeedFromBottom:
            sweep_up   (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_right(s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_left (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_down (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            break;
        case kTileSeedAny:
        default:
            sweep_up   (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_left (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_right(s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            sweep_down (s_phase, s_flags, s_soln, tx, ty, &s_progress, &s_reached);
            __syncthreads();
            break;
    }

    /* Tile-completion accounting. A tile is "complete" when every unblocked
     * pixel inside it has a finite value -- otherwise the host scheduler
     * will revisit it from another direction in a later wave. */
    if (in_bounds && !is_blocked_flag(s_flags[ty][tx])) {
        if (isfinite(s_soln[ty][tx]))
            atomicExch(&s_reached, 1);
        else
            atomicExch(&s_incomplete, 1);
    }
    __syncthreads();

    if (in_bounds && isfinite(s_soln[ty][tx])) {
        const int idx = gy * xsize + gx;
        soln[idx] = s_soln[ty][tx];
    }

    if (flat == 0) {
        int status = 0;
        if (s_reached)
            status |= kTaskStatusReached;
        if (!s_incomplete)
            status |= kTaskStatusComplete;
        if (s_progress)
            status |= kTaskStatusProgress;
        task_status[task_id] = status;
    }
}

/* ------------------------------------------------------------------------- */
/*  Stage 3B: full block-wise parallel unwrap                                */
/* ------------------------------------------------------------------------- */
/*  Step 1 (k_unwrap_tiles_local): every tile is unwrapped independently in  */
/*           shared memory. We first label cut-isolated connected components */
/*           inside the tile, then seed *every* component root and run a    */
/*           shared-memory BFS. Each component carries its own arbitrary     */
/*           2*pi reference, exactly mirroring the C frontier's per-piece    */
/*           seeding -- so a branch cut slicing a tile no longer leaves any  */
/*           pixel NaN or rolls a second component into the first one's     */
/*           offset.                                                         */
/*                                                                           */
/*  Step 2: collect seam votes on the GPU, then solve only the compact       */
/*           seam graph on the host. Nodes are (tile_id, local_component_id) */
/*           slots that actually touch a seam; edges are seam-pixel votes     */
/*           for the integer 2*pi offset between two specific components on   */
/*           opposite sides of a tile boundary. BFS per connected subgraph    */
/*           yields a per-(tile, component) offset.                           */
/*                                                                           */
/*  Step 3 (k_apply_tile_offsets): each pixel reads its component id and    */
/*           adds offset_lut[tile_id, comp] * 2*pi to the local solution.   */

__global__ void k_unwrap_tiles_local(const float *__restrict__ phase,
                                     const unsigned char *__restrict__ bitflags,
                                     float *__restrict__ soln,
                                     int *__restrict__ tile_component,
                                     int xsize, int ysize)
{
    __shared__ float s_phase[UNWRAP_TILE_H][UNWRAP_TILE_W];
    __shared__ float s_soln [UNWRAP_TILE_H][UNWRAP_TILE_W];
    __shared__ unsigned char s_flags[UNWRAP_TILE_H][UNWRAP_TILE_W];
    __shared__ int s_comp[UNWRAP_TILE_H][UNWRAP_TILE_W];
    __shared__ int s_changed;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = blockIdx.x * UNWRAP_TILE_W + tx;
    const int gy = blockIdx.y * UNWRAP_TILE_H + ty;
    const int flat = ty * UNWRAP_TILE_W + tx;
    const bool in_bounds = (gx < xsize) && (gy < ysize);

    float ph = 0.0f;
    unsigned char fl = kBranchCut | kBorder;
    if (in_bounds) {
        const int idx = gy * xsize + gx;
        ph = phase[idx];
        fl = bitflags[idx];
    }

    s_phase[ty][tx] = ph;
    s_flags[ty][tx] = fl;
    s_soln [ty][tx] = is_blocked_flag(fl) ? ph : nanf("");
    s_comp [ty][tx] = is_blocked_flag(fl) ? INT_MAX : flat;
    __syncthreads();

    /* Label cut-isolated components: each unblocked pixel converges to the
     * smallest flat-index reachable through a chain of unblocked 4-neighbors.
     * Up to UNWRAP_TILE_PIXELS-1 iterations is the worst-case diameter; the
     * s_changed early-exit makes the common (single-component) case cheap. */
    for (int iter = 0; iter < UNWRAP_TILE_PIXELS; ++iter) {
        if (flat == 0)
            s_changed = 0;
        __syncthreads();

        if (!is_blocked_flag(s_flags[ty][tx])) {
            int best = s_comp[ty][tx];
            if (tx > 0 && !is_blocked_flag(s_flags[ty][tx - 1])) {
                const int v = s_comp[ty][tx - 1];
                if (v < best) best = v;
            }
            if (tx + 1 < UNWRAP_TILE_W && gx + 1 < xsize
                && !is_blocked_flag(s_flags[ty][tx + 1])) {
                const int v = s_comp[ty][tx + 1];
                if (v < best) best = v;
            }
            if (ty > 0 && !is_blocked_flag(s_flags[ty - 1][tx])) {
                const int v = s_comp[ty - 1][tx];
                if (v < best) best = v;
            }
            if (ty + 1 < UNWRAP_TILE_H && gy + 1 < ysize
                && !is_blocked_flag(s_flags[ty + 1][tx])) {
                const int v = s_comp[ty + 1][tx];
                if (v < best) best = v;
            }
            if (best < s_comp[ty][tx]) {
                s_comp[ty][tx] = best;
                atomicExch(&s_changed, 1);
            }
        }
        __syncthreads();
        if (s_changed == 0)
            break;
    }

    /* Seed every component root with its raw phase. Since components are
     * cut-isolated by construction, the BFS below cannot blend across them. */
    if (in_bounds && !is_blocked_flag(s_flags[ty][tx])
        && s_comp[ty][tx] == flat)
        s_soln[ty][tx] = s_phase[ty][tx];
    __syncthreads();

    for (int iter = 0; iter < UNWRAP_TILE_PIXELS; ++iter) {
        if (flat == 0)
            s_changed = 0;
        __syncthreads();

        if (in_bounds && !is_blocked_flag(s_flags[ty][tx]) && isnan(s_soln[ty][tx])) {
            bool found = false;
            float cand = 0.0f;

            if (!found && tx > 0 && !is_blocked_flag(s_flags[ty][tx - 1])
                && !isnan(s_soln[ty][tx - 1])) {
                cand = s_soln[ty][tx - 1]
                     + device_gradient(s_phase[ty][tx], s_phase[ty][tx - 1]);
                found = true;
            }
            if (!found && tx + 1 < UNWRAP_TILE_W && gx + 1 < xsize
                && !is_blocked_flag(s_flags[ty][tx + 1]) && !isnan(s_soln[ty][tx + 1])) {
                cand = s_soln[ty][tx + 1]
                     + device_gradient(s_phase[ty][tx], s_phase[ty][tx + 1]);
                found = true;
            }
            if (!found && ty > 0 && !is_blocked_flag(s_flags[ty - 1][tx])
                && !isnan(s_soln[ty - 1][tx])) {
                cand = s_soln[ty - 1][tx]
                     + device_gradient(s_phase[ty][tx], s_phase[ty - 1][tx]);
                found = true;
            }
            if (!found && ty + 1 < UNWRAP_TILE_H && gy + 1 < ysize
                && !is_blocked_flag(s_flags[ty + 1][tx]) && !isnan(s_soln[ty + 1][tx])) {
                cand = s_soln[ty + 1][tx]
                     + device_gradient(s_phase[ty][tx], s_phase[ty + 1][tx]);
                found = true;
            }

            if (found) {
                s_soln[ty][tx] = cand;
                atomicExch(&s_changed, 1);
            }
        }

        __syncthreads();
        if (s_changed == 0)
            break;
    }

    if (in_bounds) {
        const int idx = gy * xsize + gx;
        soln[idx] = s_soln[ty][tx];
        tile_component[idx] = is_blocked_flag(s_flags[ty][tx]) ? -1 : s_comp[ty][tx];
    }
}

__global__ void k_apply_tile_offsets(float *__restrict__ soln,
                                     const float *__restrict__ phase,
                                     const unsigned char *__restrict__ bitflags,
                                     const int *__restrict__ tile_component,
                                     const int *__restrict__ offset_lut,
                                     int xsize, int ysize, int tiles_x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize || j >= ysize)
        return;

    const int idx = j * xsize + i;
    if (!is_valid_unwrap_pixel(bitflags, idx)) {
        soln[idx] = phase[idx];
        return;
    }

    const int comp = tile_component[idx];
    if (comp < 0 || isnan(soln[idx])) {
        soln[idx] = phase[idx];
        return;
    }

    const int tile_x = i / UNWRAP_TILE_W;
    const int tile_y = j / UNWRAP_TILE_H;
    const int tile_id = tile_y * tiles_x + tile_x;
    soln[idx] += (float)offset_lut[tile_id * UNWRAP_TILE_PIXELS + comp]
               * (float)TWOPI;
}


struct SeamVote {
    int slot_a;
    int slot_b;
    int delta_k;
    int weight;
};

__global__ void k_collect_seam_votes(const float *__restrict__ phase,
                                     const unsigned char *__restrict__ bitflags,
                                     const float *__restrict__ local_soln,
                                     const int *__restrict__ tile_component,
                                     SeamVote *__restrict__ seam_votes,
                                     int *__restrict__ vote_count,
                                     int xsize, int ysize, int tiles_x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= xsize || j >= ysize)
        return;

    const int ia = j * xsize + i;

    if (i + 1 < xsize) {
        const int tile_x_a = i / UNWRAP_TILE_W;
        const int tile_x_b = (i + 1) / UNWRAP_TILE_W;
        if (tile_x_a != tile_x_b) {
            const int ib = ia + 1;
            if (is_valid_unwrap_pixel(bitflags, ia) &&
                is_valid_unwrap_pixel(bitflags, ib)) {
                const int ca = tile_component[ia];
                const int cb = tile_component[ib];
                const float ua = local_soln[ia];
                const float ub = local_soln[ib];
                if (ca >= 0 && cb >= 0 && isfinite(ua) && isfinite(ub)) {
                    const int tile_y = j / UNWRAP_TILE_H;
                    const int tile_a = tile_y * tiles_x + tile_x_a;
                    const int tile_b = tile_y * tiles_x + tile_x_b;
                    const float grad = device_gradient(phase[ib], phase[ia]);
                    const float kf = (ua + grad - ub) / (float)TWOPI;
                    const int out = atomicAdd(vote_count, 1);
                    seam_votes[out].slot_a = tile_a * UNWRAP_TILE_PIXELS + ca;
                    seam_votes[out].slot_b = tile_b * UNWRAP_TILE_PIXELS + cb;
                    seam_votes[out].delta_k = __float2int_rn(kf);
                    seam_votes[out].weight = 1;
                }
            }
        }
    }

    if (j + 1 < ysize) {
        const int tile_y_a = j / UNWRAP_TILE_H;
        const int tile_y_b = (j + 1) / UNWRAP_TILE_H;
        if (tile_y_a != tile_y_b) {
            const int ib = ia + xsize;
            if (is_valid_unwrap_pixel(bitflags, ia) &&
                is_valid_unwrap_pixel(bitflags, ib)) {
                const int ca = tile_component[ia];
                const int cb = tile_component[ib];
                const float ua = local_soln[ia];
                const float ub = local_soln[ib];
                if (ca >= 0 && cb >= 0 && isfinite(ua) && isfinite(ub)) {
                    const int tile_x = i / UNWRAP_TILE_W;
                    const int tile_a = tile_y_a * tiles_x + tile_x;
                    const int tile_b = tile_y_b * tiles_x + tile_x;
                    const float grad = device_gradient(phase[ib], phase[ia]);
                    const float kf = (ua + grad - ub) / (float)TWOPI;
                    const int out = atomicAdd(vote_count, 1);
                    seam_votes[out].slot_a = tile_a * UNWRAP_TILE_PIXELS + ca;
                    seam_votes[out].slot_b = tile_b * UNWRAP_TILE_PIXELS + cb;
                    seam_votes[out].delta_k = __float2int_rn(kf);
                    seam_votes[out].weight = 1;
                }
            }
        }
    }
}

static inline std::uint64_t pack_edge_key(int a, int b)
{
    return (std::uint64_t)(std::uint32_t)a << 32
         | (std::uint64_t)(std::uint32_t)b;
}

static void solve_slot_offsets_from_votes(const SeamVote *h_votes,
                                          int vote_count,
                                          int slot_count,
                                          std::vector<int> &offset_lut)
{
    struct EdgeAccum {
        std::unordered_map<int, int> hist;
        int total = 0;
    };
    struct Edge {
        int to;
        int delta_k;
        int support;
        int total;
    };

    std::unordered_map<int, int> node_id_of_slot;
    node_id_of_slot.reserve((size_t)vote_count * 2 + 1);

    std::vector<int> slot_of_node;
    slot_of_node.reserve((size_t)vote_count * 2);

    auto get_or_add_node = [&](int slot) -> int {
        auto it = node_id_of_slot.find(slot);
        if (it != node_id_of_slot.end())
            return it->second;
        const int id = (int)slot_of_node.size();
        node_id_of_slot.emplace(slot, id);
        slot_of_node.push_back(slot);
        return id;
    };

    std::unordered_map<std::uint64_t, EdgeAccum> edge_map;
    edge_map.reserve((size_t)vote_count * 2 + 1);

    for (int i = 0; i < vote_count; ++i) {
        int slot_a = h_votes[i].slot_a;
        int slot_b = h_votes[i].slot_b;
        int k = h_votes[i].delta_k;
        if (slot_a < 0 || slot_b < 0 || slot_a == slot_b)
            continue;

        int a = get_or_add_node(slot_a);
        int b = get_or_add_node(slot_b);
        if (a == b)
            continue;
        if (a > b) {
            std::swap(a, b);
            k = -k;
        }

        EdgeAccum &acc = edge_map[pack_edge_key(a, b)];
        acc.hist[k] += 1;
        acc.total += h_votes[i].weight;
    }

    const int node_count = (int)slot_of_node.size();
    std::vector<std::vector<Edge>> graph((size_t)node_count);

    constexpr int kMinSeamSupport = 2;
    constexpr double kMinSeamAgreement = 0.60;

    for (const auto &kv : edge_map) {
        const int a = (int)(kv.first >> 32);
        const int b = (int)(kv.first & 0xFFFFFFFFu);
        const EdgeAccum &acc = kv.second;

        int best_k = 0;
        int best_count = 0;
        for (const auto &vote : acc.hist) {
            if (vote.second > best_count) {
                best_k = vote.first;
                best_count = vote.second;
            }
        }

        if (best_count < kMinSeamSupport)
            continue;
        const double agree = (double)best_count / (double)acc.total;
        if (agree < kMinSeamAgreement)
            continue;

        graph[a].push_back({b, best_k, best_count, acc.total});
        graph[b].push_back({a, -best_k, best_count, acc.total});
    }

    for (int u = 0; u < node_count; ++u) {
        std::sort(graph[u].begin(), graph[u].end(),
                  [](const Edge &lhs, const Edge &rhs) {
                      if (lhs.support != rhs.support)
                          return lhs.support > rhs.support;
                      return lhs.total > rhs.total;
                  });
    }

    std::vector<int> node_offsets((size_t)node_count, INT_MAX);
    std::queue<int> q;
    for (int seed = 0; seed < node_count; ++seed) {
        if (node_offsets[seed] != INT_MAX)
            continue;
        node_offsets[seed] = 0;
        q.push(seed);
        while (!q.empty()) {
            const int u = q.front();
            q.pop();
            for (const Edge &e : graph[u]) {
                const int cand = node_offsets[u] + e.delta_k;
                if (node_offsets[e.to] == INT_MAX) {
                    node_offsets[e.to] = cand;
                    q.push(e.to);
                }
            }
        }
    }

    offset_lut.assign((size_t)slot_count, 0);
    for (int n = 0; n < node_count; ++n) {
        const int slot = slot_of_node[n];
        if (slot >= 0 && slot < slot_count && node_offsets[n] != INT_MAX)
            offset_lut[(size_t)slot] = node_offsets[n];
    }
}

static bool tile_has_valid_pixel(const unsigned char *h_bitflags,
                                 int tile_x, int tile_y,
                                 int xsize, int ysize)
{
    const int x0 = tile_x * UNWRAP_TILE_W;
    const int y0 = tile_y * UNWRAP_TILE_H;
    const int x1 = std::min(xsize, x0 + UNWRAP_TILE_W);
    const int y1 = std::min(ysize, y0 + UNWRAP_TILE_H);
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            if (!is_blocked_flag(h_bitflags[y * xsize + x]))
                return true;
        }
    }
    return false;
}

static int mode_from_solved_neighbor(int cur_tx, int cur_ty,
                                     int n_tx, int n_ty)
{
    if (n_tx == cur_tx - 1 && n_ty == cur_ty) return kTileSeedFromLeft;
    if (n_tx == cur_tx + 1 && n_ty == cur_ty) return kTileSeedFromRight;
    if (n_tx == cur_tx && n_ty == cur_ty - 1) return kTileSeedFromTop;
    if (n_tx == cur_tx && n_ty == cur_ty + 1) return kTileSeedFromBottom;
    return kTileSeedAny;
}

__global__ void k_noop(void) {}

/* ------------------------------------------------------------------------- */

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

/* Shared residue-capacity formula used by both the allocator and anyone who
 * wants to audit it later. ~5% of pixels is a comfortable upper bound for
 * real data; bump the divisor if you process pathologically noisy inputs. */
static inline int residue_capacity(int length)
{
    return length / 5 + 4;
}

} /* namespace */

/* =========================================================================== */

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
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_soln, (size_t)length * sizeof(float));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_residue_count, sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    /* ---- Stage 2 scratch buffers --------------------------------------- */
    /* d_pos_residues / d_neg_residues: (cap + 1) ints each, counter at [0]. */
    /* d_pairs: 2 * cap ints, covers worst case where all residues are one   */
    /*          sign (n_maj == cap) and we emit one pair per residue.        */
    const int cap = residue_capacity(length);

    e = cudaMalloc((void **)&out->d_pos_residues, (size_t)(cap + 1) * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_neg_residues, (size_t)(cap + 1) * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    e = cudaMalloc((void **)&out->d_pairs, 2 * (size_t)cap * sizeof(int));
    if (e != cudaSuccess) { unwrap_cuda_device_bufs_free(out); return (int)e; }

    return 0;
}

extern "C" void unwrap_cuda_device_bufs_free(UnwrapCudaDeviceBufs *buf)
{
    if (!buf)
        return;
    cudaFree(buf->d_phase);
    cudaFree(buf->d_bitflags);
    cudaFree(buf->d_pos_residues);
    cudaFree(buf->d_neg_residues);
    cudaFree(buf->d_pairs);
    cudaFree(buf->d_soln);
    cudaFree(buf->d_residue_count);
    memset(buf, 0, sizeof(*buf));
}

extern "C" int unwrap_cuda_launch_residue_identification(
    float *h_phase, unsigned char *h_bitflags, const UnwrapCudaDeviceBufs *dev,
    int xsize, int ysize, int length)
{
    if (!h_phase || !h_bitflags || !dev || !dev->d_phase || !dev->d_bitflags
        || !dev->d_residue_count || xsize < 2 || ysize < 2
        || length != xsize * ysize)
        return -1;

    cudaError_t e;
    int         h_count = 0;

    if ((e = cudaMemcpy(dev->d_phase, h_phase,
                        (size_t)length * sizeof(float),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return cuda_fail(e, "H2D phase");
    if ((e = cudaMemcpy(dev->d_bitflags, h_bitflags,
                        (size_t)length * sizeof(unsigned char),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return cuda_fail(e, "H2D bitflags");
    if ((e = cudaMemset(dev->d_residue_count, 0, sizeof(int))) != cudaSuccess)
        return cuda_fail(e, "memset count");

    dim3 block(16, 16);
    dim3 grid = residue_grid(xsize, ysize);
    k_identify_residues<<<grid, block>>>(dev->d_phase, dev->d_bitflags,
                                         xsize, ysize, dev->d_residue_count);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return cuda_fail(e, "k_identify_residues");
    if ((e = cudaDeviceSynchronize()) != cudaSuccess)
        return cuda_fail(e, "sync residues");

    if ((e = cudaMemcpy(h_bitflags, dev->d_bitflags,
                        (size_t)length * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return cuda_fail(e, "D2H bitflags");
    if ((e = cudaMemcpy(&h_count, dev->d_residue_count, sizeof(int),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
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

extern "C" void unwrap_cuda_launch_unwrapping(float *h_phase, unsigned char *h_bitflags,
                                              float *h_soln,
                                              const UnwrapCudaDeviceBufs *dev,
                                              int xsize, int ysize, int length)
{
    if (length < 1 || !h_phase || !h_bitflags || !h_soln || !dev
        || !dev->d_phase || !dev->d_bitflags || !dev->d_soln
        || length != xsize * ysize)
        return;

    cudaError_t e;
    if ((e = cudaMemcpy(dev->d_phase, h_phase,
                        (size_t)length * sizeof(float),
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
        cuda_fail(e, "unwrap H2D phase");
        return;
    }
    if ((e = cudaMemcpy(dev->d_bitflags, h_bitflags,
                        (size_t)length * sizeof(unsigned char),
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
        cuda_fail(e, "unwrap H2D bitflags");
        return;
    }

#if UNWRAP_STAGE3_MODE == kStage3ModeBaselineSerial

    {
        dim3 init_block(16, 16);
        dim3 init_grid(div_up_int(xsize, 16), div_up_int(ysize, 16));
        k_init_unwrap_state<<<init_grid, init_block>>>(dev->d_phase, dev->d_bitflags,
                                                       dev->d_soln, xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_init_unwrap_state");
            return;
        }
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) {
            cuda_fail(e, "sync init unwrap state");
            return;
        }

        const int tiles_x = div_up_int(xsize, UNWRAP_TILE_W);
        const int tiles_y = div_up_int(ysize, UNWRAP_TILE_H);
        const int tile_count = tiles_x * tiles_y;

        std::vector<unsigned char> tile_has_signal((size_t)tile_count, 0);
        std::vector<unsigned char> tile_reached((size_t)tile_count, 0);
        std::vector<unsigned char> tile_complete((size_t)tile_count, 0);
        for (int ty = 0; ty < tiles_y; ++ty) {
            for (int tx = 0; tx < tiles_x; ++tx) {
                const int tile_id = ty * tiles_x + tx;
                tile_has_signal[tile_id] = tile_has_valid_pixel(h_bitflags, tx, ty,
                                                                xsize, ysize) ? 1 : 0;
            }
        }

        /* One pair of GPU buffers shared across all waves. The lambda below
         * just reuses these on every call -- previously this code did
         * cudaMalloc + cudaFree for each wave, which on a 4096^2 image with
         * a hundred-plus waves was a measurable fraction of total Stage 3
         * time. Both buffers are sized to tile_count, the largest possible
         * wave. */
        TileTask *d_tasks_pool = nullptr;
        int      *d_status_pool = nullptr;
        if ((e = cudaMalloc((void **)&d_tasks_pool,
                            (size_t)tile_count * sizeof(TileTask))) != cudaSuccess) {
            cuda_fail(e, "malloc baseline tasks pool");
            return;
        }
        if ((e = cudaMalloc((void **)&d_status_pool,
                            (size_t)tile_count * sizeof(int))) != cudaSuccess) {
            cuda_fail(e, "malloc baseline status pool");
            cudaFree(d_tasks_pool);
            return;
        }

        auto launch_tile_wave = [&](const std::vector<TileTask> &tasks,
                                    std::vector<int> &h_status) -> bool {
            h_status.assign(tasks.size(), 0);
            if (tasks.empty())
                return true;

            if ((e = cudaMemcpy(d_tasks_pool, tasks.data(),
                                tasks.size() * sizeof(TileTask),
                                cudaMemcpyHostToDevice)) != cudaSuccess) {
                cuda_fail(e, "H2D baseline tasks");
                return false;
            }

            dim3 tile_block(UNWRAP_TILE_W, UNWRAP_TILE_H);
            k_unwrap_frontier_tiles_baseline<<<(unsigned)tasks.size(), tile_block>>>(
                dev->d_phase, dev->d_bitflags, dev->d_soln,
                d_tasks_pool, (int)tasks.size(), d_status_pool, xsize, ysize);
            if ((e = cudaGetLastError()) != cudaSuccess) {
                cuda_fail(e, "k_unwrap_frontier_tiles_baseline");
                return false;
            }
            if ((e = cudaDeviceSynchronize()) != cudaSuccess) {
                cuda_fail(e, "sync baseline tile wave");
                return false;
            }
            if ((e = cudaMemcpy(h_status.data(), d_status_pool,
                                tasks.size() * sizeof(int),
                                cudaMemcpyDeviceToHost)) != cudaSuccess) {
                cuda_fail(e, "D2H baseline task status");
                return false;
            }

            return true;
        };

        while (true) {
            int seed_tile = -1;
            for (int tile_id = 0; tile_id < tile_count; ++tile_id) {
                if (tile_has_signal[tile_id] && !tile_complete[tile_id]) {
                    seed_tile = tile_id;
                    break;
                }
            }
            if (seed_tile < 0)
                break;

            std::vector<TileTask> seed_tasks(1);
            seed_tasks[0].tile_x = seed_tile % tiles_x;
            seed_tasks[0].tile_y = seed_tile / tiles_x;
            seed_tasks[0].seed_mode = kTileSeedAny;
            seed_tasks[0]._pad = 0;

            std::vector<int> h_status;
            if (!launch_tile_wave(seed_tasks, h_status)) {
                cudaFree(d_tasks_pool);
                cudaFree(d_status_pool);
                return;
            }

            if (h_status[0] & kTaskStatusReached)
                tile_reached[seed_tile] = 1;
            if (h_status[0] & kTaskStatusComplete)
                tile_complete[seed_tile] = 1;

            while (true) {
                std::vector<int> candidate_mode((size_t)tile_count, -1);
                for (int tile_id = 0; tile_id < tile_count; ++tile_id) {
                    if (!tile_reached[tile_id])
                        continue;
                    const int tx = tile_id % tiles_x;
                    const int ty = tile_id / tiles_x;

                    const int nx[4] = {tx - 1, tx + 1, tx, tx};
                    const int ny[4] = {ty, ty, ty - 1, ty + 1};
                    for (int k = 0; k < 4; ++k) {
                        const int cx = nx[k];
                        const int cy = ny[k];
                        if (cx < 0 || cx >= tiles_x || cy < 0 || cy >= tiles_y)
                            continue;
                        const int cid = cy * tiles_x + cx;
                        if (tile_complete[cid] || !tile_has_signal[cid])
                            continue;
                        if (candidate_mode[cid] < 0)
                            candidate_mode[cid] = mode_from_solved_neighbor(cx, cy, tx, ty);
                    }
                }

                std::vector<TileTask> wave;
                std::vector<int> wave_ids;
                wave.reserve((size_t)tile_count);
                wave_ids.reserve((size_t)tile_count);
                for (int tile_id = 0; tile_id < tile_count; ++tile_id) {
                    if (candidate_mode[tile_id] < 0)
                        continue;
                    TileTask task;
                    task.tile_x = tile_id % tiles_x;
                    task.tile_y = tile_id / tiles_x;
                    task.seed_mode = candidate_mode[tile_id];
                    task._pad = 0;
                    wave.push_back(task);
                    wave_ids.push_back(tile_id);
                }

                if (wave.empty())
                    break;
                if (!launch_tile_wave(wave, h_status)) {
                    cudaFree(d_tasks_pool);
                    cudaFree(d_status_pool);
                    return;
                }

                int n_progress = 0;
                for (size_t i = 0; i < wave.size(); ++i) {
                    const int tile_id = wave_ids[i];
                    const int status = h_status[i];
                    if (status & kTaskStatusReached)
                        tile_reached[tile_id] = 1;
                    if (status & kTaskStatusComplete)
                        tile_complete[tile_id] = 1;
                    if (status & kTaskStatusProgress)
                        ++n_progress;
                }
                if (n_progress == 0)
                    break;
            }
        }

        cudaFree(d_tasks_pool);
        cudaFree(d_status_pool);
    }

#elif UNWRAP_STAGE3_MODE == kStage3ModeBlockwiseParallel

    {
        const int tiles_x = div_up_int(xsize, UNWRAP_TILE_W);
        const int tiles_y = div_up_int(ysize, UNWRAP_TILE_H);
        const int tile_count = tiles_x * tiles_y;

        int *d_tile_component = nullptr;
        if ((e = cudaMalloc((void **)&d_tile_component,
                            (size_t)length * sizeof(int))) != cudaSuccess) {
            cuda_fail(e, "malloc tile_component");
            return;
        }

        dim3 block(UNWRAP_TILE_W, UNWRAP_TILE_H);
        dim3 grid(tiles_x, tiles_y);
        k_unwrap_tiles_local<<<grid, block>>>(dev->d_phase, dev->d_bitflags,
                                              dev->d_soln, d_tile_component,
                                              xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_unwrap_tiles_local");
            cudaFree(d_tile_component);
            return;
        }
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) {
            cuda_fail(e, "sync local tile unwrap");
            cudaFree(d_tile_component);
            return;
        }

        const int max_seam_votes =
            ysize * std::max(0, tiles_x - 1) +
            xsize * std::max(0, tiles_y - 1);

        SeamVote *d_seam_votes = nullptr;
        int *d_vote_count = nullptr;
        int *d_offset_lut = nullptr;

        if ((e = cudaMalloc((void **)&d_seam_votes,
                            (size_t)std::max(1, max_seam_votes) * sizeof(SeamVote)))
            != cudaSuccess) {
            cuda_fail(e, "malloc seam_votes");
            cudaFree(d_tile_component);
            return;
        }
        if ((e = cudaMalloc((void **)&d_vote_count, sizeof(int))) != cudaSuccess) {
            cuda_fail(e, "malloc vote_count");
            cudaFree(d_seam_votes);
            cudaFree(d_tile_component);
            return;
        }
        if ((e = cudaMemset(d_vote_count, 0, sizeof(int))) != cudaSuccess) {
            cuda_fail(e, "memset vote_count");
            cudaFree(d_vote_count);
            cudaFree(d_seam_votes);
            cudaFree(d_tile_component);
            return;
        }

        dim3 seam_block(16, 16);
        dim3 seam_grid(div_up_int(xsize, 16), div_up_int(ysize, 16));
        k_collect_seam_votes<<<seam_grid, seam_block>>>(
            dev->d_phase, dev->d_bitflags, dev->d_soln, d_tile_component,
            d_seam_votes, d_vote_count, xsize, ysize, tiles_x);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_collect_seam_votes");
            cudaFree(d_vote_count);
            cudaFree(d_seam_votes);
            cudaFree(d_tile_component);
            return;
        }
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) {
            cuda_fail(e, "sync seam vote collection");
            cudaFree(d_vote_count);
            cudaFree(d_seam_votes);
            cudaFree(d_tile_component);
            return;
        }

        int h_vote_count = 0;
        if ((e = cudaMemcpy(&h_vote_count, d_vote_count, sizeof(int),
                            cudaMemcpyDeviceToHost)) != cudaSuccess) {
            cuda_fail(e, "D2H vote_count");
            cudaFree(d_vote_count);
            cudaFree(d_seam_votes);
            cudaFree(d_tile_component);
            return;
        }

        std::vector<SeamVote> h_seam_votes((size_t)h_vote_count);
        if (h_vote_count > 0) {
            if ((e = cudaMemcpy(h_seam_votes.data(), d_seam_votes,
                                (size_t)h_vote_count * sizeof(SeamVote),
                                cudaMemcpyDeviceToHost)) != cudaSuccess) {
                cuda_fail(e, "D2H seam_votes");
                cudaFree(d_vote_count);
                cudaFree(d_seam_votes);
                cudaFree(d_tile_component);
                return;
            }
        }
        cudaFree(d_vote_count);
        cudaFree(d_seam_votes);

        std::vector<int> h_offset_lut;
        solve_slot_offsets_from_votes(h_seam_votes.data(), h_vote_count,
                                      tile_count * UNWRAP_TILE_PIXELS,
                                      h_offset_lut);

        if ((e = cudaMalloc((void **)&d_offset_lut,
                            (size_t)tile_count * UNWRAP_TILE_PIXELS * sizeof(int)))
            != cudaSuccess) {
            cuda_fail(e, "malloc offset_lut");
            cudaFree(d_tile_component);
            return;
        }
        if ((e = cudaMemcpy(d_offset_lut, h_offset_lut.data(),
                            (size_t)tile_count * UNWRAP_TILE_PIXELS * sizeof(int),
                            cudaMemcpyHostToDevice)) != cudaSuccess) {
            cuda_fail(e, "H2D offset_lut");
            cudaFree(d_offset_lut);
            cudaFree(d_tile_component);
            return;
        }

        dim3 apply_block(16, 16);
        dim3 apply_grid(div_up_int(xsize, 16), div_up_int(ysize, 16));
        k_apply_tile_offsets<<<apply_grid, apply_block>>>(
            dev->d_soln, dev->d_phase, dev->d_bitflags,
            d_tile_component, d_offset_lut,
            xsize, ysize, tiles_x);
        e = cudaGetLastError();
        cudaFree(d_offset_lut);
        cudaFree(d_tile_component);
        if (e != cudaSuccess) {
            cuda_fail(e, "k_apply_tile_offsets");
            return;
        }
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) {
            cuda_fail(e, "sync tile-stitch unwrap");
            return;
        }
    }

#else
#error "Unsupported UNWRAP_STAGE3_MODE"
#endif

    /* AVOID-band fill: stamp branch-cut / border pixels with a one-step
     * unwrap from a non-blocked neighbor, matching the tail of
     * UnwrapAroundCutsFrontier. Without this, residue pixels (which all carry
     * the BRANCH_CUT flag after stage 2) display as raw wrapped phase. */
    {
        dim3 fill_block(16, 16);
        dim3 fill_grid(div_up_int(xsize, 16), div_up_int(ysize, 16));
        k_fill_avoid_band<<<fill_grid, fill_block>>>(dev->d_soln, dev->d_phase,
                                                     dev->d_bitflags,
                                                     xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) {
            cuda_fail(e, "k_fill_avoid_band");
            return;
        }
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) {
            cuda_fail(e, "sync fill avoid band");
            return;
        }
    }

    if ((e = cudaMemcpy(h_soln, dev->d_soln,
                        (size_t)length * sizeof(float),
                        cudaMemcpyDeviceToHost)) != cudaSuccess) {
        cuda_fail(e, "unwrap D2H soln");
        return;
    }

    for (int idx = 0; idx < length; ++idx) {
        if (!std::isfinite(h_soln[idx]))
            h_soln[idx] = h_phase[idx];
    }
}
