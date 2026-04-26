#include <cstdio>
#include <cstring>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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
#define STAGE1_RESIDUE_TILE_W 32
#endif
#ifndef STAGE1_RESIDUE_TILE_H
#define STAGE1_RESIDUE_TILE_H 32
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

#ifndef STAGE3_BFS_THREADS
#define STAGE3_BFS_THREADS 256
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


/* -----------------------------------------------------------------------
 * Stage 2: compact residues, Morton-sort them, and grow local charged
 * clusters.  This is a GPU-friendly Goldstein-style approximation:
 *   - residues are stored as one combined array, not fixed +/− pairs;
 *   - each seed owns a small active cluster;
 *   - nearby residues are searched in Morton order with expanding windows;
 *   - each newly absorbed residue is connected to the nearest active residue;
 *   - unresolved non-neutral clusters are connected to the nearest border.
 * ----------------------------------------------------------------------- */

__device__ __forceinline__ unsigned int part1by1(unsigned int x)
{
    x &= 0x0000ffffu;
    x = (x ^ (x << 8)) & 0x00ff00ffu;
    x = (x ^ (x << 4)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2)) & 0x33333333u;
    x = (x ^ (x << 1)) & 0x55555555u;
    return x;
}

__device__ __host__ __forceinline__ unsigned int morton2d_hostdev(unsigned int x, unsigned int y)
{
#ifdef __CUDA_ARCH__
    return (part1by1(y) << 1) | part1by1(x);
#else
    x &= 0x0000ffffu;
    x = (x ^ (x << 8)) & 0x00ff00ffu;
    x = (x ^ (x << 4)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2)) & 0x33333333u;
    x = (x ^ (x << 1)) & 0x55555555u;
    y &= 0x0000ffffu;
    y = (y ^ (y << 8)) & 0x00ff00ffu;
    y = (y ^ (y << 4)) & 0x0f0f0f0fu;
    y = (y ^ (y << 2)) & 0x33333333u;
    y = (y ^ (y << 1)) & 0x55555555u;
    return (y << 1) | x;
#endif
}

__device__ __forceinline__ void atomic_or_byte(unsigned char *flags, int idx, unsigned char mask)
{
    unsigned int *word = (unsigned int *)(flags + (idx & ~3));
    const unsigned int bit = ((unsigned int)mask) << ((idx & 3) * 8);
    atomicOr(word, bit);
}

__device__ __forceinline__ int iabs_dev(int x) { return x < 0 ? -x : x; }

__device__ void d_place_cut(unsigned char *flags, int a, int b, int c, int d,
                            int xsize, int ysize)
{
    /* Same endpoint convention as CPU PlaceCut: residue coordinates are the
       upper-left corner of a 2x2 wrapped loop, so cuts are shifted onto the
       intervening pixels depending on direction. */
    if (c > a && a > 0) a++;
    else if (c < a && c > 0) c++;
    if (d > b && b > 0) b++;
    else if (d < b && d > 0) d++;

    if (a < 0) a = 0; if (a >= xsize) a = xsize - 1;
    if (c < 0) c = 0; if (c >= xsize) c = xsize - 1;
    if (b < 0) b = 0; if (b >= ysize) b = ysize - 1;
    if (d < 0) d = 0; if (d >= ysize) d = ysize - 1;

    if (a == c && b == d) {
        atomic_or_byte(flags, b * xsize + a, kBranchCut);
        return;
    }

    const int m = iabs_dev(c - a);
    const int n = iabs_dev(d - b);
    if (m > n) {
        const int istep = (a < c) ? 1 : -1;
        const float r = (float)(d - b) / (float)(c - a);
        for (int i = a; i != c + istep; i += istep) {
            int j = b + (int)((float)(i - a) * r + 0.5f);
            if ((unsigned)i < (unsigned)xsize && (unsigned)j < (unsigned)ysize)
                atomic_or_byte(flags, j * xsize + i, kBranchCut);
        }
    } else {
        const int jstep = (b < d) ? 1 : -1;
        const float r = (float)(c - a) / (float)(d - b);
        for (int j = b; j != d + jstep; j += jstep) {
            int i = a + (int)((float)(j - b) * r + 0.5f);
            if ((unsigned)i < (unsigned)xsize && (unsigned)j < (unsigned)ysize)
                atomic_or_byte(flags, j * xsize + i, kBranchCut);
        }
    }
}

__device__ __forceinline__ void nearest_border(int x, int y, int xsize, int ysize,
                                                int *bx, int *by)
{
    int dl = x;
    int dr = xsize - 1 - x;
    int dt = y;
    int db = ysize - 1 - y;
    int best = dl;
    *bx = 0; *by = y;
    if (dr < best) { best = dr; *bx = xsize - 1; *by = y; }
    if (dt < best) { best = dt; *bx = x; *by = 0; }
    if (db < best) { *bx = x; *by = ysize - 1; }
}

__global__ void k_pack_residues_morton(const unsigned char *bitflags,
                                       unsigned int *keys,
                                       int *packed,
                                       int *count,
                                       int xsize,
                                       int ysize,
                                       int length)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= length) return;
    const unsigned char f = bitflags[idx];
    const bool pos = (f & kPosRes) != 0;
    const bool neg = (f & kNegRes) != 0;
    if (!pos && !neg) return;

    const int slot = atomicAdd(count, 1);
    const int x = idx % xsize;
    const int y = idx / xsize;
    keys[slot] = morton2d_hostdev((unsigned int)x, (unsigned int)y);
    packed[slot] = idx | (neg ? INT_MIN : 0);
}

__device__ __forceinline__ int unpack_idx(int p) { return p & INT_MAX; }
__device__ __forceinline__ int unpack_charge(int p) { return (p < 0) ? -1 : 1; }

__global__ void k_goldstein_morton_growth(unsigned char *bitflags,
                                          const int *packed_sorted,
                                          int *claimed,
                                          int nres,
                                          int max_cut_len,
                                          int xsize,
                                          int ysize)
{
    const int seed_ord = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed_ord >= nres) return;

    const int seed_pack = packed_sorted[seed_ord];
    const int seed_idx = unpack_idx(seed_pack);
    const int seed_charge = unpack_charge(seed_pack);

    /* A residue can only seed one cluster.  Other clusters may absorb it first. */
    if (atomicCAS(&claimed[seed_ord], 0, seed_ord + 1) != 0)
        return;

    constexpr int MAX_ACTIVE = STAGE2_CLUSTER_MAX_ACTIVE;
    int active_idx[MAX_ACTIVE];
    int active_x[MAX_ACTIVE];
    int active_y[MAX_ACTIVE];
    int active_n = 1;

    active_idx[0] = seed_idx;
    active_x[0] = seed_idx % xsize;
    active_y[0] = seed_idx / xsize;
    int charge = seed_charge;

    const int max_window = nres < STAGE2_CLUSTER_MAX_WINDOW ? nres : STAGE2_CLUSTER_MAX_WINDOW;
    for (int window = 8; charge != 0 && window <= max_window && active_n < MAX_ACTIVE; window <<= 1) {
        int best_ord = -1;
        int best_dist2 = INT_MAX;
        int best_parent = 0;
        int best_idx = -1;
        int best_charge = 0;

        int lo = seed_ord - window;
        int hi = seed_ord + window;
        if (lo < 0) lo = 0;
        if (hi >= nres) hi = nres - 1;

        for (int ord = lo; ord <= hi; ++ord) {
            if (claimed[ord] != 0) continue;
            const int p = packed_sorted[ord];
            const int idx = unpack_idx(p);
            const int cx = idx % xsize;
            const int cy = idx / xsize;

            int parent = 0;
            int local_best = INT_MAX;
            for (int a = 0; a < active_n; ++a) {
                const int dx = cx - active_x[a];
                const int dy = cy - active_y[a];
                const int d2 = dx * dx + dy * dy;
                if (d2 < local_best) {
                    local_best = d2;
                    parent = a;
                }
            }

            if (local_best < best_dist2) {
                best_dist2 = local_best;
                best_ord = ord;
                best_parent = parent;
                best_idx = idx;
                best_charge = unpack_charge(p);
            }
        }

        if (best_ord < 0)
            continue;
        if (max_cut_len > 0 && best_dist2 > max_cut_len * max_cut_len)
            continue;
        if (atomicCAS(&claimed[best_ord], 0, seed_ord + 1) != 0)
            continue;

        const int bx = best_idx % xsize;
        const int by = best_idx / xsize;
        d_place_cut(bitflags, bx, by, active_x[best_parent], active_y[best_parent], xsize, ysize);

        active_idx[active_n] = best_idx;
        active_x[active_n] = bx;
        active_y[active_n] = by;
        active_n++;
        charge += best_charge;
    }

    if (charge != 0) {
        /* CPU Goldstein connects the active cluster to a border when it cannot
           neutralize locally.  Here we use the active residue nearest any image
           border, then draw one CPU-convention cut to that border. */
        int best_a = 0;
        int best_d = INT_MAX;
        for (int a = 0; a < active_n; ++a) {
            const int x = active_x[a], y = active_y[a];
            int d = x;
            int t = xsize - 1 - x; if (t < d) d = t;
            t = y; if (t < d) d = t;
            t = ysize - 1 - y; if (t < d) d = t;
            if (d < best_d) { best_d = d; best_a = a; }
        }
        int bx, by;
        nearest_border(active_x[best_a], active_y[best_a], xsize, ysize, &bx, &by);
        d_place_cut(bitflags, active_x[best_a], active_y[best_a], bx, by, xsize, ysize);
    }
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

    /* Stage 2 scratch buffers declared in unwrap_cuda.h.  The new Morton-growth
       implementation allocates additional local temporaries inside Stage 2, but
       keeping these fields allocated preserves compatibility with older code. */
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
                                                    const UnwrapCudaDeviceBufs *dev, int max_cut_len,
                                                    int num_res, int xsize, int ysize, int length)
{
    if (length < 1 || !h_bitflags || !dev || !dev->d_bitflags || xsize < 2 || ysize < 2)
        return;

    cudaError_t e;
    unsigned int *d_keys = nullptr;
    int *d_packed = nullptr;
    int *d_count = nullptr;
    int *d_claimed = nullptr;
    int h_count = 0;

    const size_t bitflag_bytes = (((size_t)length + 3u) & ~((size_t)3u)) * sizeof(unsigned char);

    if ((e = cudaMemcpy(dev->d_bitflags, h_bitflags, (size_t)length * sizeof(unsigned char),
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
        cuda_fail(e, "Stage2 H2D bitflags"); return;
    }
    /* Zero the padding bytes so byte-level atomicOr on the last 32-bit word is safe. */
    if (bitflag_bytes > (size_t)length)
        cudaMemset(((unsigned char *)dev->d_bitflags) + length, 0, bitflag_bytes - (size_t)length);

    if ((e = cudaMalloc((void **)&d_keys, (size_t)length * sizeof(unsigned int))) != cudaSuccess) goto fail;
    if ((e = cudaMalloc((void **)&d_packed, (size_t)length * sizeof(int))) != cudaSuccess) goto fail;
    if ((e = cudaMalloc((void **)&d_count, sizeof(int))) != cudaSuccess) goto fail;
    if ((e = cudaMemset(d_count, 0, sizeof(int))) != cudaSuccess) goto fail;

    {
        const int threads = STAGE2_PACK_THREADS;
        const int blocks = (length + threads - 1) / threads;
        k_pack_residues_morton<<<blocks, threads>>>(dev->d_bitflags, d_keys, d_packed,
                                                    d_count, xsize, ysize, length);
        if ((e = cudaGetLastError()) != cudaSuccess) goto fail;
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) goto fail;
    }

    if ((e = cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess) goto fail;
    if (h_count <= 0) {
        (void)cudaMemcpy(h_bitflags, dev->d_bitflags, (size_t)length * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost);
        cudaFree(d_keys); cudaFree(d_packed); cudaFree(d_count);
        return;
    }

    /* Sort compact residues by Morton key. */
    try {
        thrust::device_ptr<unsigned int> keys_begin(d_keys);
        thrust::device_ptr<int> vals_begin(d_packed);
        thrust::sort_by_key(keys_begin, keys_begin + h_count, vals_begin);
    } catch (...) {
        fprintf(stderr, "unwrap_cuda: Stage2 thrust::sort_by_key failed\n");
        goto cleanup;
    }

    if ((e = cudaMalloc((void **)&d_claimed, (size_t)h_count * sizeof(int))) != cudaSuccess) goto fail;
    if ((e = cudaMemset(d_claimed, 0, (size_t)h_count * sizeof(int))) != cudaSuccess) goto fail;

    {
        const int threads = STAGE2_GROW_THREADS;
        const int blocks = (h_count + threads - 1) / threads;
        k_goldstein_morton_growth<<<blocks, threads>>>(dev->d_bitflags, d_packed, d_claimed,
                                                       h_count, max_cut_len, xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess) goto fail;
        if ((e = cudaDeviceSynchronize()) != cudaSuccess) goto fail;
    }

    if ((e = cudaMemcpy(h_bitflags, dev->d_bitflags, (size_t)length * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost)) != cudaSuccess) goto fail;

    printf("  [GPU] Stage2 Morton-growth residues: packed=%d expected=%d\n", h_count, num_res);

cleanup:
    cudaFree(d_keys);
    cudaFree(d_packed);
    cudaFree(d_count);
    cudaFree(d_claimed);
    return;

fail:
    cuda_fail(e, "Stage2 Morton-growth");
    goto cleanup;
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

        const int threads = STAGE3_BFS_THREADS;
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
        dim3 block(STAGE3_AVOID_TILE_W, STAGE3_AVOID_TILE_H);
        dim3 grid((xsize + STAGE3_AVOID_TILE_W - 1) / STAGE3_AVOID_TILE_W,
                  (ysize + STAGE3_AVOID_TILE_H - 1) / STAGE3_AVOID_TILE_H);
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
