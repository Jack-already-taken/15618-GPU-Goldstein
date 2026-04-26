/*
 * unwrap_cuda_tiled.cu
 *
 * DROP-IN additions for unwrap_cuda.cu
 *x
 * Add the following to unwrap_cuda.h:
 *   void unwrap_cuda_launch_unwrapping_tiled(
 *       float *h_phase, unsigned char *h_bitflags, float *h_soln,
 *       float *h_gradx, float *h_grady,
 *       const UnwrapCudaDeviceBufs *dev,
 *       int xsize, int ysize, int length);
 *
 * The existing d_frontier_a is reused as the "global changed flag" scratch
 * (just one int) -- or allocate d_changed separately (shown below).
 *
 * APPROACH
 * --------
 * Phase 1 — intra-tile relaxation
 *   Each block owns one TILE_SZ x TILE_SZ region.
 *   Every thread loads phase[k] as its initial soln (if not AVOID).
 *   Then for TILE_SZ*2 iterations, each non-AVOID thread tries to pull a
 *   better-anchored value from its 4-connected shared-memory neighbors.
 *   "Better anchored" = neighbor is not AVOID and has a lower generation
 *   counter (was seeded earlier in this tile's local BFS).
 *   After convergence, write back soln and set tile_changed[blockIdx] if
 *   any boundary pixel changed value (i.e. needs to propagate into neighbor
 *   tiles next round).
 *
 * Phase 2 — inter-tile iteration
 *   The host re-launches phase 1 until no tile reports a change.
 *   For typical phase maps this converges in O(image_size / TILE_SZ) rounds.
 *
 * WHY THIS IS CORRECT
 * -------------------
 * Gradients gradx/grady already encode Itoh's method.  Within a tile, each
 * relaxation iteration propagates soln[nb] = soln[curr] +/- grad exactly as
 * UnwrapAroundCutsFrontier does.  Across tile boundaries, re-launching the
 * kernel lets corrected values flow from one tile into adjacent tiles.
 * Branch cuts (kBranchCut | kBorder) are respected by the AVOID check.
 *
 * BENCHMARKING NOTE
 * -----------------
 * unwrap_cuda_launch_unwrapping_tiled returns elapsed_ms for the unwrap
 * stage only (excluding H2D/D2H transfers that are shared setup).
 * Compare this directly against the BFS version's unwrap_ms.
 * End-to-end time = residue_ms + branchcut_ms (CPU) + unwrap_ms.
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include "unwrap_cuda.h"
#include "pi.h"

/* -----------------------------------------------------------------------
 * Internal helpers (duplicate of device_gradient in unwrap_cuda.cu --
 * if merging into one file, keep only one copy)
 * -------------------------------------------------------------------- */

#define TFLOOD_TILE  32          /* tile side length; 32 => 32x32 = 1024 threads */
#define TFLOOD_ITERS (TFLOOD_TILE * TFLOOD_TILE)  /* local relaxation iterations per launch */

namespace tflood {

constexpr unsigned char kAvoid     = 0x10 | 0x20;   /* BRANCH_CUT | BORDER */
constexpr unsigned char kUnwrapped = 0x40;

__device__ __forceinline__ float dgrad(float a, float b)
{
    float r = a - b;
    if      (r >  (float)PI)   r -= (float)TWOPI;
    else if (r < -(float)PI)   r += (float)TWOPI;
    return r;
}

/* -----------------------------------------------------------------------
 * k_tiled_relax
 *
 * Each block processes one TILE_SZ x TILE_SZ patch.
 * Shared memory layout:
 *   s_soln [TILE+2][TILE+2]  — 1-pixel halo so boundary threads can
 *                               read neighbor tiles' current global soln
 *   s_flags[TILE+2][TILE+2]
 *   s_gen  [TILE+2][TILE+2]  — generation: 0 = uninitialised seed,
 *                               >0 = distance from nearest local seed
 *   s_changed               — set to 1 if any boundary pixel was updated
 *
 * Halo pixels are loaded from global memory (read-only during relaxation).
 * After relaxation, interior pixels whose values changed are written back
 * and any change to a boundary pixel sets s_changed.
 * -------------------------------------------------------------------- */

#define TILE  TFLOOD_TILE
#define TILES (TILE + 2)   /* shared-memory side including halo */

__global__ void k_tiled_relax(
    const float   *phase,
    unsigned char *bitflags,
    float         *soln,
    const float   *gradx,
    const float   *grady,
    int  *gen,             /* persistent gen array */
    int           *d_any_changed,   /* single global int; atomicOr'd */
    int            xsize,
    int            ysize)
{
    /* shared memory */
    __shared__ float         s_soln [TILES][TILES];
    __shared__ unsigned char s_flags[TILES][TILES];
    // __shared__ unsigned char s_gen  [TILES][TILES];  /* generation counter */
    __shared__ int s_gen [TILES][TILES];
    __shared__ int           s_changed;

    const int tx = threadIdx.x;   /* 0 .. TILE-1 */
    const int ty = threadIdx.y;
    /* global pixel coords of this thread's interior cell */
    const int gx = blockIdx.x * TILE + tx;
    const int gy = blockIdx.y * TILE + ty;
    /* shared-memory coords (offset by 1 for halo) */
    const int sx = tx + 1;
    const int sy = ty + 1;

    if (tx == 0 && ty == 0) s_changed = 0;

    /* ---- Load interior cell ---- */
    if (gx < xsize && gy < ysize) {
        const int k = gy * xsize + gx;
        s_soln [sy][sx] = soln[k];
        s_flags[sy][sx] = bitflags[k];
        // s_gen  [sy][sx] = (bitflags[k] & kUnwrapped) ? 1 : 0;
        s_gen  [sy][sx] = gen[k];
    } else {
        /* out-of-image: treat as AVOID so nothing propagates here */
        s_soln [sy][sx] = 0.f;
        s_flags[sy][sx] = kAvoid;
        s_gen  [sy][sx] = 0;
    }

    /* ---- Load halo (left/right columns, top/bottom rows) ----
     * Only the edge threads load their respective halo cells.
     * Halo values are read-only during relaxation.                */

    /* left halo: tx==0 loads column sx-1 = 0 */
    if (tx == 0) {
        const int hx = gx - 1, hy = gy;
        if (hx >= 0 && hy >= 0 && hy < ysize) {
            const int k = hy * xsize + hx;
            s_soln [sy][0] = soln[k];
            s_flags[sy][0] = bitflags[k];
            // s_gen  [sy][0] = (bitflags[k] & kUnwrapped) ? 1 : 0;
            s_gen  [sy][0] = gen[k];
        } else {
            s_soln [sy][0] = 0.f; s_flags[sy][0] = kAvoid; s_gen[sy][0] = 0;
        }
    }
    /* right halo */
    if (tx == TILE - 1) {
        const int hx = gx + 1, hy = gy;
        if (hx < xsize && hy >= 0 && hy < ysize) {
            const int k = hy * xsize + hx;
            s_soln [sy][TILE+1] = soln[k];
            s_flags[sy][TILE+1] = bitflags[k];
            // s_gen  [sy][TILE+1] = (bitflags[k] & kUnwrapped) ? 1 : 0;
            s_gen  [sy][TILE+1] = gen[k];
        } else {
            s_soln [sy][TILE+1] = 0.f; s_flags[sy][TILE+1] = kAvoid; s_gen[sy][TILE+1] = 0;
        }
    }
    /* top halo */
    if (ty == 0) {
        const int hx = gx, hy = gy - 1;
        if (hx >= 0 && hx < xsize && hy >= 0) {
            const int k = hy * xsize + hx;
            s_soln [0][sx] = soln[k];
            s_flags[0][sx] = bitflags[k];
            // s_gen  [0][sx] = (bitflags[k] & kUnwrapped) ? 1 : 0;
            s_gen  [0][sx] = gen[k];
        } else {
            s_soln [0][sx] = 0.f; s_flags[0][sx] = kAvoid; s_gen[0][sx] = 0;
        }
    }
    /* bottom halo */
    if (ty == TILE - 1) {
        const int hx = gx, hy = gy + 1;
        if (hx >= 0 && hx < xsize && hy < ysize) {
            const int k = hy * xsize + hx;
            s_soln [TILE+1][sx] = soln[k];
            s_flags[TILE+1][sx] = bitflags[k];
            // s_gen  [TILE+1][sx] = (bitflags[k] & kUnwrapped) ? 1 : 0;
            s_gen  [TILE+1][sx] = gen[k];
        } else {
            s_soln [TILE+1][sx] = 0.f; s_flags[TILE+1][sx] = kAvoid; s_gen[TILE+1][sx] = 0;
        }
    }
    /* corners (needed for completeness but corner halos are never used
       in 4-connected propagation, so we can leave them uninitialised) */

    __syncthreads();

    /* ---- Relaxation: each pixel takes value from its lowest-gen neighbor ----
     * Pure consistency-based: no generation tracking needed.
     * Seed pixel has soln set; others start at 0.
     * Each iter: if I have no value yet (soln==0 and not seed), pull from
     * any neighbor that has a value. Use s_flags bit kUnwrapped to track
     * whether a pixel has been reached.                                    */
    const bool valid = (gx < xsize) && (gy < ysize)
                    && !(s_flags[sy][sx] & kAvoid);
    const int gk = gy * xsize + gx;

    for (int iter = 0; iter < TFLOOD_ITERS; iter++) {

        if (valid && !(s_flags[sy][sx] & kUnwrapped)) {

            float new_val = 0.f;
            bool found = false;

            /* left */
            if (!found && !(s_flags[sy][sx-1] & kAvoid) && (s_flags[sy][sx-1] & kUnwrapped)) {
                const int nk = gy * xsize + (gx - 1);
                new_val = s_soln[sy][sx-1] - gradx[nk];
                found = true;
            }
            /* right */
            if (!found && !(s_flags[sy][sx+1] & kAvoid) && (s_flags[sy][sx+1] & kUnwrapped)) {
                new_val = s_soln[sy][sx+1] + gradx[gk];
                found = true;
            }
            /* top */
            if (!found && !(s_flags[sy-1][sx] & kAvoid) && (s_flags[sy-1][sx] & kUnwrapped)) {
                const int nk = (gy - 1) * xsize + gx;
                new_val = s_soln[sy-1][sx] - grady[nk];
                found = true;
            }
            /* bottom */
            if (!found && !(s_flags[sy+1][sx] & kAvoid) && (s_flags[sy+1][sx] & kUnwrapped)) {
                new_val = s_soln[sy+1][sx] + grady[gk];
                found = true;
            }

            if (found) {
                s_soln[sy][sx] = new_val;
                s_flags[sy][sx] |= kUnwrapped;
            }
        }

        __syncthreads();
    }

    /* ---- Write back ---- */
    if (gx < xsize && gy < ysize) {
        float old_val = soln[gk];
        float new_val = s_soln[sy][sx];

        if (s_flags[sy][sx] & kUnwrapped) {
            soln[gk]     = new_val;
            bitflags[gk] |= kUnwrapped;

            bool is_boundary = (tx == 0 || tx == TILE-1 ||
                                ty == 0 || ty == TILE-1);
            if (is_boundary && fabsf(new_val - old_val) > 1e-6f)
                s_changed = 1;
        }
    }

    __syncthreads();

    if (tx == 0 && ty == 0 && s_changed)
        atomicOr(d_any_changed, 1);
}

/* -----------------------------------------------------------------------
 * k_avoid_fill_tiled — same as the BFS version, runs once after convergence
 * -------------------------------------------------------------------- */
__global__ void k_avoid_fill_t(
    const float   *phase,
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
        float r = phase[k] - phase[k-1];
        if      (r >  (float)PI)  r -= (float)TWOPI;
        else if (r < -(float)PI)  r += (float)TWOPI;
        soln[k] = soln[k-1] + r;
    } else if (!(bitflags[k - xsize] & kAvoid)) {
        float r = phase[k] - phase[k-xsize];
        if      (r >  (float)PI)  r -= (float)TWOPI;
        else if (r < -(float)PI)  r += (float)TWOPI;
        soln[k] = soln[k-xsize] + r;
    }
}

} /* namespace tflood */


/* -----------------------------------------------------------------------
 * Public entry point
 *
 * Mirrors unwrap_cuda_launch_unwrapping but uses tiled relaxation.
 * Returns elapsed_ms for the GPU unwrap stage only (kernel time).
 * H2D transfers are included in elapsed_ms because they are part of the
 * stage cost (same convention as the BFS version).
 * -------------------------------------------------------------------- */
extern "C" float unwrap_cuda_launch_unwrapping_tiled(
    float         *h_phase,
    unsigned char *h_bitflags,
    float         *h_soln,
    float         *h_gradx,
    float         *h_grady,
    const UnwrapCudaDeviceBufs *dev,
    int            xsize,
    int            ysize,
    int            length)
{
    if (length < 1 || !h_phase || !h_bitflags || !h_soln || !h_gradx || !h_grady
        || !dev || !dev->d_phase || !dev->d_bitflags || !dev->d_soln
        || !dev->d_gradx || !dev->d_grady || !dev->d_frontier_a)
        return -1.f;

    cudaError_t e;

    /* d_frontier_a[0] reused as the "any_changed" flag (just 1 int) */
    int *d_any_changed = dev->d_frontier_a;

    //  /* Allocate persistent gen array on device */
    // unsigned char *d_gen = NULL;
    // cudaMalloc((void**)&d_gen, (size_t)length * sizeof(unsigned char));
    // cudaMemset(d_gen, 0, (size_t)length * sizeof(unsigned char));

    int *d_gen = NULL;
    cudaMalloc((void**)&d_gen, (size_t)length * sizeof(int));
    cudaMemset(d_gen, 0, (size_t)length * sizeof(int));

    /* ---- Seed: every non-AVOID pixel gets its own wrapped phase ---- */
    /* Clear unwrapped flags and soln first */
    constexpr unsigned char kAvoidU  = 0x10 | 0x20;
    constexpr unsigned char kUnwrappedS = 0x40;

    for (int k = 0; k < length; k++) {
        h_bitflags[k] &= ~kUnwrappedS;
        h_soln[k] = 0.f;
    }

    // /* Seed one pixel per tile — generation counter needs sparse seeds
    // so it has a gradient to propagate from */
    // for (int ty = 0; ty < ysize; ty += TFLOOD_TILE) {
    //     for (int tx = 0; tx < xsize; tx += TFLOOD_TILE) {
    //         for (int dy = 0; dy < TFLOOD_TILE && ty+dy < ysize; dy++) {
    //             for (int dx = 0; dx < TFLOOD_TILE && tx+dx < xsize; dx++) {
    //                 int k = (ty+dy)*xsize + (tx+dx);
    //                 if (!(h_bitflags[k] & kAvoidU)) {
    //                     h_soln[k] = h_phase[k];
    //                     h_bitflags[k] |= kUnwrappedS;
    //                     goto next_tile;
    //                 }
    //             }
    //         }
    //         next_tile:;
    //     }
    // }

    /* Seed only ONE pixel globally — first non-AVOID pixel.
       All other pixels propagate from this single seed over
       multiple inter-tile rounds. This gives consistent absolute
       values across all tiles. */
    for (int k = 0; k < length; k++) {
        if (!(h_bitflags[k] & kAvoidU)) {
            h_soln[k] = h_phase[k];
            h_bitflags[k] |= kUnwrappedS;
            break;
        }
    }

    /* Build and upload gen array — seeds get gen=1, rest stay 0 */
    // unsigned char *h_gen = (unsigned char*)calloc(length, sizeof(unsigned char));
    // for (int k = 0; k < length; k++) {
    //     if (h_bitflags[k] & kUnwrappedS)
    //         h_gen[k] = 1;
    // }
    // cudaMemcpy(d_gen, h_gen, (size_t)length * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // free(h_gen);

    int *h_gen = (int*)calloc(length, sizeof(int));
    for (int k = 0; k < length; k++) {
        if (h_bitflags[k] & kUnwrappedS)
            h_gen[k] = 1;
    }
    cudaMemcpy(d_gen, h_gen, (size_t)length * sizeof(int), cudaMemcpyHostToDevice);
    free(h_gen);

    /* ---- H2D ---- */
    cudaMemcpy(dev->d_phase,    h_phase,    (size_t)length * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_bitflags, h_bitflags, (size_t)length * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_soln,     h_soln,     (size_t)length * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_gradx,    h_gradx,    (size_t)length * sizeof(float),         cudaMemcpyHostToDevice);
    cudaMemcpy(dev->d_grady,    h_grady,    (size_t)length * sizeof(float),         cudaMemcpyHostToDevice);

    /* ---- Grid / block dims ---- */
    const dim3 block(TFLOOD_TILE, TFLOOD_TILE);
    const dim3 grid((xsize + TFLOOD_TILE - 1) / TFLOOD_TILE,
                    (ysize + TFLOOD_TILE - 1) / TFLOOD_TILE);

    /* ---- Time the kernel loop ---- */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    int round = 0;
    int h_any_changed = 1;

    while (h_any_changed) {
        /* reset flag */
        cudaMemset(d_any_changed, 0, sizeof(int));

        // tflood::k_tiled_relax<<<grid, block>>>(
        //     dev->d_phase, dev->d_bitflags, dev->d_soln,
        //     dev->d_gradx, dev->d_grady,
        //     d_any_changed,
        //     xsize, ysize);
        

        tflood::k_tiled_relax<<<grid, block>>>(
            dev->d_phase, dev->d_bitflags, dev->d_soln,
            dev->d_gradx, dev->d_grady,
            d_gen,
            d_any_changed,
            xsize, ysize);

        if ((e = cudaGetLastError()) != cudaSuccess) {
            fprintf(stderr, "k_tiled_relax: %s\n", cudaGetErrorString(e));
            break;
        }
        cudaDeviceSynchronize();

        cudaMemcpy(&h_any_changed, d_any_changed, sizeof(int), cudaMemcpyDeviceToHost);
        ++round;
        printf("  [GPU tiled] round %d, changed=%d\n", round, h_any_changed);
        fflush(stdout);

        // /* Safety cap: should converge in O(image/TILE) rounds */
        if (round > (xsize + ysize) / TFLOOD_TILE * 4) {
            printf("  [GPU tiled] WARNING: hit round cap at %d\n", round);
            break;
        }

        // if (round > 500) {   // lower the cap drastically for debugging
        // printf("  [GPU tiled] INFINITE LOOP DETECTED, breaking\n");
        // break;
    }
    printf("  [GPU tiled] flood-fill: %d rounds\n", round);

    /* ---- AVOID-band fill ---- */
    {
        dim3 b2(16, 16);
        dim3 g2((xsize + 15) / 16, (ysize + 15) / 16);
        tflood::k_avoid_fill_t<<<g2, b2>>>(
            dev->d_phase, dev->d_bitflags, dev->d_soln, xsize, ysize);
        if ((e = cudaGetLastError()) != cudaSuccess)
            fprintf(stderr, "k_avoid_fill_t: %s\n", cudaGetErrorString(e));
        cudaDeviceSynchronize();
    }

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float kernel_ms = 0.f;
    cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
    printf("  [GPU tiled] unwrap kernel total: %.4f ms\n", kernel_ms);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    /* ---- D2H ---- */
    cudaMemcpy(h_soln,     dev->d_soln,     (size_t)length * sizeof(float),         cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bitflags, dev->d_bitflags, (size_t)length * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    /* Debug: count how many pixels are still gen=0 after convergence */
    int *h_gen_out = (int*)calloc(length, sizeof(int));
    cudaMemcpy(h_gen_out, d_gen, (size_t)length * sizeof(int), cudaMemcpyDeviceToHost);
    int zero_gen = 0, max_gen = 0;
    for (int k = 0; k < length; k++) {
        if (h_gen_out[k] == 0) zero_gen++;
        if (h_gen_out[k] > max_gen) max_gen = h_gen_out[k];
    }
    printf("  [GPU tiled] gen stats: zero_gen=%d max_gen=%d\n", zero_gen, max_gen);
    free(h_gen_out);

    cudaFree(d_gen);
    return kernel_ms;
}
