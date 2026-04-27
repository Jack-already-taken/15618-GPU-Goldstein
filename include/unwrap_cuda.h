#ifndef UNWRAP_CUDA_H
#define UNWRAP_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/** Device pointers for unwrap stages; allocated by unwrap_cuda_device_bufs_alloc. */
typedef struct UnwrapCudaDeviceBufs {
    // Stage 1: Residue Identification
    float          *d_phase;
    unsigned char  *d_bitflags;

    // Stage 2: Residue Matching
    int            *d_pos_residues;
    int            *d_neg_residues;
    int            *d_pairs;
    int            *d_residue_count;

    // Stage 2 fixed-bin scratch, allocated once in unwrap_cuda_device_bufs_alloc.
    int            *d_bin_counts;
    int            *d_bin_items;
    int            *d_bin_overflow;
    int            *d_stage2_verify_stats;
    int             stage2_nbins;
    int             stage2_bin_items;

    // Stage 3: tile-local unwrap and tile-graph stitching.
    float          *d_soln;
    float          *d_gradx;
    float          *d_grady;
    int            *d_tile_has_valid;
    int            *d_edge_valid;
    int            *d_edge_delta_k;
    int            *d_tile_known;
    int            *d_tile_offset_k;

    // Stage 3 host/CPU scratch for the tile-graph solve, allocated once.
    int            *h_tile_has_valid;
    int            *h_edge_valid;
    int            *h_edge_delta_k;
    int            *h_tile_known;
    int            *h_tile_offset_k;
    int            *h_queue;

    int             stage3_tile_capacity;
    int             stage3_edge_capacity;
} UnwrapCudaDeviceBufs;

/**
 * Probe CUDA runtime (device count, cudaSetDevice(0), synchronize).
 * Returns 0 on success, non-zero cudaError_t cast to int on failure.
 */
int unwrap_cuda_init(void);

/**
 * Allocates all device buffers for the CUDA unwrap path.
 * Returns 0 on success, cudaError_t cast to int on failure (partial buffers freed).
 */
int unwrap_cuda_device_bufs_alloc(int length, UnwrapCudaDeviceBufs *out);


void unwrap_cuda_device_bufs_free(UnwrapCudaDeviceBufs *buf);


int unwrap_cuda_launch_residue_identification(float                 *h_phase,
                                              unsigned char         *h_bitflags,
                                              const UnwrapCudaDeviceBufs *dev,
                                              int                    xsize,
                                              int                    ysize,
                                              int                    length);


void unwrap_cuda_launch_residue_matching(unsigned char               *h_bitflags,
                                         const UnwrapCudaDeviceBufs *dev,
                                         int                        max_cut_len,
                                         int                        num_res,
                                         int                        xsize,
                                         int                        ysize,
                                         int                        length);


void unwrap_cuda_launch_unwrapping(float                      *h_phase,
                                   unsigned char              *h_bitflags,
                                   float                      *h_soln,
                                   float                      *h_gradx,
                                   float                      *h_grady,
                                   const UnwrapCudaDeviceBufs *dev,
                                   int                         xsize,
                                   int                         ysize,
                                   int                         length);


#ifdef __cplusplus
}
#endif

#endif /* UNWRAP_CUDA_H */
