#ifndef UNWRAP_CUDA_H
#define UNWRAP_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

/** Device pointers for unwrap stages; allocated by unwrap_cuda_device_bufs_alloc. */
typedef struct UnwrapCudaDeviceBufs {
    float          *d_phase;
    unsigned char  *d_bitflags;
    float          *d_soln;
    int            *d_residue_count;
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


void unwrap_cuda_launch_unwrapping(float                 *h_phase,
                                   unsigned char         *h_bitflags,
                                   float                 *h_soln,
                                   const UnwrapCudaDeviceBufs *dev,
                                   int                    xsize,
                                   int                    ysize,
                                   int                    length);

#ifdef __cplusplus
}
#endif

#endif /* UNWRAP_CUDA_H */
