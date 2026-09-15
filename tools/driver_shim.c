/*
 * LD_PRELOAD shim for CUDA 12.3+ driver introspection APIs missing in driver 535 (CUDA 12.2).
 * libtensorrt_llm.so 1.0.0 references these as undefined symbols; driver 535 lacks them.
 * All three are cold-path profiling/introspection APIs — stubs return
 * CUDA_ERROR_NOT_SUPPORTED (801) and never touch the driver.
 */
#include <stddef.h>

typedef int CUresult;
#define CUDA_ERROR_NOT_SUPPORTED 801
typedef void *CUkernel;
typedef void *CUlibrary;

CUresult cuKernelGetName(const char **name, CUkernel hKernel)
{
    (void)hKernel;
    if (name)
        *name = NULL;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuLibraryEnumerateKernels(CUkernel *kernels, size_t numKernels, CUlibrary lib)
{
    (void)kernels;
    (void)numKernels;
    (void)lib;
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuLibraryGetKernelCount(size_t *count, CUlibrary lib)
{
    (void)lib;
    if (count)
        *count = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
}
