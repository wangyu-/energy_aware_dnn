#include "cudnn.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cuda.h"

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
	assert(0==1);														\
    /*exit(1);*/                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
    std::stringstream _error;                                          \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    if (status != 0) {                                                 \
    std::stringstream _error;                                          \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

