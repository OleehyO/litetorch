#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

#ifndef NDEBUG
#define cudaCheck(ans)                                                         \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
#else
#define cudaCheck(ans) ans
#endif

namespace litetorch {
namespace cuda {
#define BASE_THREAD_NUM 128

#define TILE 4
#define MAX_SHARED_MEM 49152
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaCheck(cudaMalloc(&ptr, size * ELEM_SIZE));
    this->size = size;
  }
  ~CudaArray() { cudaCheck(cudaFree(ptr)); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t *ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t> &x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE)
    throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t *out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = val;
}

void Fill(CudaArray *out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// utility functions
////////////////////////////////////////////////////////////////////////////////

__device__ inline size_t gid2idx(size_t gid, CudaVec shape, CudaVec strides,
                                 size_t offset) {
  CudaVec idx_vec;
  idx_vec.size = shape.size;
  assert(shape.size == strides.size &&
         "Shape and strides must be the same size");
  int rem = gid;
  for (int i = shape.size - 1; i >= 0; --i) {
    idx_vec.data[i] = rem % shape.data[i];
    rem /= shape.data[i];
  }
  assert(rem == 0 && "Invalid gid");

  size_t res = offset;
  for (int i = 0; i < shape.size; ++i)
    res += idx_vec.data[i] * strides.data[i];
  return res;
}

size_t calSize(std::vector<int32_t> x) {
  size_t res = 1;
  for (int i = 0; i < x.size(); ++i)
    res *= x[i];
  return res;
}

template <typename T>
__global__ void ElementwiseBinaryKernel(const scalar_t *a, const scalar_t *b,
                                        scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = T::forward(a[gid], b[gid]);
}

template <typename T>
__global__ void ScalarBinaryKernel(const scalar_t *a, scalar_t val,
                                   scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = T::forward(a[gid], val);
}

template <typename T>
__global__ void ElementwiseUnaryKernel(const scalar_t *a, scalar_t *out,
                                       size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = T::forward(a[gid]);
}

struct Multiply {
  __device__ static scalar_t forward(scalar_t a, scalar_t b) { return a * b; }
};

struct Divide {
  __device__ static scalar_t forward(scalar_t a, scalar_t b) { return a / b; }
};

struct Maximum {
  __device__ static scalar_t forward(scalar_t a, scalar_t b) {
    return a > b ? a : b;
  }
};

struct Equal {
  __device__ static scalar_t forward(scalar_t a, scalar_t b) { return a == b; }
};

struct GreaterEqual {
  __device__ static scalar_t forward(scalar_t a, scalar_t b) { return a >= b; }
};

struct Power {
  __device__ static scalar_t forward(scalar_t a, scalar_t b) {
    return powf(a, b);
  }
};

struct Tanh {
  __device__ static scalar_t forward(scalar_t a) { return tanhf(a); }
};

struct Log {
  __device__ static scalar_t forward(scalar_t a) { return logf(a); }
};

struct Exp {
  __device__ static scalar_t forward(scalar_t a) { return expf(a); }
};

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from
// strides

__launch_bounds__(256, 6) __global__
    void CompactKernel(const scalar_t *a, scalar_t *out, size_t size,
                       CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a
   * single entry in the non-compact input a, to the corresponding item (at
   * location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past
   * passing to CUDA kernel) strides: vector of strides of out array offset:
   * offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    size_t aidx = gid2idx(gid, shape, strides, offset);
    out[gid] = a[aidx];
  }
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will
   * primarily call the relevant CUDA kernel. 
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being
   * compact)
   */

  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out,
                                   size_t size, CudaVec shape, CudaVec strides,
                                   size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t outidx = gid2idx(gid, shape, strides, offset);
    out[outidx] = a[gid];
  }
}

void EwiseSetitem(const CudaArray &a, CudaArray *out,
                  std::vector<int32_t> shape, std::vector<int32_t> strides,
                  size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being
   * compact)
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, calSize(shape),
                                              VecToCuda(shape),
                                              VecToCuda(strides), offset);
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t *out,
                                    CudaVec shape, CudaVec strides,
                                    size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t outidx = gid2idx(gid, shape, strides, offset);
    out[outidx] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray *out,
                   std::vector<int32_t> shape, std::vector<int32_t> strides,
                   size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will not
   * be the same as out.size, because out is a non-compact subset array);  it
   * _will_ be the same as the product of items in shape, but covenient to just
   * pass it here.
   *   val: scalar value to write to out: non-compact array whose items are to
   * be written.
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(calSize(shape), val, out->ptr,
                                               VecToCuda(shape),
                                               VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b,
                               scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out,
                                size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////
__device__ void __load_tile(scalar_t *tile, const scalar_t *X,
                            const size_t *shape, const size_t *stride,
                            size_t offset) {
  for (size_t tid = threadIdx.x; tid < shape[0] * shape[1]; tid += blockDim.x) {
    size_t i = tid / shape[1];
    size_t j = tid % shape[1];
    tile[tid] = X[offset + i * stride[0] + j * stride[1]];
  }
}

__device__ void __matmul_thread(const scalar_t *a, const scalar_t *b,
                                scalar_t *c, size_t N, size_t P, size_t bOff,
                                size_t cOff) {
  scalar_t aRTile[TILE * TILE];
  scalar_t bRTile[TILE * TILE];
  scalar_t cRTile[TILE * TILE];

  // Init c register tile with 0
  for (size_t i = 0; i < TILE * TILE; ++i)
    cRTile[i] = 0;

  for (size_t ii = 0; ii < N / TILE; ++ii) {
    // load a register tile from shared memory
    for (size_t k = 0; k < TILE * TILE; ++k)
      aRTile[k] = a[ii * TILE * TILE + k];

    // load b register tile from global memory
    for (size_t k = 0; k < TILE * TILE; ++k)
      bRTile[k] = b[bOff + ii * TILE * P + k * (P / TILE)];

    // compute c register tile
    for (size_t i = 0; i < TILE; ++i)
      for (size_t j = 0; j < TILE; ++j)
        for (size_t k = 0; k < TILE; ++k)
          cRTile[i * TILE + j] += aRTile[i * TILE + k] * bRTile[k * TILE + j];
  }

  // Store c register tile to global memory
  for (size_t k = 0; k < TILE * TILE; ++k)
    c[cOff + k * (P / TILE)] = cRTile[k];
}

__global__ void MatmulTiledKernel(const scalar_t *a, const scalar_t *b,
                                  scalar_t *out, uint32_t M, uint32_t N,
                                  uint32_t P) {
  extern __shared__ scalar_t shared[];
  scalar_t *a_tile = shared;

  size_t aShape[2] = {TILE, N};
  size_t aStride[2] = {N, 1};
  size_t bShape[2] = {N / TILE, TILE * TILE};
  size_t bStride[2] = {P * TILE, 1};
  size_t cShape[2] = {TILE, TILE};
  size_t cStride[2] = {TILE, 1};

  for (size_t bid = blockIdx.x; bid < M / TILE; bid += gridDim.x) {
    size_t aOff = bid * TILE * N;
    __load_tile(a_tile, a, aShape, aStride, aOff);
    __syncthreads();

    for (size_t tid = threadIdx.x; tid < P / TILE; tid += blockDim.x) {
      size_t bOff = tid;
      size_t cOff = bid * TILE * P + tid;
      __matmul_thread(a_tile, b, out, N, P, bOff, cOff);
    }
    __syncthreads();
  }
}

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N,
            uint32_t P) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < M) {
    for (int j = 0; j < P; j++) {
      int r = gid * P + j;
      out[r] = 0;
      for (int k = 0; k < N; k++) {
        int s = gid * N + k;
        int t = k * P + j;
        out[r] += a[s] * b[t];
      }
    }
  }
}

void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M,
            uint32_t N, uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix. 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  CudaDims dim = CudaOneDim(M);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

void Matmul_tiled(const CudaArray &a, const CudaArray &b, CudaArray *out,
                  uint32_t M, uint32_t N, uint32_t P) {
  size_t shared_mem = (TILE * N) * sizeof(scalar_t);
  size_t MAXN = MAX_SHARED_MEM / (TILE * sizeof(scalar_t));
  if (N > MAXN) {
    char error_msg[128];
    sprintf(error_msg,
            "Too large N in (M, N)x(N, P) matmul: N should be less than %zu",
            MAXN);
    throw std::runtime_error(error_msg);
  }
  size_t nblock = (M + TILE - 1) / TILE;
  MatmulTiledKernel<<<nblock, BASE_THREAD_NUM, shared_mem>>>(a.ptr, b.ptr,
                                                             out->ptr, M, N, P);
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t size,
                                size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t max_val = a[gid * reduce_size];
    for (int i = 0; i < reduce_size; ++i)
      max_val = max(max_val, a[gid * reduce_size + i]);
    out[gid] = max_val;
  }
}

__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t size,
                                size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t sum_val = 0;
    for (int i = 0; i < reduce_size; ++i)
      sum_val += a[gid * reduce_size + i];
    out[gid] = sum_val;
  }
}

void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks. 
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size,
                                           reduce_size);
}

void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size,
                                           reduce_size);
}

} // namespace cuda
} // namespace litetorch

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace litetorch::cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(),
                   numpy_strides.begin(),
                   [](size_t &c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t *host_ptr = (scalar_t *)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0)
      throw std::bad_alloc();
    cudaError_t err =
        cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void *p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset,
                                 deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out) {
    cudaError_t err = cudaMemcpy(out->ptr, a.request().ptr,
                                 out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
  m.def("ewise_mul",
        [](const CudaArray &a, const CudaArray &b, CudaArray *out) {
          CudaDims dim = CudaOneDim(out->size);
          ElementwiseBinaryKernel<Multiply>
              <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        });
  m.def("scalar_mul", [](const CudaArray &a, scalar_t val, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarBinaryKernel<Multiply>
        <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
  });
  m.def("ewise_div",
        [](const CudaArray &a, const CudaArray &b, CudaArray *out) {
          CudaDims dim = CudaOneDim(out->size);
          ElementwiseBinaryKernel<Divide>
              <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        });
  m.def("scalar_div", [](const CudaArray &a, scalar_t val, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarBinaryKernel<Divide>
        <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
  });
  m.def("scalar_power", [](const CudaArray &a, scalar_t val, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarBinaryKernel<Power>
        <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
  });
  m.def("ewise_maximum",
        [](const CudaArray &a, const CudaArray &b, CudaArray *out) {
          CudaDims dim = CudaOneDim(out->size);
          ElementwiseBinaryKernel<Maximum>
              <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
        });
  m.def("scalar_maximum", [](const CudaArray &a, scalar_t val, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarBinaryKernel<Maximum>
        <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
  });
  m.def("ewise_eq", [](const CudaArray &a, const CudaArray &b, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ElementwiseBinaryKernel<Equal>
        <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
  });
  m.def("scalar_eq", [](const CudaArray &a, scalar_t val, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarBinaryKernel<Equal>
        <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
  });
  m.def("ewise_ge", [](const CudaArray &a, const CudaArray &b, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ElementwiseBinaryKernel<GreaterEqual>
        <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
  });
  m.def("scalar_ge", [](const CudaArray &a, scalar_t val, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ScalarBinaryKernel<GreaterEqual>
        <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
  });
  m.def("ewise_log", [](const CudaArray &a, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ElementwiseUnaryKernel<Log>
        <<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
  });
  m.def("ewise_exp", [](const CudaArray &a, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ElementwiseUnaryKernel<Exp>
        <<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
  });
  m.def("ewise_tanh", [](const CudaArray &a, CudaArray *out) {
    CudaDims dim = CudaOneDim(out->size);
    ElementwiseUnaryKernel<Tanh>
        <<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
  });

  m.def("matmul", Matmul);
  m.def("matmul_tiled", litetorch::cuda::Matmul_tiled);
  m.def("reduce_max", litetorch::cuda::ReduceMax);
  m.def("reduce_sum", litetorch::cuda::ReduceSum);
}