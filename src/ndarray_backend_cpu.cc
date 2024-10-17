// Copyright (c) 2024, OleehyO
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

#define __ASSERT__(condition, message)                                         \
  do {                                                                         \
    if (!(condition)) {                                                        \
      throw std::runtime_error(message);                                       \
    }                                                                          \
  } while (false)

namespace litetorch {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT
 * boundaries in memory.  This alignment should be at least TILE * ELEM_SIZE,
 * though we make it even larger here by default.
 */
struct AlignedArray {
  explicit AlignedArray(const size_t size) {
    int ret = posix_memalign(reinterpret_cast<void **>(&ptr), ALIGNMENT,
                             size * ELEM_SIZE);
    if (ret != 0)
      throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {
    return static_cast<size_t>(reinterpret_cast<uintptr_t>(ptr));
  }
  scalar_t *ptr;
  size_t size;
};

class Indexer {
public: // NOLINT
  Indexer(std::vector<int> shape, std::vector<int> strides, int offset)
      : shape{shape}, strides{strides}, offset{offset} {
    cur_idx.resize(shape.size(), 0);
  }

  int getIdx() {
    int res = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
      res += cur_idx[i] * strides[i];
    }
    if (!_next()) {
      _endflag = true;
    }
    return res;
  }

  bool isEnd() { return _endflag; }

private: // NOLINT
  std::vector<int> shape;
  std::vector<int> strides;
  std::vector<int> cur_idx;
  int offset;
  bool _endflag = false;

  bool _next() {
    int carry = 1;
    for (int i = shape.size() - 1; carry > 0 && i >= 0; --i) {
      cur_idx[i] += carry;
      assert(cur_idx[i] <= shape[i] && "Index out of bounds");
      if (cur_idx[i] == shape[i]) {
        cur_idx[i] = 0;
        carry = 1;
      } else {
        carry = 0;
      }
    }
    return carry == 0;
  }
};

void Fill(AlignedArray *out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

void Compact(const AlignedArray &a, AlignedArray *out,
             std::vector<int32_t> shape, std::vector<int32_t> strides,
             size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being
   * compact)
   */
  Indexer indexer{shape, strides, offset};
  int cnt = 0;
  while (!indexer.isEnd()) {
    int idx = indexer.getIdx();
    out->ptr[cnt++] = a.ptr[idx];
  }

#ifndef NDEBUG
  int check =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  __ASSERT__(cnt == check, "Compact array size does not match");
#endif
}

void EwiseSetitem(const AlignedArray &a, AlignedArray *out,
                  std::vector<int32_t> shape, std::vector<int32_t> strides,
                  size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being
   * compact)
   */
  Indexer indexer{shape, strides, offset};
  int cnt = 0;
  while (!indexer.isEnd()) {
    int idx = indexer.getIdx();
    out->ptr[idx] = a.ptr[cnt++];
  }
#ifndef NDEBUG
  int check =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  __ASSERT__(cnt == check, "EwiseSetitem array size does not match");
#endif
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out,
                   std::vector<int32_t> shape, std::vector<int32_t> strides,
                   size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note
   * be the same as out.size, because out is a non-compact subset array);  it
   * _will_ be the same as the product of items in shape, but convenient to just
   * pass it here.
   *   val: scalar value to write to out: non-compact array whose
   * items are to be written shape: shapes of each dimension of out strides:
   * strides of the out array offset: offset of the out array
   */

  Indexer indexer{shape, strides, offset};
  int cnt = 0;
  while (!indexer.isEnd()) {
    int idx = indexer.getIdx();
    out->ptr[idx] = val;
#ifndef NDEBUG
    cnt++;
#endif
  }

#ifndef NDEBUG
  int check =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  __ASSERT__(cnt == check && cnt == size,
             "ScalarSetitem array size does not match");
#endif
}

void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the
   * scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

template <typename Array, typename OpType>
void EwiseBinaryOp(const Array &a, const Array &b, Array *out) {
  OpType op;
  for (int i = 0; i < a.size; ++i) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}

template <typename Array, typename OpType>
void ScalarBinaryOp(const Array &a, const scalar_t &val, Array *out) {
  OpType op;
  for (int i = 0; i < a.size; ++i) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}

template <typename Array, typename OpType>
void EwiseUnaryOp(const Array &a, Array *out) {
  OpType op;
  for (int i = 0; i < a.size; ++i) {
    out->ptr[i] = op(a.ptr[i]);
  }
}

struct powerOp {
  scalar_t operator()(scalar_t a, scalar_t b) const { return std::pow(a, b); }
};

struct maximumOp {
  scalar_t operator()(scalar_t a, scalar_t b) const { return std::max(a, b); }
};

struct logOp {
  scalar_t operator()(scalar_t a) const { return std::log(a); }
};

struct expOp {
  scalar_t operator()(scalar_t a) const { return std::exp(a); }
};

struct tanhOp {
  scalar_t operator()(scalar_t a) const { return std::tanh(a); }
};


void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out,
            uint32_t m, uint32_t n, uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  Fill(out, 0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      for (int k = 0; k < n; ++k) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
}


void ReduceMax(const AlignedArray &a, AlignedArray *out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  __ASSERT__(a.size % reduce_size == 0, "ReduceMax array size does not match");
  __ASSERT__(a.size / reduce_size == out->size,
             "ReduceMax array size does not match");
  for (int i = 0; i < out->size; ++i) {
    scalar_t max_val = a.ptr[i * reduce_size];
    for (int j = 0; j < reduce_size; ++j) {
      max_val = std::max(max_val, a.ptr[i * reduce_size + j]);
    }
    out->ptr[i] = max_val;
  }
}

void ReduceSum(const AlignedArray &a, AlignedArray *out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  __ASSERT__(a.size % reduce_size == 0, "ReduceSum array size does not match");
  __ASSERT__(a.size / reduce_size == out->size,
             "ReduceSum array size does not match");
  for (int i = 0; i < out->size; ++i) {
    scalar_t sum_val = 0;
    for (int j = 0; j < reduce_size; ++j) {
      sum_val += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum_val;
  }
}

} // namespace cpu
} // namespace litetorch

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace litetorch; // NOLINT
  using namespace cpu;    // NOLINT

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(),
                   numpy_strides.begin(),
                   [](size_t &c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray *out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseBinaryOp<AlignedArray, std::multiplies<scalar_t>>);
  m.def("scalar_mul", ScalarBinaryOp<AlignedArray, std::multiplies<scalar_t>>);
  m.def("ewise_div", EwiseBinaryOp<AlignedArray, std::divides<scalar_t>>);
  m.def("scalar_div", ScalarBinaryOp<AlignedArray, std::divides<scalar_t>>);
  m.def("scalar_power", ScalarBinaryOp<AlignedArray, powerOp>);

  m.def("ewise_maximum", EwiseBinaryOp<AlignedArray, maximumOp>);
  m.def("scalar_maximum", ScalarBinaryOp<AlignedArray, maximumOp>);
  m.def("ewise_eq", EwiseBinaryOp<AlignedArray, std::equal_to<scalar_t>>);
  m.def("scalar_eq", ScalarBinaryOp<AlignedArray, std::equal_to<scalar_t>>);
  m.def("ewise_ge", EwiseBinaryOp<AlignedArray, std::greater_equal<scalar_t>>);
  m.def("scalar_ge",
        ScalarBinaryOp<AlignedArray, std::greater_equal<scalar_t>>);

  m.def("ewise_log", EwiseUnaryOp<AlignedArray, logOp>);
  m.def("ewise_exp", EwiseUnaryOp<AlignedArray, expOp>);
  m.def("ewise_tanh", EwiseUnaryOp<AlignedArray, tanhOp>);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
