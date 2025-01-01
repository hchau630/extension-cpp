#include <torch/extension.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include "mymath.h"

#include <vector>

namespace extension_cpp {

template <typename scalar_t>
void myexp_cpu_kernel(const int64_t n, const scalar_t* a_ptr, scalar_t* result_ptr) {
  for (int64_t i = 0; i < n; i++) {
    result_ptr[i] = std::exp(a_ptr[i]);
  }
}

at::Tensor myexp_cpu(const at::Tensor& a) {
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "myexp_cpu", ([&] {
    myexp_cpu_kernel(result.numel(), a_contig.data_ptr<scalar_t>(), result.data_ptr<scalar_t>());
  }));
  return result;
}

at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  at::Tensor result = torch::empty(at::IntArrayRef()).resize_(0);
  auto iter = (
    at::TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .promote_integer_inputs_to_float(true)
    .add_output(result)
    .add_input(a)
    .add_input(b)
    .add_input(c)
  ).build();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "mymuladd_cpu", [&]() {
    at::native::cpu_kernel(iter, [](scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        return calc_mymuladd(a, b, c);
    });
  });
  return result;
}

// at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
//   TORCH_CHECK(a.sizes() == b.sizes());
//   TORCH_CHECK(b.sizes() == c.sizes());
//   TORCH_CHECK(a.dtype() == at::kFloat);
//   TORCH_CHECK(b.dtype() == at::kFloat);
//   TORCH_CHECK(c.dtype() == at::kFloat);
//   TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
//   TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
//   TORCH_INTERNAL_ASSERT(c.device().type() == at::DeviceType::CPU);
//   at::Tensor a_contig = a.contiguous();
//   at::Tensor b_contig = b.contiguous();
//   at::Tensor c_contig = c.contiguous();
//   at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
//   const float* a_ptr = a_contig.data_ptr<float>();
//   const float* b_ptr = b_contig.data_ptr<float>();
//   const float* c_ptr = c_contig.data_ptr<float>();
//   float* result_ptr = result.data_ptr<float>();
//   for (int64_t i = 0; i < result.numel(); i++) {
//     result_ptr[i] = a_ptr[i] * b_ptr[i] + c_ptr[i];
//   }
//   return result;
// }

at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
  at::Tensor result = torch::empty(at::IntArrayRef()).resize_(0);
  auto iter = (
    at::TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .promote_integer_inputs_to_float(true)
    .add_output(result)
    .add_input(a)
    .add_input(b)
  ).build();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "mymul_cpu", [&]() {
    at::native::cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        return calc_mymul(a, b);
    });
  });
  return result;
}

// at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
//   TORCH_CHECK(a.sizes() == b.sizes());
//   TORCH_CHECK(a.dtype() == at::kFloat);
//   TORCH_CHECK(b.dtype() == at::kFloat);
//   TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
//   TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
//   at::Tensor a_contig = a.contiguous();
//   at::Tensor b_contig = b.contiguous();
//   at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
//   const float* a_ptr = a_contig.data_ptr<float>();
//   const float* b_ptr = b_contig.data_ptr<float>();
//   float* result_ptr = result.data_ptr<float>();
//   for (int64_t i = 0; i < result.numel(); i++) {
//     result_ptr[i] = a_ptr[i] * b_ptr[i];
//   }
//   return result;
// }

// An example of an operator that mutates one of its inputs.
void myadd_out_cpu(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); i++) {
    result_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(extension_cpp, m) {
  m.def("myexp(Tensor a) -> Tensor");
  m.def("mymuladd(Tensor a, Tensor b, Tensor c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("myexp", &myexp_cpu);
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);
}

}
