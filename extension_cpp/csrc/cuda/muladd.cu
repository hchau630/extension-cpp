#include <torch/extension.h>
#include <ATen/native/cuda/Loops.cuh>
#include "../mymath.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp {

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(c.device().type() == at::DeviceType::CUDA);
  at::Tensor result = torch::empty(at::IntArrayRef(), at::DeviceType::CUDA).resize_(0);
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "mymuladd_cuda", [&]() {
    at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        return calc_mymuladd(a, b, c);
    });
  });
  return result;
}

// __global__ void muladd_kernel(int numel, const float* a, const float* b, const float* c, float* result) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < numel) result[idx] = a[idx] * b[idx] + c[idx];
// }
// 
// 
// at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
//   TORCH_CHECK(a.sizes() == b.sizes());
//   TORCH_CHECK(a.dtype() == at::kFloat);
//   TORCH_CHECK(b.dtype() == at::kFloat);
//   TORCH_CHECK(c.dtype() == at::kFloat);
//   TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
//   TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
//   TORCH_INTERNAL_ASSERT(c.device().type() == at::DeviceType::CUDA);
//   at::Tensor a_contig = a.contiguous();
//   at::Tensor b_contig = b.contiguous();
//   at::Tensor c_contig = c.contiguous();
//   at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
//   const float* a_ptr = a_contig.data_ptr<float>();
//   const float* b_ptr = b_contig.data_ptr<float>();
//   const float* c_ptr = c_contig.data_ptr<float>();
//   float* result_ptr = result.data_ptr<float>();
// 
//   int numel = a_contig.numel();
//   muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c_ptr, result_ptr);
//   return result;
// }

at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA); 
  at::Tensor result = torch::empty(at::IntArrayRef(), at::DeviceType::CUDA).resize_(0);
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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "mymul_cuda", [&]() {
    at::native::gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return calc_mymul(a, b);
    });
  });
  return result;
}

// __global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < numel) result[idx] = a[idx] * b[idx];
// }
// 
// at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
//   TORCH_CHECK(a.sizes() == b.sizes());
//   TORCH_CHECK(a.dtype() == at::kFloat);
//   TORCH_CHECK(b.dtype() == at::kFloat);
//   TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
//   TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
//   at::Tensor a_contig = a.contiguous();
//   at::Tensor b_contig = b.contiguous();
//   at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
//   const float* a_ptr = a_contig.data_ptr<float>();
//   const float* b_ptr = b_contig.data_ptr<float>();
//   float* result_ptr = result.data_ptr<float>();
//   int numel = a_contig.numel();
//   mul_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
//   return result;
// }

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] + b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  add_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}


// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);
}

}
