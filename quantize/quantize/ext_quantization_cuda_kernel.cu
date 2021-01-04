/*
 * Cuda kernels for quantization and mixed-precision packing
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void compute_scale_kernel(const int32_t* __restrict__ bits,
                                     const float* __restrict__ min,
                                     const float* __restrict__ max,
                                     float* __restrict__ scale,
                                     int N,
                                     int num_groups) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N * num_groups) {
    scale[id] = ((float)((1 << bits[id / num_groups]) - 1)) / (max[id] - min[id] + 2e-6);
  }
}

// Pack float32 data into int32 bit stream
__global__ void pack_mixed_precision_kernel(const int32_t* __restrict__ bits,
                                            const int32_t* __restrict__ prefix_sum,
                                            const float* __restrict__ data,
                                            const float* __restrict__ scale,
                                            const float* __restrict__ min,
                                            const float* __restrict__ noise,
                                            int32_t* __restrict__ packed,
                                            int N,
                                            int num_groups,
                                            int group_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N * num_groups * group_size) {
    int n = id / (num_groups * group_size);
    int group_id = (id / group_size) % num_groups;
    int d = id % group_size;
  
    int bit_offset = (n == 0 ? 0 : prefix_sum[n-1]) * (num_groups * group_size) +
                     bits[n] * (group_id * group_size + d);
  
    int val = __float2int_rn(fmax((data[id] - min[n * num_groups + group_id]) * scale[n * num_groups + group_id] + noise[id] - 0.5, 0.0f));
  
    for (int i = 0; i < bits[n]; i++) {
      atomicOr(packed + (bit_offset + i) / 32, (1 & (val >> i)) << ((bit_offset + i) % 32));
    }
  }
}

// Pack float32 data into int32 bit stream
std::pair<torch::Tensor, torch::Tensor> pack_mixed_precision_cuda(torch::Tensor data,
                                                                  torch::Tensor min,
                                                                  torch::Tensor max,
                                                                  torch::Tensor bits,
                                                                  bool stochastic) {
  int N = data.size(0);
  int num_groups = data.size(1);
  int group_size = data.size(2);

  torch::Tensor prefix_sum = torch::cumsum(bits, 0, torch::kInt32);
  int total_bits = prefix_sum[-1].item<int32_t>() * num_groups * group_size;

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  torch::Tensor packed = torch::zeros({(total_bits + 31) / 32,}, options);
  options = torch::TensorOptions().dtype(torch::kFloat).device(data.device());
  torch::Tensor scale = torch::empty({N, num_groups, 1}, options);

  int threads = 256;
  int blocks = (N * num_groups + threads - 1) / threads;

  compute_scale_kernel<<<blocks, threads>>>(
    bits.data_ptr<int32_t>(), min.data_ptr<float>(), max.data_ptr<float>(),
    scale.data_ptr<float>(), N, num_groups);

  blocks = (N * num_groups * group_size + threads - 1) / threads;

  torch::Tensor noise;
  if (stochastic) {
    noise = torch::rand({N, num_groups, group_size}, options);
  } else {
    noise = torch::full({N, num_groups, group_size}, 0.5, options);
  }

  pack_mixed_precision_kernel<<<blocks, threads>>>(
    bits.data_ptr<int32_t>(), prefix_sum.data_ptr<int32_t>(),
    data.data_ptr<float>(),
    scale.data_ptr<float>(), min.data_ptr<float>(),
    noise.data_ptr<float>(),
    packed.data_ptr<int32_t>(),
    N, num_groups, group_size);

  return std::make_pair(packed, scale);
}

// Unpack int32 bit stream to float32 data
__global__ void unpack_mixed_precision_kernel(const int32_t* __restrict__ bits,
                                              const int32_t* __restrict__ prefix_sum,
                                              const int32_t* __restrict__ data,
                                              const float* __restrict__ scale,
                                              const float* __restrict__ min,
                                              float* __restrict__ unpacked,
                                              int N,
                                              int num_groups,
                                              int group_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N * num_groups * group_size) {
    int n = id / (num_groups * group_size);
    int group_id = (id / group_size) % num_groups;
    int d = id % group_size;

    int bit_offset = (n == 0 ? 0 : prefix_sum[n-1]) * (num_groups * group_size) +
                     bits[n] * (group_id * group_size + d);
  
    int val = 0;
    for (int i = 0; i < bits[n]; i++) {
      val |= (1 & (data[(bit_offset + i) / 32] >> ((bit_offset + i) % 32))) << i;
    }

    unpacked[id] = ((float)val) / scale[n * num_groups + group_id] + min[n * num_groups + group_id];
  }
}

// Unpack int32 bit stream to float32 data
torch::Tensor unpack_mixed_precision_cuda(torch::Tensor data,
                                          torch::Tensor bits,
                                          torch::Tensor scale,
                                          torch::Tensor min,
                                          int N,
                                          int num_groups,
                                          int group_size) {
  torch::Tensor prefix_sum = torch::cumsum(bits, 0, torch::kInt32);

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(data.device());
  torch::Tensor unpacked = torch::empty({N, num_groups, group_size}, options);

  int threads = 128;
  int blocks = (N * num_groups * group_size + threads - 1) / threads;

  unpack_mixed_precision_kernel<<<blocks, threads>>>(
    bits.data_ptr<int32_t>(), prefix_sum.data_ptr<int32_t>(),
    data.data_ptr<int32_t>(), 
    scale.data_ptr<float>(), min.data_ptr<float>(),
    unpacked.data_ptr<float>(),
    N, num_groups, group_size);

  return unpacked;
}

// Unpack int32 bit stream to float32 data
__global__ void act_quantized_relu_forward_kernel(const float* __restrict__ data,
                                                  int32_t* __restrict__ mask,
                                                  float* __restrict__ output,
                                                  int N,
                                                  int mask_len) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N) {
    int bit = data[id] > 0;
    atomicOr(mask + id % mask_len, bit << (id / mask_len));
    output[id] = fmax(data[id], 0.0f);
  }
}

std::pair<torch::Tensor, torch::Tensor> act_quantized_relu_forward_cuda(torch::Tensor data) {
  int n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
  int mask_len = (n_elements + 31) / 32;
  torch::Tensor mask = torch::zeros({mask_len}, options);
  torch::Tensor output = torch::empty_like(data);

  int threads = 256;
  int blocks = (n_elements + threads - 1) / threads;

  act_quantized_relu_forward_kernel<<<blocks, threads>>>(
    data.data_ptr<float>(), mask.data_ptr<int32_t>(), output.data_ptr<float>(),
    n_elements, mask_len);

  return std::make_pair(output, mask);
}

__global__ void act_quantized_relu_backward_kernel(const float* __restrict__ data,
                                                   int32_t* __restrict__ mask,
                                                   float* __restrict__ output,
                                                   int N,
                                                   int mask_len) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < N) {
    if ((mask[id % mask_len] >> (id / mask_len)) & 1) {
      output[id] = data[id];
    }
  }
}

torch::Tensor act_quantized_relu_backward_cuda(torch::Tensor mask, torch::Tensor data) {
  int n_elements = 1;
  for (size_t i = 0; i < data.dim(); ++i) {
    n_elements *= data.size(i);
  }

  int mask_len = (n_elements + 31) / 32;
  int threads = 256;
  int blocks = (n_elements + threads - 1) / threads;

  torch::Tensor output = torch::zeros_like(data);

  act_quantized_relu_backward_kernel<<<blocks, threads>>>(
    data.data_ptr<float>(), mask.data_ptr<int32_t>(), output.data_ptr<float>(),
    n_elements, mask_len);

  return output;
}
