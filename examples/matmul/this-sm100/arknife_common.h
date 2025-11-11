#pragma once

static char error_buf[1024];
extern "C" const char* get_last_error() { return error_buf; }

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>

#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/container/tuple.hpp>

using f32_t = float;
using f16_t = cutlass::half_t;
using bf16_t = cutlass::bfloat16_t;
using i32_t = int32_t;
using i16_t = int16_t;
using i8_t = int8_t;
using u32_t = uint32_t;
using u16_t = uint16_t;
using u8_t = uint8_t;
using u32x2_t = uint2;
using u32x4_t = uint4;
using i32x2_t = int2;
using i32x4_t = int4;
using f32x2_t = float2;
using f32x4_t = float4;

#define ARK_STR_(x) #x
#define ARK_STR(x) ARK_STR_(x)
#define ARK_CAT_(a, b) a##b
#define ARK_CAT(a, b) ARK_CAT_(a, b)

#define ARK_CHECK(__cond, __printf_args...)                         \
  do {                                                              \
    if (!(__cond)) {                                                \
      snprintf(error_buf, sizeof(error_buf),                        \
               __FILE__ ":" ARK_STR(__LINE__) ": check \"" ARK_STR( \
                   __cond) "\" failed: " __printf_args);            \
      snprintf(error_buf, sizeof(error_buf), "\n");                 \
      return -1;                                                    \
    }                                                               \
  } while (0)

#define CO_VAR(__x) ARK_CAT(CUR_COROUTINE_NAME, ARK_CAT(_, __x))
#define CO_RESUME_POINT CO_VAR(resume_point)
#define CO_BEGIN            \
  int CO_RESUME_POINT = -1; \
  auto CUR_COROUTINE_NAME = [&]() -> bool { switch (CO_RESUME_POINT) { case -1:
#define CO_END            \
  }                       \
  CO_RETURN;              \
  co_break_point:         \
  return CO_RESUME_POINT; \
  }                       \
  ;
#define CO_RETURN        \
  do {                   \
    CO_RESUME_POINT = 0; \
    return false;        \
  } while (0)
#define CO_YIELD                \
  do {                          \
    CO_RESUME_POINT = __LINE__; \
    goto co_break_point;        \
    case __LINE__:;             \
  } while (0)

#define ARK_UNROLL _Pragma("unroll")

#define CRZ_STR_(x) #x
#define CRZ_STR(x) CRZ_STR_(x)
#define CRZ_CHECK(__cond, __action, __printf_args...)             \
  do {                                                            \
    if (!(__cond)) {                                              \
      printf(__FILE__ ":" CRZ_STR(__LINE__) ": check \"" CRZ_STR( \
          __cond) "\" failed: " __printf_args);                   \
      printf("\n");                                               \
      __action;                                                   \
    }                                                             \
  } while (0)
#define CUDA_CHECK(cmd, act)                                             \
  do {                                                                   \
    const auto& err = (cmd);                                             \
    CRZ_CHECK(err == cudaSuccess, act, "%s - %s", cudaGetErrorName(err), \
              cudaGetErrorString(err));                                  \
  } while (0)
#define CUDA_DRIVER_CHECK(cmd, act)                  \
  do {                                               \
    const auto& err = (cmd);                         \
    const char *name, *info;                         \
    CRZ_CHECK(err == CUDA_SUCCESS, act, "%s - %s",   \
              (cuGetErrorName(err, &name), name),    \
              (cuGetErrorString(err, &info), info)); \
  } while (0)

template <typename T>
static CUtensorMap make_2d_tensor_map(T* data_ptr, int gmem_inner_dim,
                                      int gmem_outer_dim, int smem_inner_dim,
                                      int smem_outer_dim,
                                      const int& gmem_outer_stride) {
  CUtensorMap tensor_map;
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                   static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                   static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride * sizeof(T)),
  };
  const cuuint32_t elem_strides[2] = {1, 1};
  CUDA_DRIVER_CHECK(
      cuTensorMapEncodeTiled(
          &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
          reinterpret_cast<void*>(data_ptr), gmem_dims, gmem_strides, smem_dims,
          elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      throw std::runtime_error("cuda driver check failed"));
  return tensor_map;
}

__device__ __forceinline__ auto make_8x128B_atom_smem_desc(uint16_t addr) {
  cute::UMMA::SmemDescriptor desc;
  desc.version_ = 1, desc.lbo_mode_ = 0, desc.base_offset_ = 0;
  desc.layout_type_ =
      static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
  desc.stride_byte_offset_ = (8 * 128) >> 4;
  desc.leading_byte_offset_ = 0;
  desc.start_address_ = addr;
  return desc;
};

template <int N>
struct ClusterChannel
    : public cutlass::Array<cutlass::arch::ClusterTransactionBarrier, N> {
  static constexpr int N_SLOTS = N;
};

template <uint32_t NUM_STAGES>
struct PipelineState {
  uint32_t index{0}, stage{0}, phase{0};
  __device__ __forceinline__ void next() {
    ++this->index;
    if (++this->stage >= NUM_STAGES) this->stage = 0, this->phase ^= 1;
  }
  __device__ __forceinline__ void operator++() { this->next(); }
  __device__ __forceinline__ void operator++(int) { this->next(); }
};
