#include "./arknife_common.h"
__device__ __forceinline__
void device_func(const cute::TmaDescriptor& A, const cute::TmaDescriptor& B, const cute::TmaDescriptor& C, int _lane_dim, int _warp_dim, int _warpgroup_dim, int _block_dim, int _cluster_dim, int _lane_idx, int _warp_idx, int _warpgroup_idx, int _block_idx, int _cluster_idx) {
  int lane_idx = _lane_idx, warp_idx = _warp_idx, warpgroup_idx = _warpgroup_idx, block_idx = _block_idx, cluster_idx = _cluster_idx;
  constexpr int lane_dim = 32, warp_dim = 4, warpgroup_dim = 2, block_dim = 2, cluster_dim = 76, grid_dim = 1;
  int sp_m_e128s32_ = warp_idx/1%4, sp_m_e512s256_ = block_idx/1%2;
  
  extern __shared__ unsigned char shared_storage[];
  bf16_t* (*A_SMEM)[16384] = reinterpret_cast<decltype(A_SMEM)>(shared_storage + 0);
  bf16_t* (*B_SMEM)[8192] = reinterpret_cast<decltype(B_SMEM)>(shared_storage + 131072);
  bf16_t* (*C_SMEM)[8192] = reinterpret_cast<decltype(C_SMEM)>(shared_storage + 196608);
  
  if (warpgroup_idx == 0 && warp_idx == 0) cute::TMEM::Allocator2Sm().allocate(512, reinterpret_cast<uint32_t*>(shared_storage));
  
  uint32_t C_TMEM[1] = {0};
  ClusterChannel<4>& A_SMEM_empty_chans = *reinterpret_cast<ClusterChannel<4>*>(shared_storage + 229376);
  ClusterChannel<4>& A_SMEM_ready_chans = *reinterpret_cast<ClusterChannel<4>*>(shared_storage + 229376 + sizeof(ClusterChannel<4>));
  ClusterChannel<1>& C_TMEM_empty_chans = *reinterpret_cast<ClusterChannel<1>*>(shared_storage + 229376 + sizeof(ClusterChannel<4>) + sizeof(ClusterChannel<4>));
  ClusterChannel<1>& C_TMEM_ready_chans = *reinterpret_cast<ClusterChannel<1>*>(shared_storage + 229376 + sizeof(ClusterChannel<4>) + sizeof(ClusterChannel<4>) + sizeof(ClusterChannel<1>));
  ARK_UNROLL for (int i = 0; i < 4; ++i) {
    if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) A_SMEM_empty_chans[i].init(1);
  }
  ARK_UNROLL for (int i = 0; i < 4; ++i) {
    if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) A_SMEM_ready_chans[i].init(2);
  }
  ARK_UNROLL for (int i = 0; i < 1; ++i) {
    if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) C_TMEM_empty_chans[i].init(256);
  }
  ARK_UNROLL for (int i = 0; i < 1; ++i) {
    if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) C_TMEM_ready_chans[i].init(1);
  }
  if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) {
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::fence_barrier_init();
  }
  f32_t C_REG[8];
  bf16_t C_REG_BF16[8];
  PipelineState<4> A_SMEM_state;
  PipelineState<1> C_TMEM_state;
  PipelineState<4> A_SMEM_state1;
  PipelineState<1> C_TMEM_state1;
  PipelineState<2> C_SMEM_state;
  PipelineState<2> C_SMEM_state1;
  if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) {
    cute::prefetch_tma_descriptor(&A);
    cute::prefetch_tma_descriptor(&B);
    cute::prefetch_tma_descriptor(&C);
  }
  cute::cluster_sync();
  if (warpgroup_idx == 0 && warp_idx == 0) {
    for (int cluster_idx = _cluster_idx; cluster_idx < 2048; cluster_idx += cluster_dim) {
        break;
      int sp_m_e16384s512_ = cluster_idx/64%32, sp_n_e16384s256_ = cluster_idx/1%64;
      for (int k_e16384s64 = 0; k_e16384s64 < 256; ++k_e16384s64) {
        A_SMEM_empty_chans[(A_SMEM_state.index % A_SMEM_empty_chans.N_SLOTS)].wait(A_SMEM_state.phase ^ 1);
        #define A_SMEM_off(i0, i1) i0*64+((i0)%64^(i1))*1
        #define A_off(i0, i1, i2) sp_m_e16384s512_*8388608+k_e16384s64*64+i0*4194304+i1*16384+i2
        if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) cute::SM100_TMA_2SM_LOAD_2D::copy(&A, reinterpret_cast<uint64_t*>(&A_SMEM_ready_chans[(A_SMEM_state.index % A_SMEM_ready_chans.N_SLOTS)]), static_cast<uint32_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), ((bf16_t*)A_SMEM[A_SMEM_state.stage]+A_SMEM_off(0, 0)), k_e16384s64, sp_m_e16384s512_);
        #define B_SMEM_off(i0, i1) i0*64+((i0)%64^(i1))*1
        #define B_off(i0, i1, i2) sp_n_e16384s256_*4194304+k_e16384s64*64+i0*2097152+i1*16384+i2
        if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) cute::SM100_TMA_2SM_LOAD_2D::copy(&B, reinterpret_cast<uint64_t*>(&A_SMEM_ready_chans[(A_SMEM_state.index % A_SMEM_ready_chans.N_SLOTS)]), static_cast<uint32_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), ((bf16_t*)B_SMEM[A_SMEM_state.stage]+B_SMEM_off(0, 0)), k_e16384s64, sp_n_e16384s256_);
        if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) { if (block_idx == 0) A_SMEM_ready_chans[(A_SMEM_state.index % A_SMEM_ready_chans.N_SLOTS)].arrive_and_expect_tx(65536 + 32768); else A_SMEM_ready_chans[(A_SMEM_state.index % A_SMEM_ready_chans.N_SLOTS)].arrive(0u); }
        if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) { if (block_idx == 0) A_SMEM_ready_chans[(A_SMEM_state.index % A_SMEM_ready_chans.N_SLOTS)].arrive(); else A_SMEM_ready_chans[(A_SMEM_state.index % A_SMEM_ready_chans.N_SLOTS)].arrive(0u); }
        A_SMEM_state.next();
      }
    }
  }
  else if (block_idx == 0 && warpgroup_idx == 0 && warp_idx == 1) {
    int warp_idx = _warp_idx - 1;
    constexpr int block_dim = 1, warpgroup_dim = 1, warp_dim = 1;
    for (int cluster_idx = _cluster_idx; cluster_idx < 2048; cluster_idx += cluster_dim) {
      int sp_m_e16384s512_ = cluster_idx/64%32, sp_n_e16384s256_ = cluster_idx/1%64;
      #define C_TMEM_off() 0
      uint32_t C_TMEM_inited; C_TMEM_inited = 0;
      C_TMEM_empty_chans[(C_TMEM_state.index % C_TMEM_empty_chans.N_SLOTS)].wait(C_TMEM_state.phase ^ 1);
      for (int k_e16384s64 = 0; k_e16384s64 < 256; ++k_e16384s64) {
        A_SMEM_ready_chans[(A_SMEM_state1.index % A_SMEM_ready_chans.N_SLOTS)].wait(A_SMEM_state1.phase);
        for (int k_e64s16 = 0; k_e64s16 < 4; ++k_e64s16) for (int m_e256s128 = 0; m_e256s128 < 2; ++m_e256s128) {
          #define C_TMEM_off1(i0, i1, i2) m_e256s128*256+i0*128+i1*8+i2
          #define A_SMEM_off1(i0, i1, i2) m_e256s128*8192+i0*512+i1*64+((m_e256s128*128+i0*8+i1)%64^(k_e64s16*16+i2))*1
          #define B_SMEM_off1(i0, i1, i2) i0*512+i1*64+((i0*8+i1)%64^(k_e64s16*16+i2))*1
          cute::SM100_MMA_F16BF16_2x1SM_SS<bf16_t, bf16_t, float, 256, 256, cute::UMMA::Major::K, cute::UMMA::Major::K>::fma(make_8x128B_atom_smem_desc(static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(((bf16_t*)A_SMEM[A_SMEM_state1.stage]+A_SMEM_off1(0, 0, 0))))), make_8x128B_atom_smem_desc(static_cast<uint16_t>(cute::cast_smem_ptr_to_uint(((bf16_t*)B_SMEM[A_SMEM_state1.stage]+B_SMEM_off1(0, 0, 0))))), ((uint32_t)C_TMEM[C_TMEM_state.stage]+C_TMEM_off1(0, 0, 0)), (C_TMEM_inited & (1 << (C_TMEM_off1(0, 0, 0) / 256))) != 0, cute::UMMA::make_runtime_instr_desc(cute::UMMA::make_instr_desc<bf16_t, bf16_t, float, 256, 256, cute::UMMA::Major::K, cute::UMMA::Major::K>())); C_TMEM_inited |= (1 << (C_TMEM_off1(0, 0, 0) / 256));
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&A_SMEM_empty_chans[(A_SMEM_state1.index % A_SMEM_empty_chans.N_SLOTS)]), 3);
        A_SMEM_state1.next();
      }
      cutlass::arch::umma_arrive_multicast_2x1SM(reinterpret_cast<uint64_t*>(&C_TMEM_ready_chans[(C_TMEM_state.index % C_TMEM_ready_chans.N_SLOTS)]), 3);
      C_TMEM_state.next();
    }
  }
  else if (warpgroup_idx == 1) {
    int warpgroup_idx = _warpgroup_idx - 1;
    constexpr int warpgroup_dim = 1;
    #define CUR_COROUTINE_NAME seq_task_0
    #define n_e64s8 CO_VAR(n_e64s8)
    #define n_e256s64 CO_VAR(n_e256s64)
    #define m_e256s128 CO_VAR(m_e256s128)
    #define sp_n_e16384s256_ CO_VAR(sp_n_e16384s256_)
    #define n_e8s1 CO_VAR(n_e8s1)
    #define cluster_idx CO_VAR(cluster_idx)
    #define sp_m_e16384s512_ CO_VAR(sp_m_e16384s512_)
    int n_e64s8, n_e256s64, m_e256s128, sp_n_e16384s256_, n_e8s1, cluster_idx, sp_m_e16384s512_;
    CO_BEGIN
    for (cluster_idx = _cluster_idx; cluster_idx < 2048; cluster_idx += cluster_dim) {
      sp_m_e16384s512_ = cluster_idx/64%32, sp_n_e16384s256_ = cluster_idx/1%64;
      C_TMEM_ready_chans[(C_TMEM_state1.index % C_TMEM_ready_chans.N_SLOTS)].wait(C_TMEM_state1.phase);
      for (m_e256s128 = 0; m_e256s128 < 2; ++m_e256s128) for (n_e256s64 = 0; n_e256s64 < 4; ++n_e256s64) {
        while (C_SMEM_state.index >= C_SMEM_state1.index + 2) CO_YIELD;
        if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) cute::tma_store_wait<1>();
        cutlass::arch::NamedBarrier(warpgroup_dim * warp_dim * lane_dim).sync();
        for (n_e64s8 = 0; n_e64s8 < 8; ++n_e64s8) {
          #define C_REG_off(i0) i0
          #define C_TMEM_off2(i0) m_e256s128*256+n_e256s64*64+n_e64s8*8+i0
          cute::SM100_TMEM_LOAD_32dp32b8x::copy(((uint32_t)C_TMEM[C_TMEM_state1.stage]+C_TMEM_off2(0)), ((u32_t*)C_REG)[C_REG_off(0)], ((u32_t*)C_REG)[C_REG_off(1)], ((u32_t*)C_REG)[C_REG_off(2)], ((u32_t*)C_REG)[C_REG_off(3)], ((u32_t*)C_REG)[C_REG_off(4)], ((u32_t*)C_REG)[C_REG_off(5)], ((u32_t*)C_REG)[C_REG_off(6)], ((u32_t*)C_REG)[C_REG_off(7)]); cutlass::arch::fence_view_async_tmem_load();
          for (n_e8s1 = 0; n_e8s1 < 8; ++n_e8s1) {
            ((bf16_t*)C_REG_BF16)[n_e8s1] = bf16_t(((f32_t*)C_REG)[n_e8s1]);
          }
          #define C_SMEM_off(i0, i1, i2) sp_m_e128s32_*2048+lane_idx/1%32*64+((sp_m_e128s32_*32+lane_idx/1%32)%64^(n_e64s8*8+i1*2+i2))*1
          #define C_REG_BF16_off(i0) i0
          asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(reinterpret_cast<uint8_t*>(((bf16_t*)C_SMEM[C_SMEM_state.stage]+C_SMEM_off(0, 0, 0)))), "r"(((u32_t*)C_REG_BF16)[C_REG_BF16_off(0)]), "r"(((u32_t*)C_REG_BF16)[C_REG_BF16_off(1)]), "r"(((u32_t*)C_REG_BF16)[C_REG_BF16_off(2)]), "r"(((u32_t*)C_REG_BF16)[C_REG_BF16_off(3)]));
        }
        cutlass::arch::NamedBarrier(warpgroup_dim * warp_dim * lane_dim).sync();
        C_SMEM_state.next();
        CO_YIELD;
      }
      C_TMEM_empty_chans[(C_TMEM_state1.index % C_TMEM_empty_chans.N_SLOTS)].arrive();
      C_TMEM_state1.next();
    }
    CO_END
    #undef CUR_COROUTINE_NAME
    #undef n_e64s8
    #undef n_e256s64
    #undef m_e256s128
    #undef sp_n_e16384s256_
    #undef n_e8s1
    #undef cluster_idx
    #undef sp_m_e16384s512_
    
    #define CUR_COROUTINE_NAME seq_task_1
    #define n_e256s64 CO_VAR(n_e256s64)
    #define m_e256s128 CO_VAR(m_e256s128)
    #define sp_n_e16384s256_ CO_VAR(sp_n_e16384s256_)
    #define cluster_idx CO_VAR(cluster_idx)
    #define sp_m_e16384s512_ CO_VAR(sp_m_e16384s512_)
    int n_e256s64, m_e256s128, sp_n_e16384s256_, cluster_idx, sp_m_e16384s512_;
    CO_BEGIN
    for (cluster_idx = _cluster_idx; cluster_idx < 2048; cluster_idx += cluster_dim) {
      sp_m_e16384s512_ = cluster_idx/64%32, sp_n_e16384s256_ = cluster_idx/1%64;
      for (m_e256s128 = 0; m_e256s128 < 2; ++m_e256s128) for (n_e256s64 = 0; n_e256s64 < 4; ++n_e256s64) {
        while (C_SMEM_state.index <= C_SMEM_state1.index) CO_YIELD;
        #define C_off(i0, i1) sp_m_e16384s512_*8388608+sp_n_e16384s256_*256+sp_m_e512s256_*4194304+m_e256s128*2097152+n_e256s64*64+i0*16384+i1
        #define C_SMEM_off1(i0, i1) i0*64+((i0)%64^(i1))*1
        cute::SM90_TMA_STORE_2D::copy(&C_SMEM, ((bf16_t*)C_SMEM[C_SMEM_state1.stage]+C_SMEM_off1(0, 0)), sp_n_e16384s256_*256+n_e256s64*64, sp_m_e16384s512_*8388608+sp_m_e512s256_*4194304+m_e256s128*2097152);
        if (warpgroup_idx == 0 && warp_idx == 0 && lane_idx == 0) cute::tma_store_arrive();
        C_SMEM_state1.next();
      }
    }
    CO_END
    #undef CUR_COROUTINE_NAME
    #undef n_e256s64
    #undef m_e256s128
    #undef sp_n_e16384s256_
    #undef cluster_idx
    #undef sp_m_e16384s512_
    
    while (seq_task_0() | seq_task_1());
  }
  if (warpgroup_idx == 0 && warp_idx == 0) cute::TMEM::Allocator2Sm().free(0, 512);
}
__global__
__cluster_dims__(2, 1, 1)
void kernel_func(const __grid_constant__ cute::TmaDescriptor A, const __grid_constant__ cute::TmaDescriptor B, const __grid_constant__ cute::TmaDescriptor C, int _lane_dim, int _warp_dim, int _warpgroup_dim, int _block_dim, int _cluster_dim) {
  device_func(A, B, C, _lane_dim, _warp_dim, _warpgroup_dim, _block_dim, _cluster_dim, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y);
}
extern "C" int matmul_sm100_bf16_2sm_256x256x64(bf16_t* A, bf16_t* B, bf16_t* C) {
  constexpr uint32_t smem_size = 229376 + sizeof(ClusterChannel<4>) + sizeof(ClusterChannel<4>) + sizeof(ClusterChannel<1>) + sizeof(ClusterChannel<1>);
  static bool init_flag = false;
  if (!init_flag) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size), return -1);
    init_flag = true;
  }
  const auto A_desc = make_2d_tensor_map(A, 16384, 16384, 64, 256, 16384);
  const auto B_desc = make_2d_tensor_map(B, 16384, 16384, 64, 128, 16384);
  const auto C_desc = make_2d_tensor_map(C, 16384, 16384, 64, 128, 16384);
  kernel_func<<<dim3(2, 76, 1), dim3(32, 4, 2), smem_size>>>(A_desc, B_desc, C_desc, 32, 4, 2, 2, 76);
  CUDA_CHECK(cudaGetLastError(), return -1);
  return 0;
}

