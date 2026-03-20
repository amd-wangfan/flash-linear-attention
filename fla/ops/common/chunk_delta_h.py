# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl

from fla.ops.backends import dispatch
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp, exp2
from fla.utils import IS_AMD_MI325, IS_NVIDIA_HOPPER, USE_CUDA_GRAPH, autotune_cache_kwargs, check_shared_mem, get_multiprocessor_count

NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]
FWD_H_NUM_WARPS = [4, 8, 16] if IS_AMD_MI325 else [2, 4]
FWD_H_BV_LIST = [16, 32, 64] if IS_AMD_MI325 else ([32, 64] if check_shared_mem('ada') else [32])
FWD_H_NUM_STAGES = [1, 2] if IS_AMD_MI325 else ([4, 3, 2] if check_shared_mem('ampere') else [2, 1])
BWD_H_NUM_WARPS = [1, 2, 4] if IS_AMD_MI325 else [2, 4]
BWD_H_NUM_STAGES = [1, 2] if IS_AMD_MI325 else ([4, 3, 2] if check_shared_mem('ampere') else [1])
BWD_H_BV_LIST = [8, 16, 32] if IS_AMD_MI325 else ([64, 32] if check_shared_mem('ada') else [32])

# Persistent kernel configs for MI325X: includes BV=8 for maximum CU occupancy
PERSISTENT_FWD_H_BV_LIST = [8, 16, 32]
PERSISTENT_FWD_H_NUM_WARPS = [4, 8, 16]
PERSISTENT_FWD_H_NUM_STAGES = [1, 2]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in FWD_H_NUM_WARPS
        for num_stages in FWD_H_NUM_STAGES
        for BV in FWD_H_BV_LIST
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G', 'SAVE_NEW_VALUE', 'USE_EXP2', 'TRANSPOSE_STATE'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K*V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_h2 *= exp(b_gk_last2)[None, :]
                else:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_h3 *= exp(b_gk_last3)[None, :]
                else:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_h4 *= exp(b_gk_last4)[None, :]
                else:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in PERSISTENT_FWD_H_NUM_WARPS
        for num_stages in PERSISTENT_FWD_H_NUM_STAGES
        for BV in PERSISTENT_FWD_H_BV_LIST
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G', 'SAVE_NEW_VALUE', 'USE_EXP2'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_persistent(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NH,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    """Persistent fwd_h kernel for MI325X.

    Uses a 1D grid of NUM_SMS blocks. Each block picks up (i_v, i_nh) work
    items in round-robin order, enabling BV=8 for up to 256 blocks on 304 CUs.
    Non-varlen only.
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    NUM_V_BLOCKS = tl.cdiv(V, BV)
    TOTAL_WORK = NUM_V_BLOCKS * NH
    NT = tl.cdiv(T, BT)

    work_id = pid
    while work_id < TOTAL_WORK:
        i_v = work_id % NUM_V_BLOCKS
        i_nh = work_id // NUM_V_BLOCKS
        i_n = i_nh // H
        i_h = i_nh % H

        bos = i_n * T
        boh = i_n * NT

        # [BK, BV]
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

        # calculate base pointers
        h_base = h + (boh * H + i_h).to(tl.int64) * K * V
        v_base = v + (bos * H + i_h).to(tl.int64) * V
        k_base = k + (bos * H + i_h).to(tl.int64) * K
        w_base = w + (bos * H + i_h).to(tl.int64) * K

        if USE_INITIAL_STATE:
            h0_base = h0 + i_nh * K * V

        # load initial state
        if USE_INITIAL_STATE:
            p_h0_1 = tl.make_block_ptr(h0_base, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                p_h0_2 = tl.make_block_ptr(h0_base, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                p_h0_3 = tl.make_block_ptr(h0_base, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                p_h0_4 = tl.make_block_ptr(h0_base, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

        # main recurrence
        for i_t in range(NT):
            i_t_int64 = i_t.to(tl.int64)
            p_h1 = tl.make_block_ptr(h_base + i_t_int64 * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(h_base + i_t_int64 * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(h_base + i_t_int64 * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(h_base + i_t_int64 * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

            p_w = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w_base, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
            p_v = tl.make_block_ptr(v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

            if SAVE_NEW_VALUE:
                p_vn = tl.make_block_ptr(v_new + (bos * H + i_h).to(tl.int64) * V,
                                         (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

            last_idx = min((i_t + 1) * BT, T) - 1
            if USE_G:
                m_t = (i_t * BT + tl.arange(0, BT)) < T
                b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(tl.float32)
                p_g = tl.make_block_ptr(g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                if USE_EXP2:
                    b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                else:
                    b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            if USE_GK:
                o_k1 = tl.arange(0, 64)
                b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                                     mask=(o_k1 < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                                         mask=(o_k2 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                                         mask=(o_k3 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
                if K > 192:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                                         mask=(o_k4 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

            b_v = b_v.to(k.dtype.element_ty)

            p_k = tl.make_block_ptr(k_base, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h1 += tl.dot(b_k, b_v)
            if K > 64:
                p_k = tl.make_block_ptr(k_base, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h2 += tl.dot(b_k, b_v)
            if K > 128:
                p_k = tl.make_block_ptr(k_base, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h3 += tl.dot(b_k, b_v)
            if K > 192:
                p_k = tl.make_block_ptr(k_base, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h4 += tl.dot(b_k, b_v)

        if STORE_FINAL_STATE:
            ht_base = ht + i_nh * K * V
            p_ht = tl.make_block_ptr(ht_base, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_ht = tl.make_block_ptr(ht_base, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_ht = tl.make_block_ptr(ht_base, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_ht = tl.make_block_ptr(ht_base, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))

        # next work item (round-robin stride)
        work_id += num_pids


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in BWD_H_NUM_WARPS
        for num_stages in BWD_H_NUM_STAGES
        for BV in BWD_H_BV_LIST
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G', 'USE_GK', 'USE_EXP2', 'TRANSPOSE_STATE'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_dh1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    do += (bos * H + i_h).to(tl.int64) * V
    dv += (bos * H + i_h).to(tl.int64) * V
    dv2 += (bos * H + i_h).to(tl.int64) * V
    dh += (boh * H + i_h).to(tl.int64) * K*V
    if USE_GK:
        gk += (bos * H + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dht1 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dht2 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dht3 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dht4 = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)
            else:
                bg_last_exp = exp(bg_last)
                b_g_exp = exp(b_g)

        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * H*K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        if TRANSPOSE_STATE:
            b_dv = tl.dot(b_k, tl.trans(b_dh1).to(b_k.dtype))
        else:
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * H*K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh2).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * H*K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh3).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * H*K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                b_dv += tl.dot(b_k, tl.trans(b_dh4).to(b_k.dtype))
            else:
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            if USE_EXP2:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            else:
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_dh1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1[:, None])
                else:
                    b_dh1 *= exp(b_gk_last1[:, None])
        if TRANSPOSE_STATE:
            b_dh1 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
        else:
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_dh2 *= exp(b_gk_last2)[None, :]
                else:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2[:, None])
                    else:
                        b_dh2 *= exp(b_gk_last2[:, None])
            if TRANSPOSE_STATE:
                b_dh2 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_dh3 *= exp(b_gk_last3)[None, :]
                else:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3[:, None])
                    else:
                        b_dh3 *= exp(b_gk_last3[:, None])
            if TRANSPOSE_STATE:
                b_dh3 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_dh4 *= exp(b_gk_last4)[None, :]
                else:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4[:, None])
                    else:
                        b_dh4 *= exp(b_gk_last4[:, None])
            if TRANSPOSE_STATE:
                b_dh4 += tl.trans(tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype)))
            else:
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_dh0 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_dh1 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_dh2 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_dh3 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def _lightweight_cp_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
    save_new_value: bool,
    use_exp2: bool,
    h: torch.Tensor,
    v_new: torch.Tensor | None,
    final_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Lightweight CP for non-varlen B=1 on MI325X.

    Instead of the full intracard machinery (cu_seqlens, IS_VARLEN, prepare_chunk_indices),
    reshape (1, T) -> (S, T/S) and run the persistent kernel with B=S.
    Pre-scan and merge use the same reshaping trick.
    """
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size
    NUM_SMS = get_multiprocessor_count()

    # Compute number of splits: target enough blocks to fill CUs
    NT = triton.cdiv(T, BT)
    # Each split needs at least 32 chunks for the pre_scan to be worthwhile
    MIN_CHUNKS_PER_SPLIT = 32
    # Target: ~8 splits for T=16384 (32 chunks each), ~16 for T=32768
    max_splits = max(1, NT // MIN_CHUNKS_PER_SPLIT)
    # Cap to avoid excessive merge overhead
    num_splits = min(max_splits, 16)
    # Ensure T is evenly divisible by num_splits * chunk_size
    while num_splits > 1 and (T % (num_splits * BT)) != 0:
        num_splits -= 1

    if num_splits <= 1:
        # Fall back to persistent kernel without splitting
        chunk_gated_delta_rule_fwd_kernel_h_persistent[(NUM_SMS,)](
            k=k, v=u, w=w, v_new=v_new, g=g, gk=gk,
            h=h, h0=initial_state, ht=final_state,
            T=T, H=H, K=K, V=V, BT=BT, NH=B * H,
            USE_EXP2=use_exp2,
        )
        return h, v_new, final_state

    S = num_splits
    subT = T // S
    subNT = subT // BT

    # Reshape (1, T, H, K) -> (S, subT, H, K) — zero-copy view
    k_s = k.view(S, subT, H, K)
    w_s = w.view(S, subT, H, K)
    u_s = u.view(S, subT, H, V)
    g_s = g.view(S, subT, H) if g is not None else None
    gk_s = gk.view(S, subT, H, K) if gk is not None else None
    # h output: (1, NT, H, K, V) -> need (S, subNT, H, K, V)
    h_s = h.view(S, subNT, H, K, V)
    v_new_s = v_new.view(S, subT, H, V) if v_new is not None else None

    # Step 1: Run fwd_h for all splits independently (h0=0), extract final states
    # This gives us both h[t] (wrong for splits>0) and v_new (wrong for splits>0)
    # AND the final state of each split (= the carry for the next split).
    split_final = k.new_empty(S, H, K, V, dtype=torch.float32)
    chunk_gated_delta_rule_fwd_kernel_h_persistent[(NUM_SMS,)](
        k=k_s, v=u_s, w=w_s, v_new=v_new_s, g=g_s, gk=gk_s,
        h=h_s, h0=None, ht=split_final,
        T=subT, H=H, K=K, V=V, BT=BT, NH=S * H,
        USE_EXP2=use_exp2,
    )
    # split_final[i] = final state of split i when starting from h0=0

    # Step 2: Sequential carry propagation to get correct initial states
    # For the affine recurrence, the correction requires re-running with correct h0.
    # Since pre_scan computes (A, b) carry matrices which is expensive,
    # use a simpler approach: sequentially run splits 0..S-1, accumulating states.
    # split 0's result is already correct (h0=0 or initial_state=0 when None).
    # For splits 1..S-1, we need to re-run with the correct initial state.
    #
    # Approach: extract the carry from the pre_scan-free method.
    # Actually, the correct h0 for split i is the final state of the full
    # recurrence through splits 0..i-1. This is NOT just split_final[i-1]
    # because split_final[i-1] was computed with h0=0, not the true h0.
    #
    # For the gated delta rule: h' = decay*h + k@(v - w@h)
    # This is affine: h' = (decay - k@w) * h + k@v
    # With h0=c: h_final(c) = A*c + h_final(0), where A is the decay product
    #
    # So: true_h0[1] = split_final[0]  (split 0 starts from 0)
    #     true_h0[2] = A[1]*true_h0[1] + split_final[1]
    #                = A[1]*split_final[0] + split_final[1]
    # But we don't have A[i] without the pre_scan...
    #
    # Simpler approach: just re-run each split sequentially with correct h0.
    # This is S sequential kernel launches, each processing subT tokens.
    # Total compute = S * subT = T (same as no-split), but each launch is shorter
    # and subsequent splits can overlap with other work.
    #
    # Actually the simplest correct approach: run split 0 first (already done above,
    # result is correct). Then for each subsequent split, re-run with h0 = true final
    # state of previous splits.

    # But this defeats the purpose of parallelism! We need the pre_scan approach.
    # Let me use a lightweight pre_scan: just run fwd_h for each split to get
    # final_state(h0=0), then use the linearity property.
    # We already have split_final[i] = b[i] (the bias term).
    # We need A[i] (the decay product through split i).
    # A[i] can be computed by running fwd_h with a known h0 and comparing.
    # Or: use the pre_scan kernel directly but with the reshaped tensor.

    # Use the existing pre_scan kernel to compute carries (A, b) for each split.
    # Pass g=None following the same convention as intracard_pre_scan — the scalar
    # gate g is handled implicitly through w (which encodes exp(g)*k*beta from
    # recompute_w_u_fwd). The carry matrices are correct without explicit g.
    from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged

    # Build cu_seqlens for the S splits within the flat (1, T) tensor
    cu_seqlens_splits = torch.arange(S + 1, dtype=torch.long, device=k.device) * subT

    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    hm = k.new_empty(S, H, K, V + K, dtype=torch.float32)

    grid_ps = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H, S)
    pre_process_fwd_kernel_merged[grid_ps](
        k=k, v=u, w=w, g=None, gk=gk, hm=hm,
        cu_seqlens=cu_seqlens_splits,
        T=0, H=H, K=K, V=V, BT=BT,
        BLOCK_SIZE=BLOCK_SIZE, BK1=BK,
        USE_EXP2=use_exp2, MULTI_SEQS=True,
    )

    # Step 3: Sequential merge of carries to get correct h0 for each split
    # hm[i, h, :, :V] = b[i] (bias = final_state from h0=0)
    # hm[i, h, :, V:] = A[i] (decay product matrix)
    # true_h0[0] = initial_state or 0
    # true_h0[i] = A[i-1] @ true_h0[i-1] + b[i-1]
    correct_h0 = k.new_zeros(S, H, K, V, dtype=torch.float32)
    if initial_state is not None:
        correct_h0[0] = initial_state[0]

    for i in range(1, S):
        # A[i-1] is hm[i-1, :, :, V:V+K] of shape (H, K, K)
        # b[i-1] is hm[i-1, :, :, :V] of shape (H, K, V)
        A_prev = hm[i - 1, :, :, V:]     # (H, K, K)
        b_prev = hm[i - 1, :, :, :V]     # (H, K, V)
        h0_prev = correct_h0[i - 1]      # (H, K, V)
        # true_h0[i] = A[i-1] @ true_h0[i-1] + b[i-1]
        correct_h0[i] = torch.bmm(A_prev, h0_prev) + b_prev

    # Step 4: Re-run fwd_h for ALL splits with correct initial states
    # This overwrites h_s and v_new_s with correct values
    # Need a (S, H, K, V) tensor for final states since kernel indexes by i_nh
    split_final_out = k.new_empty(S, H, K, V, dtype=torch.float32) if output_final_state else None
    chunk_gated_delta_rule_fwd_kernel_h_persistent[(NUM_SMS,)](
        k=k_s, v=u_s, w=w_s, v_new=v_new_s, g=g_s, gk=gk_s,
        h=h_s, h0=correct_h0, ht=split_final_out,
        T=subT, H=H, K=K, V=V, BT=BT, NH=S * H,
        USE_EXP2=use_exp2,
    )

    # Extract the last split's final state as the overall final state
    if output_final_state and final_state is not None:
        final_state.copy_(split_final_out[S - 1:S])

    return h, v_new, final_state


@dispatch('common')
def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if transpose_state_layout:
        h = k.new_empty(B, NT, H, V, K)
        final_state = k.new_zeros(N, H, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, H, K, V)
        final_state = k.new_zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None

    # On MI325X for non-varlen B=1, use lightweight CP with the persistent kernel.
    # Instead of the full intracard machinery (cu_seqlens, IS_VARLEN overhead,
    # prepare_chunk_indices), we reshape (1,T) -> (S, T/S) and run the persistent
    # kernel with B=S. This avoids the 3 main overhead sources:
    # 1. prepare_chunk_indices: 2.2ms overhead eliminated
    # 2. IS_VARLEN per-block loads: eliminated (non-varlen path)
    # 3. alloc+scatter of initial_state_expanded: replaced with direct (S,H,K,V) tensor
    if IS_AMD_MI325 and cu_seqlens is None and not transpose_state_layout and B == 1 and T >= 32768:
        h, v_new, final_state = _lightweight_cp_fwd_h(
            k=k, w=w, u=u, g=g, gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=BT, save_new_value=save_new_value,
            use_exp2=use_exp2,
            h=h, v_new=v_new, final_state=final_state,
        )
    elif IS_AMD_MI325 and cu_seqlens is None and not transpose_state_layout:
        NUM_SMS = get_multiprocessor_count()
        chunk_gated_delta_rule_fwd_kernel_h_persistent[(NUM_SMS,)](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=gk,
            h=h,
            h0=initial_state,
            ht=final_state,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            NH=N * H,
            USE_EXP2=use_exp2,
        )
    else:
        def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            g=g,
            gk=gk,
            h=h,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            USE_EXP2=use_exp2,
            TRANSPOSE_STATE=transpose_state_layout,
        )
    return h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if transpose_state_layout:
        dh = q.new_empty(B, NT, H, V, K)
    else:
        dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return dh, dh0, dv2
