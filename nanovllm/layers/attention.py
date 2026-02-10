import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

# 尝试导入 flash_attn 库以利用高效的注意力计算内核
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    # 检查 GPU 是否支持 Flash Attention (Ampere 架构及以上，即能力 >= 8.0)
    _FLASH_ATTN_SUPPORTED = torch.cuda.get_device_capability()[0] >= 8
except ImportError:
    _FLASH_ATTN_SUPPORTED = False

from nanovllm.utils.context import get_context


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, block_table=None):
    """
    使用 PyTorch 原生的 scaled_dot_product_attention (SDPA) 实现变长序列的注意力计算。
    作为 Flash Attention 的回退方案。

    参数:
        q, k, v: 查询、键、值张量。
        cu_seqlens_q, cu_seqlens_k: 累积序列长度，用于切分变长批次。
        max_seqlen_q, max_seqlen_k: 最大序列长度。
        softmax_scale: 注意力评分的缩放因子。
        causal: 是否应用因果掩码（Causal Mask）。
        block_table: 分块表，用于 PagedAttention 机制下的 KV 缓存检索。
    """
    nheads = q.shape[1]
    nkvheads = k.shape[1] if block_table is None else k.shape[2]
    hdim = q.shape[2]
    out = torch.empty_like(q)
    # 遍历批次中的每个请求
    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i+1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i+1].item()
        qi = q[q_start:q_end].transpose(0, 1)  # 转换形状为 [nheads, seqlen_q, hdim]

        if block_table is not None:
            # 如果提供了 block_table，说明正在使用 PagedAttention 机制
            seqlen_k = k_end - k_start
            block_size = k.shape[1]
            num_blocks = (seqlen_k + block_size - 1) // block_size
            blocks = block_table[i, :num_blocks]
            # 从分块缓存中重新组合出完整的 K 和 V
            ki = k[blocks].reshape(-1, nkvheads, hdim)[:seqlen_k]
            vi = v[blocks].reshape(-1, nkvheads, hdim)[:seqlen_k]
        else:
            # 传统的连续内存 KV
            ki, vi = k[k_start:k_end], v[k_start:k_end]
        
        ki, vi = ki.transpose(0, 1), vi.transpose(0, 1)  # 转换形状为 [nkvheads, seqlen_k, hdim]
        
        # 如果是 Grouped Query Attention (GQA)，需要对 K 和 V 进行广播
        if nheads != nkvheads:
            ki = ki.repeat_interleave(nheads // nkvheads, dim=0)
            vi = vi.repeat_interleave(nheads // nkvheads, dim=0)
        
        # 调用 PyTorch 原生的 SDPA 内核
        oi = F.scaled_dot_product_attention(
            qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
            attn_mask=None, dropout_p=0.0, is_causal=causal, scale=softmax_scale
        )
        out[q_start:q_end] = oi.squeeze(0).transpose(0, 1)
    return out


def _sdpa_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal):
    """
    在解码阶段（Decode Stage）使用 Paged KV Cache 进行注意力计算。
    
    参数:
        q: 当前步的查询张量，形状通常为 [batch, 1, nheads, hdim]。
        k_cache, v_cache: 存储在分块内存中的 KV 缓存。
        cache_seqlens: 每个请求当前已有的 KV 长度。
        block_table: 指向 KV 块索引的表。
    """
    batch_size = q.shape[0]
    nheads = q.shape[2]
    nkvheads = k_cache.shape[2]
    hdim = q.shape[3]
    block_size = k_cache.shape[1]
    out = torch.empty((batch_size, nheads, hdim), dtype=q.dtype, device=q.device)
    
    for i in range(batch_size):
        qi = q[i].transpose(0, 1)  # [nheads, 1, hdim]
        seqlen = cache_seqlens[i].item()
        num_blocks = (seqlen + block_size - 1) // block_size
        blocks = block_table[i, :num_blocks]
        
        # 从 Paged KV Cache 中提取该请求的所有历史 KV
        ki = k_cache[blocks].reshape(-1, nkvheads, hdim)[:seqlen].transpose(0, 1)
        vi = v_cache[blocks].reshape(-1, nkvheads, hdim)[:seqlen].transpose(0, 1)
        
        # 处理 GQA
        if nheads != nkvheads:
            ki = ki.repeat_interleave(nheads // nkvheads, dim=0)
            vi = vi.repeat_interleave(nheads // nkvheads, dim=0)
            
        # 使用 SDPA 计算注意力
        oi = F.scaled_dot_product_attention(
            qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=softmax_scale
        )
        out[i] = oi.squeeze(0).transpose(0, 1).squeeze(0)
    return out


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton 内核：将当前推理步生成的 K 和 V 写入到指定的 KV Cache 插槽（Slot）中。
    这是 PagedAttention 内存管理的核心步骤，实现了非连续内存的快速写入。
    """
    idx = tl.program_id(0)
    # 加载该 token 对应的插槽索引
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return # 无效插槽则跳过
    
    # 计算当前 K, V 的读取偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # 从输入张量加载数据
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # 计算 KV Cache 的写入偏移并执行存储
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    调用 Triton 内核将新生成的 KV 存入缓存。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 确保内存布局符合 Triton 内核的预期（通常是连续的）
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 Triton Kernel
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    Attention 层，支持 Prefill（预填充）和 Decode（解码）两种推理模式。
    实现了 PagedAttention 和 FlashAttention 的集成。
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 初始化为空，运行时由 BlockManager 分配具体的 KV 缓存内存
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # 获取全局上下文信息，包含当前的推理状态（prefill/decode）、序列长度、插槽映射等
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 如果已经分配了 KV 缓存，则将当前计算得到的 K, V 存储进去
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            
        if context.is_prefill:
            # 预填充阶段：处理输入 Prompt
            if context.block_tables is not None:    # 如果使用了 Prefix Caching (前缀缓存)
                k, v = k_cache, v_cache
            
            if _FLASH_ATTN_SUPPORTED:
                # 优先使用 Flash Attention 变长内核以获得高性能
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                # 不支持时回退到原生 SDPA
                o = _sdpa_varlen(q, k, v,
                                 cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=context.cu_seqlens_k,
                                 max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
                                 softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # 解码阶段：逐 Token 推理
            if _FLASH_ATTN_SUPPORTED:
                # 使用 Flash Attention 提供的带 KV Cache 的解码优化内核
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
            else:
                # 回退方案
                o = _sdpa_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                       cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                       softmax_scale=self.scale, causal=True)
        return o

