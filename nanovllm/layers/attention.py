import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _FLASH_ATTN_SUPPORTED = torch.cuda.get_device_capability()[0] >= 8
except ImportError:
    _FLASH_ATTN_SUPPORTED = False

from nanovllm.utils.context import get_context


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, block_table=None):
    nheads = q.shape[1]
    nkvheads = k.shape[1] if block_table is None else k.shape[2]
    hdim = q.shape[2]
    out = torch.empty_like(q)
    for i in range(len(cu_seqlens_q) - 1):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i+1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i+1].item()
        qi = q[q_start:q_end].transpose(0, 1)  # [nheads, seqlen_q, hdim]
        if block_table is not None:
            seqlen_k = k_end - k_start
            block_size = k.shape[1]
            num_blocks = (seqlen_k + block_size - 1) // block_size
            blocks = block_table[i, :num_blocks]
            ki = k[blocks].reshape(-1, nkvheads, hdim)[:seqlen_k]
            vi = v[blocks].reshape(-1, nkvheads, hdim)[:seqlen_k]
        else:
            ki, vi = k[k_start:k_end], v[k_start:k_end]
        ki, vi = ki.transpose(0, 1), vi.transpose(0, 1)  # [nkvheads, seqlen_k, hdim]
        if nheads != nkvheads:
            ki = ki.repeat_interleave(nheads // nkvheads, dim=0)
            vi = vi.repeat_interleave(nheads // nkvheads, dim=0)
        oi = F.scaled_dot_product_attention(
            qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0),
            attn_mask=None, dropout_p=0.0, is_causal=causal, scale=softmax_scale
        )
        out[q_start:q_end] = oi.squeeze(0).transpose(0, 1)
    return out


def _sdpa_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal):
    # q: [batch, 1, nheads, hdim]
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
        ki = k_cache[blocks].reshape(-1, nkvheads, hdim)[:seqlen].transpose(0, 1)
        vi = v_cache[blocks].reshape(-1, nkvheads, hdim)[:seqlen].transpose(0, 1)
        if nheads != nkvheads:
            ki = ki.repeat_interleave(nheads // nkvheads, dim=0)
            vi = vi.repeat_interleave(nheads // nkvheads, dim=0)
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
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

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
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            if _FLASH_ATTN_SUPPORTED:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                o = _sdpa_varlen(q, k, v,
                                 cu_seqlens_q=context.cu_seqlens_q, cu_seqlens_k=context.cu_seqlens_k,
                                 max_seqlen_q=context.max_seqlen_q, max_seqlen_k=context.max_seqlen_k,
                                 softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if _FLASH_ATTN_SUPPORTED:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
            else:
                o = _sdpa_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                       cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                       softmax_scale=self.scale, causal=True)
        return o
