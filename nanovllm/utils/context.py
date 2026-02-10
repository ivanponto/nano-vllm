from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    推理上下文元数据。
    用于在模型的前向传播过程中传递 PagedAttention 和 FlashAttention 所需的张量。
    这些元数据在每一推理步（Step）都会更新。
    """
    is_prefill: bool = False             # 当前推理步是否为 Prefill 阶段
    cu_seqlens_q: torch.Tensor | None = None # Q 的累积长度（用于 FlashAttention 处理拼接的 Prompt）
    cu_seqlens_k: torch.Tensor | None = None # K 的累积长度
    max_seqlen_q: int = 0                # 本次批次中 Q 的最大长度
    max_seqlen_k: int = 0                # 本次批次中 K 的最大长度
    slot_mapping: torch.Tensor | None = None # 每个 token 在物理 KV Cache 中的 slot 索引 [num_tokens]
    context_lens: torch.Tensor | None = None # 每个序列当前的长度（用于 Decode 阶段的 PagedAttention）
    block_tables: torch.Tensor | None = None # 物理块表映射 [batch_size, max_num_blocks_per_seq]

# 全局上下文单例
_CONTEXT = Context()

def get_context():
    """获取当前全局推理上下文。"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """
    更新全局推理上下文。
    
    Args:
        is_prefill: 是否为预填充。
        cu_seqlens_q: Q 的累积长度。
        cu_seqlens_k: K 的累积长度。
        max_seqlen_q: Q 最大长度。
        max_seqlen_k: K 最大长度。
        slot_mapping: 显存槽位映射。
        context_lens: 序列长度。
        block_tables: 物理块映射表。
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """重置全局推理上下文。"""
    global _CONTEXT
    _CONTEXT = Context()
