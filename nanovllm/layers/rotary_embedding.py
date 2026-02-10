from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    将旋转位置嵌入 (RoPE) 应用于输入的查询 (Query) 或键 (Key) 张量。
    
    RoPE 通过将 D 维向量视为 D/2 个复数，并对每个复数进行旋转来注入位置信息。
    实现逻辑：
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
    其中 [x1, x2] 是输入向量的切片。
    """
    # 将输入切分为两半
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 应用旋转变换
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 合并结果并恢复原始数据类型
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置嵌入 (Rotary Positional Embedding, RoPE) 层。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # 目前假设 rotary_dim 等于 head_size，即对整个 head dim 进行旋转
        assert rotary_dim == head_size
        
        # 计算频率因子 inv_freq: 1 / (base^(2i/d))
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 生成时间步（位置）索引
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 计算外积得到 freqs: t * inv_freq
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 预计算 cos 和 sin 并缓存
        cos = freqs.cos()
        sin = freqs.sin()
        # 合并 cos 和 sin 方便一次性索引，形状为 [max_pos, 1, head_size]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：为 Q 和 K 注入位置信息。
        
        参数:
            positions: 批次中每个 token 的位置索引，形状为 [num_tokens]。
            query: 查询张量。
            key: 键张量。
        """
        # 根据位置索引从缓存中获取 cos 和 sin
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        # 应用旋转变换
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def _get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: tuple | None = None,
):
    """带缓存的 RoPE 对象创建函数"""
    assert rope_scaling is None # 当前简化实现暂不支持 RoPE 缩放
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取 RoPE 层的工厂函数，处理不同的 RoPE 缩放配置。
    """
    if rope_scaling is not None:
        # 处理 transformers 库中的 RoPEScalingConfig 或普通字典
        rope_type = getattr(rope_scaling, "rope_type", None) or (rope_scaling.get("rope_type") if isinstance(rope_scaling, dict) else None)
        if not rope_type:
            rope_type = getattr(rope_scaling, "type", None) or (rope_scaling.get("type") if isinstance(rope_scaling, dict) else None)
        
        if not rope_scaling or rope_type == "default":
            rope_scaling = None
        else:
            # 将字典转换为元组以便 lru_cache 进行哈希处理
            if isinstance(rope_scaling, dict):
                rope_scaling = tuple(sorted(rope_scaling.items()))
            else:
                try:
                    d = dict(rope_scaling)
                    rope_scaling = tuple(sorted(d.items()))
                except:
                    rope_scaling = str(rope_scaling) # 回退为字符串
    return _get_rope(head_size, rotary_dim, max_position, base, rope_scaling)

