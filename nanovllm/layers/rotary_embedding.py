from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
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
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    if rope_scaling is not None:
        # handle transformers RoPEScalingConfig or dict
        rope_type = getattr(rope_scaling, "rope_type", None) or (rope_scaling.get("rope_type") if isinstance(rope_scaling, dict) else None)
        if not rope_type:
            rope_type = getattr(rope_scaling, "type", None) or (rope_scaling.get("type") if isinstance(rope_scaling, dict) else None)
        
        if not rope_scaling or rope_type == "default":
            rope_scaling = None
        else:
            if isinstance(rope_scaling, dict):
                rope_scaling = tuple(sorted(rope_scaling.items()))
            else:
                # If it's some other object, try to convert to dict first
                try:
                    d = dict(rope_scaling)
                    rope_scaling = tuple(sorted(d.items()))
                except:
                    rope_scaling = str(rope_scaling) # Fallback to string for hashability
    return _get_rope(head_size, rotary_dim, max_position, base, rope_scaling)
