import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SiLU (Sigmoid Linear Unit) 激活函数与逐元素乘法的结合。
    常用于 SwiGLU 激活函数，它是现代 LLM（如 Llama, Qwen）中 MLP 层的标准配置。
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将输入张量在最后一个维度切分为两部分，对第一部分应用 SiLU，然后与第二部分相乘。
        
        参数:
            x: 输入张量，其最后一个维度的长度应为 2 的倍数。
        """
        # 在最后一个维度切分为 x 和 y
        x, y = x.chunk(2, -1)
        # SwiGLU 的核心计算：SiLU(x) * y
        return F.silu(x) * y

