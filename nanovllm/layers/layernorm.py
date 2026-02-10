import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) 实现。
    相比于传统的 LayerNorm，RMSNorm 移除了均值中心化步骤，只进行缩放，通常能提高计算效率且保持模型性能。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # 防止除零的小增量
        self.eps = eps
        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """纯 RMSNorm 前向传播"""
        orig_dtype = x.dtype
        # 使用 float32 进行中间计算以保证数值稳定性
        x = x.float()
        # 计算均方根: sqrt(mean(x^2))
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 归一化: x / sqrt(var + eps)
        x.mul_(torch.rsqrt(var + self.eps))
        # 转换回原始数据类型并应用可学习的权重
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        残差连接 + RMSNorm 的合并操作。
        在 LLM 架构中（如 Llama, Qwen），通常先将当前层的输出与残差相加，然后再进行归一化。
        """
        orig_dtype = x.dtype
        # x = x + residual
        x = x.float().add_(residual.float())
        # 更新残差项
        residual = x.to(orig_dtype)
        # 执行 RMSNorm
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播入口。
        如果提供了 residual，则执行残差相加 + 归一化，并返回新的 x 和残差。
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

