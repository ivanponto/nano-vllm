import torch
from torch import nn


class Sampler(nn.Module):
    """
    采样层：将模型的 Logits 转换为具体的 Token ID。
    支持温度缩放（Temperature Scaling）和 Gumbel-max 采样。
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        前向传播：执行采样逻辑。
        
        参数:
            logits: 模型输出的未归一化对数概率，形状为 [batch_size, vocab_size]。
            temperatures: 批次中每个请求的温度值，形状为 [batch_size]。
        """
        # 应用温度缩放：logits = logits / temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        # 计算 Softmax 概率
        probs = torch.softmax(logits, dim=-1)
        
        # 使用 Gumbel-max 技巧进行采样：
        # 1. 生成服从指数分布的随机噪声。
        # 2. 将概率除以这些噪声（等效于在 log 空间加 Gumbel 噪声）。
        # 3. 取最大值索引作为采样到的 Token。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        
        return sample_tokens

