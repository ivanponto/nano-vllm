from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    模型推理时的采样参数配置。
    """
    temperature: float = 1.0  # 采样温度。1.0 表示原始概率分布，值越高生成越随机，值越低生成越确定。
    max_tokens: int = 64      # 本次请求允许生成的新 Token 最大数量。
    ignore_eos: bool = False  # 是否忽略 EOS (End Of Sentence) 停止符。如果为 True，则会一直生成直到达到 max_tokens。

    def __post_init__(self):
        """参数校验。"""
        # 为了实现简单，本项目目前要求 temperature 必须大于 0。
        # greedy sampling（贪婪搜索）通常通过设置极低温度来实现。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
