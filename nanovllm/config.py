import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    LLM 推理引擎的全局配置类。
    包含了模型路径、推理性能参数、内存管理设置等。
    """
    model: str                      # 模型权重和配置文件的本地路径
    max_num_batched_tokens: int = 16384 # 单次批处理（Batch）允许的最大 Token 总数
    max_num_seqs: int = 512          # 允许的最大并行序列（Request）数量
    max_model_len: int = 4096        # 允许的最大序列长度（超过部分会被截断或拒绝）
    gpu_memory_utilization: float = 0.9 # 显存利用率。预留 10% 给 PyTorch 算子和临时开销
    tensor_parallel_size: int = 1    # 张量并行度。1 表示单卡，>1 表示多卡并行
    enforce_eager: bool = False      # 是否强制使用 Eager 模式。False 则启用 CUDA Graph 加速
    hf_config: AutoConfig | None = None # 从 Hugging Face 自动加载的模型原始配置对象
    eos: int = -1                   # 停止符 (End Of Sentence) 的 Token ID
    kvcache_block_size: int = 256    # PagedAttention 的物理块大小（Token 数量）
    num_kvcache_blocks: int = -1     # 自动计算得到的物理块总数（初始化后更新）

    def __post_init__(self):
        """初始化后的参数验证和补充。"""
        # 1. 验证模型路径是否存在
        assert os.path.isdir(self.model), f"Model directory {self.model} not found."
        # 2. 块大小目前限制为 256 以匹配某些内核实现
        assert self.kvcache_block_size % 256 == 0
        # 3. 验证并行度（通常限制在 1-8 之间）
        assert 1 <= self.tensor_parallel_size <= 8
        
        # 4. 从本地路径加载 Hugging Face 的 config.json
        self.hf_config = AutoConfig.from_pretrained(self.model)
        
        # 5. 更新最大模型长度（取配置和实际支持的最小值）
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
        # 6. 确保 Batch 限制足以容纳至少一个完整序列
        assert self.max_num_batched_tokens >= self.max_model_len
