import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词表并行 Embedding 层：将词汇表 (Vocabulary) 切分到不同的 GPU 上。
    当词表非常大时，这有助于平衡显存负载。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 确保词表大小能被 TP Size 整除
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 每个分片负责的词项数量
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前 rank 负责的词汇范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 初始化当前分片的权重
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """根据 TP Rank 加载词表权重分片"""
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播逻辑：
        每个 GPU 只查找属于自己负责范围内的词向量，不属于该范围的词向量置为 0。
        最后通过 All-Reduce 将所有 GPU 的结果相加，得到完整的词向量。
        """
        if self.tp_size > 1:
            # 标记出属于当前 GPU 词表范围的输入索引
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 调整索引为相对于当前分片起始位置的偏移
            x = mask * (x - self.vocab_start_idx)
            
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 将不属于当前 GPU 范围的项清零
            y = mask.unsqueeze(1) * y
            # 聚合所有 GPU 的结果
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行的语言模型输出头 (LM Head)：计算 Logits。
    由于 LM Head 的权重通常与 Embedding 共享（或结构对称），它也采用词表并行。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播：
        在预填充阶段，通常只对每个请求的最后一个 token 计算 logits。
        """
        context = get_context()
        if context.is_prefill:
            # 预填充阶段：只取每个序列的最后一位输出进行预测
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
            
        # 计算本地分片的 logits
        logits = F.linear(x, self.weight)
        
        if self.tp_size > 1:
            # 聚合所有 GPU 计算得到的 logits 分片。
            # 这里使用了 gather 将所有结果收集到 rank 0，以便统一采样。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
            
        return logits

