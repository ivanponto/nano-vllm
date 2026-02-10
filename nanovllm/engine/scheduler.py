from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    请求调度器。
    负责管理等待队列和运行队列，决定哪些请求进入推理，并处理抢占逻辑。
    """
    def __init__(self, config: Config):
        """
        初始化调度器。
        
        Args:
            config: 全局配置对象。
        """
        self.max_num_seqs = config.max_num_seqs               # 最大并行序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens # 单批次最大 Token 数
        self.eos = config.eos                                 # 停止符 ID
        # 初始化块管理器，负责物理 KV Cache 的分配
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()               # 等待调度的序列队列
        self.running: deque[Sequence] = deque()               # 正在运行的序列队列

    def is_finished(self):
        """判断是否所有请求都已处理完毕。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新请求添加到等待队列。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度核心逻辑。
        优先进行 Prefill（预填充），如果没有 Prefill 则进行 Decode（解码）。
        
        Returns:
            tuple: (被调度的序列列表, 是否为 Prefill 阶段)
        """
        # --- 1. Prefill 阶段调度 ---
        # 只要等待队列中有请求，且未达到并发上限，就尝试调度 Prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 检查：1) 加上当前序列后总 Token 数是否超标 2) 是否有足够的 KV Cache 块
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            
            num_seqs += 1
            # 分配 KV Cache 物理块
            self.block_manager.allocate(seq)
            # 计算本次实际需要计算的 Token 数（排除 Prefix Caching 命中的部分）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            
        if scheduled_seqs:
            return scheduled_seqs, True

        # --- 2. Decode 阶段调度 ---
        # 如果没有 Prefill 请求，则调度运行队列中的所有请求进行 Decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查是否有空间追加新生成的 KV Cache 块
            while not self.block_manager.can_append(seq):
                # 内存不足，需要抢占（Preempt）已有的请求
                if self.running:
                    # 抢占运行队列末尾的请求（通常是最新加入的）
                    self.preempt(self.running.pop())
                else:
                    # 如果只剩自己，也要被抢占（等待后续资源）
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                # 标记该序列可以追加 KV Cache
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                
        assert scheduled_seqs
        # 将本次调度的序列重新放回运行队列头部，保持顺序
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占逻辑。
        当 KV Cache 内存不足时，将正在运行的序列暂停并移回等待队列。
        采用 Recomputation 策略：释放其所有物理块，下次调度时重新 Prefill。
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        推理步后的后处理逻辑。
        更新序列内容，检查是否结束，并释放已完成请求的资源。
        
        Args:
            seqs: 本次参与推理的序列列表。
            token_ids: 模型生成的新 Token ID 列表。
        """
        for seq, token_id in zip(seqs, token_ids):
            # 将新 Token 添加到序列中
            seq.append_token(token_id)
            # 检查结束条件：1) 生成了 EOS 2) 达到了最大 Token 限制
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # 释放 KV Cache 资源
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
