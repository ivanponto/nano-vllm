from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    Sequence 的状态枚举类。
    """
    WAITING = auto()   # 等待调度中
    RUNNING = auto()   # 正在推理中
    FINISHED = auto()  # 生成已结束


class Sequence:
    """
    代表一个推理请求（序列）。
    维护序列的 Token ID 列表、状态、采样参数以及 PagedAttention 相关的块信息。
    """
    block_size = 256  # 默认块大小
    counter = count()  # 全局序列 ID 计数器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化序列。
        
        Args:
            token_ids: 初始输入的 Token ID 列表（Prompt）。
            sampling_params: 采样参数（如温度、最大生成长度等）。
        """
        self.seq_id = next(Sequence.counter)  # 唯一序列 ID
        self.status = SequenceStatus.WAITING  # 初始状态为等待
        self.token_ids = copy(token_ids)      # 完整的 Token ID 列表
        self.last_token = token_ids[-1]       # 最近生成（或输入）的一个 Token
        self.num_tokens = len(self.token_ids) # 当前总 Token 数
        self.num_prompt_tokens = len(token_ids) # Prompt 的 Token 数
        self.num_cached_tokens = 0            # 已被缓存（Prefix Caching）的 Token 数
        self.block_table = []                 # 映射到物理 KV Cache 块的逻辑表（Block IDs）
        self.temperature = sampling_params.temperature # 采样温度
        self.max_tokens = sampling_params.max_tokens   # 最大生成 Token 数
        self.ignore_eos = sampling_params.ignore_eos   # 是否忽略 EOS 停止符

    def __len__(self):
        """返回当前序列的总长度。"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持通过索引访问 Token ID。"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """判断序列是否已完成生成。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """计算已生成的 Token 数量。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取 Prompt 部分的 Token ID。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取生成部分的 Token ID。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """计算完全被缓存的块数量。"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """计算该序列当前总共需要的逻辑块数量。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """计算最后一个块中包含的 Token 数量。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个逻辑块包含的 Token ID。
        
        Args:
            i: 逻辑块索引。
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        向序列添加一个新生成的 Token。
        
        Args:
            token_id: 新生成的 Token ID。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        自定义序列化状态。
        在多进程通信时（如发送给 ModelRunner），减少不必要的数据传输。
        如果是生成阶段，只传输最后一个 token 以节省带宽。
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """反序列化恢复状态。"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
