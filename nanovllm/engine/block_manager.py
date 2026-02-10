from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    物理 KV Cache 块。
    """
    def __init__(self, block_id):
        self.block_id = block_id  # 物理块的索引 ID
        self.ref_count = 0        # 引用计数。当多个序列共享同一个 Prefix 块时，计数 > 1
        self.hash = -1            # 该块内容的哈希值，用于 Prefix Caching 匹配
        self.token_ids = []       # 该块中存储的 Token ID 列表

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希值和内容。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    物理 KV Cache 块管理器。
    实现基于 PagedAttention 的内存管理和基于 Prefix Caching 的缓存共享。
    """
    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化管理器。
        
        Args:
            num_blocks: 物理块的总数。
            block_size: 每个块能容纳的 Token 数量。
        """
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 所有物理块
        self.hash_to_block_id: dict[int, int] = dict() # 哈希值到物理块 ID 的映射，用于缓存查找
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 空闲块队列
        self.used_block_ids: set[int] = set() # 已使用的块 ID 集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算 Token 序列的哈希值。
        
        Args:
            token_ids: 当前块的 Token ID 列表。
            prefix: 前一个块的哈希值，用于构建哈希链。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """从空闲列表中分配一个指定的物理块。"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """将一个物理块释放回空闲列表。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块来满足序列的需求。"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配物理块。
        利用 Prefix Caching 尝试复用已有的物理块。
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只有完整的块（长度等于 block_size）才参与 Prefix Caching 哈希匹配
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 检查是否缓存命中且内容一致
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
                
            if cache_miss:
                # 缓存未命中：分配一个新的空闲块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：增加引用计数并复用
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            
            # 更新物理块的哈希和内容映射
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有物理块。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否需要并能够为新生成的 Token 分配新块。"""
        # 如果新 Token 导致跨越块边界（len % size == 1），则需要一个新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        在 Decode 步处理 KV Cache 块的追加逻辑。
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        # 情况 1：需要新物理块（当前 Token 是新块的第一个）
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # 情况 2：当前物理块刚填满（最后一个 Token 刚加入）
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            # 计算该满块的哈希值，以便后续请求复用
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 块尚未填满，无需特殊操作
            assert last_block.hash == -1
