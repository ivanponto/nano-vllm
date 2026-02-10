import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型执行器。
    负责 GPU 内存管理（KV Cache 分配）、模型加载、CUDA Graph 捕获以及实际的推理前向传播。
    支持张量并行（Tensor Parallelism），通过多进程和 NCCL 进行通信。
    """
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化 ModelRunner。
        
        Args:
            config: 配置对象。
            rank: 当前进程的 GPU 排名（ID）。
            event: 用于进程间同步的事件对象。
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 1. 初始化分布式环境 (NCCL)
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # 2. 设置模型默认精度和设备
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 3. 初始化并加载模型权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # 4. 预热模型（初始化各种算子和显存空间）
        self.warmup_model()
        # 5. 分配物理 KV Cache 显存
        self.allocate_kv_cache()
        # 6. 如果未禁用，捕获 CUDA Graph 以加速 Decode 阶段
        if not self.enforce_eager:
            self.capture_cudagraph()
            
        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 7. 如果是多 GPU 环境，初始化共享内存进行进程间通信
        if self.world_size > 1:
            if rank == 0:
                # Rank 0 创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                # 其他 Rank 等待创建完成后挂载
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 进入循环监听 Rank 0 的指令
                self.loop()

    def exit(self):
        """退出清理资源。"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        子进程循环。
        不断读取共享内存中的指令并执行，直到接收到 'exit'。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """从共享内存读取序列化的方法名和参数。"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait() # 等待 Rank 0 写入完成的通知
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """向共享内存写入指令（仅 Rank 0 调用）。"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set() # 通知所有子进程

    def call(self, method_name, *args):
        """
        统一的方法调用接口。
        如果是 Rank 0，负责将指令分发给其他进程。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """通过一次 dummy 推理预热模型，确保算子初始化完成。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 构造一组 dummy 序列进行 Prefill 预热
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        根据剩余显存自动分配 KV Cache 物理块。
        """
        config = self.config
        hf_config = config.hf_config
        # 获取当前显存状态
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算单个 KV Cache 块所需的字节数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 每个块存储: 2 (K和V) * 层数 * 块大小 * 头数 * 每个头的维度 * 数据类型大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # 计算可以分配的块数：总容量 * 利用率 - (模型已占用的净显存)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        # 分配大块连续显存作为 KV Cache 池 [K/V, 层数, 总块数, 块大小, 头数, 维度]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        # 将分配好的 KV Cache 绑定到模型各层的 Attention 模块中
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """将序列的逻辑块表转换为张量，以便 GPU 算子高效访问。"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 Prefill 阶段的模型输入数据。
        包括打平的 input_ids、位置编码、以及用于 FlashAttention 的偏移信息（cu_seqlens）。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0] # Q 的累积长度（用于处理不同长度的 prompt 拼接）
        cu_seqlens_k = [0] # K 的累积长度
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []  # 映射每个 token 到物理 KV Cache 中的 slot 索引
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            # 仅处理尚未被缓存的部分
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:    # 仅在预热阶段可能为空
                continue
                
            # 计算每个 token 在物理块中的位置
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
                
        # 如果存在 Prefix Caching，需要提供 block_tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
            
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 将元数据设置到全局上下文中，供 Attention 算子调用
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 Decode 阶段的模型输入数据（每次仅推理一个 token）。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token) # 仅输入最后一个 token
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 找到新 token 对应的物理存储位置（最后一个块的最后一位）
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置 Decode 上下文（PagedAttention 需要 block_tables 和 context_lens）
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样所需的温度张量。"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向传播并获取 Logits。
        支持 Eager 模式和 CUDA Graph 模式。
        """
        # Prefill 阶段、显存紧张或 Batch 太大时，使用 Eager 模式
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Decode 阶段优先使用捕获好的 CUDA Graph
            bs = input_ids.size(0)
            context = get_context()
            # 匹配最接近的 Batch Size Graph（减少 Graph 数量）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 将输入数据拷贝到 Graph 的固定输入缓冲区中
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放 CUDA Graph
            graph.replay()
            # 从固定输出缓冲区获取结果
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        一次完整的模型推理循环：数据准备 -> 模型计算 -> 采样。
        
        Returns:
            list[int]: 生成的新 Token ID 列表。
        """
        # 1. 数据准备
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # 2. 执行模型前向传播
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # 3. 采样（仅 Rank 0 负责并返回结果）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        # 4. 重置上下文，清理元数据
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        为 Decode 阶段捕获不同 Batch Size 的 CUDA Graph。
        通过减少 CPU 提交算子的开销来显著提升 Decode 性能。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 定义固定缓冲区（Static Buffers）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 定义要捕获的 Batch Size 梯度
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None # 共享内存池

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 第一遍推理进行预热（分配必要的中间变量显存）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 第二遍进行捕获
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                
            if self.graph_pool is None:
                self.graph_pool = graph.pool() # 捕获第一个 Graph 后锁定显存池
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存固定变量的引用，以便在 run_model 中填充
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
