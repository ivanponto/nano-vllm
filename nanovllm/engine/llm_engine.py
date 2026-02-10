import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM 推理引擎的核心类，负责协调调度器 (Scheduler)、分发任务给 ModelRunner 以及管理分布式进程。
    """

    def __init__(self, model, **kwargs):
        """
        初始化引擎。
        
        参数:
            model: 模型路径或名称。
            **kwargs: 其他配置参数。
        """
        # 提取 Config 类中定义的字段
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 创建全局配置对象
        config = Config(model, **config_kwargs)
        
        self.ps = [] # 存储辅助进程列表
        self.events = [] # 存储用于同步的事件
        
        # 使用 spawn 启动多进程，这在 CUDA 环境下是必须的
        ctx = mp.get_context("spawn")
        
        # 启动辅助 GPU 进程（针对张量并行）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # ModelRunner 负责实际的模型加载和前向计算
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
            
        # 主进程中的 ModelRunner (Rank 0)
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        # 初始化请求调度器
        self.scheduler = Scheduler(config)
        
        # 注册退出钩子，确保辅助进程能被正确关闭
        atexit.register(self.exit)

    def exit(self):
        """清理资源并关闭辅助进程"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        向引擎添加一个新的推理请求。
        """
        if isinstance(prompt, str):
            # 将文本转换为 Token ID
            prompt = self.tokenizer.encode(prompt)
        # 封装为 Sequence 对象
        seq = Sequence(prompt, sampling_params)
        # 加入调度器的等待队列
        self.scheduler.add(seq)

    def step(self):
        """
        执行一个推理步。包括调度、模型推理和后处理。
        """
        # 1. 调度器决定当前步要处理哪些请求
        seqs, is_prefill = self.scheduler.schedule()
        # 2. 调用 ModelRunner 执行模型前向传播，返回新生成的 Token ID
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 3. 后处理：更新 Sequence 状态（如添加新 token，检查是否结束）
        self.scheduler.postprocess(seqs, token_ids)
        
        # 收集本次步中完成生成的请求
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 计算吞吐量指标：正数表示 prefill token 数，负数表示 decode token 数（负号仅作标识）
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已处理完毕"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        同步接口：批量生成文本。
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
            
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            
        # 批量添加请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
            
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        
        # 循环迭代直到所有请求完成
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            
            if use_tqdm:
                # 更新进度条显示的吞吐量信息
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
                
            # 保存完成的结果
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
                    
        # 按请求添加的顺序排序结果
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 解码为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()
        return outputs

