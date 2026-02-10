import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 模型的注意力模块，集成张量并行和 PagedAttention。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        # 验证总头数是否能被 TP Size 整除
        assert self.total_num_heads % tp_size == 0
        # 当前 GPU 负责的 Q 头数
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        # 当前 GPU 负责的 KV 头数
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 计算 Q 和 KV 的总投影维度
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # 并行化的 QKV 线性投影层
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 并行化的输出投影层
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 旋转位置嵌入
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # 核心注意力计算内核封装
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen3 的某些版本可能会在 Q 和 K 上应用额外的归一化
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 执行 QKV 投影
        qkv = self.qkv_proj(hidden_states)
        # 2. 将结果切分为 Q, K, V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 3. 调整形状以适应注意力计算 [num_tokens, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 4. (可选) 对 Q 和 K 进行归一化
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 5. 应用旋转位置嵌入
        q, k = self.rotary_emb(positions, q, k)
        # 6. 计算注意力（内部处理 Prefill/Decode 和 Paged KV Cache）
        o = self.attn(q, k, v)
        # 7. 输出投影并应用 All-Reduce
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3 的前馈网络 (MLP)，使用 SwiGLU 结构。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # 将 Gate 投影和 Up 投影合并为一个线性层以提高计算效率
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 下投影层
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        # 封装了 SiLU(x) * y 操作的激活层
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # 1. 执行合并的 Gate 和 Up 投影
        gate_up = self.gate_up_proj(x)
        # 2. 应用 SwiGLU 激活函数
        x = self.act_fn(gate_up)
        # 3. 下投影回 hidden_size
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 的单层解码器（Transformer Block）。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 自注意力子层
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # MLP 子层
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 注意力前的归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # MLP 前的归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：
        采用 Pre-Norm 结构，并结合残差路径。
        """
        # 1. 注意力路径
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        
        # 2. MLP 路径
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 核心模型结构。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词嵌入层
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 堆叠 Transformer 解码器层
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终层归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Token 映射到向量
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        # 2. 逐层通过 Decoder Layers
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 3. 最终归一化
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    用于因果语言模型任务的 Qwen3 模型类。
    """
    # 权重加载时的模块映射关系，用于将 HF 格式的权重映射到本地并行层
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # 输出 Logits 的头
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 如果配置要求绑定词嵌入，则共享权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """模型前向传播入口"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """根据隐藏状态计算词表 Logits"""
        return self.lm_head(hidden_states)

