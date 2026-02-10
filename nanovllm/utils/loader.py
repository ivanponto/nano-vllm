import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    默认权重加载器。
    直接将加载的张量拷贝到模型参数中。
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    模型权重加载工具。
    支持从指定路径加载 safetensors 格式的权重，并处理参数切分（如张量并行下的权重加载）。
    
    Args:
        model: 模型实例。
        path: 包含 .safetensors 文件的目录路径。
    """
    # 获取模型定义的打包模块映射（用于处理 QKV 合并权重等特殊情况）
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 遍历目录下所有的 safetensors 文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查当前权重是否属于打包模块（如 QKV 权重需要拆分加载）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 获取目标参数名和对应的分片 ID
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 调用参数自带的自定义加载器进行处理
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重加载逻辑
                    try:
                        param = model.get_parameter(weight_name)
                        # 优先使用参数自定义的加载器，否则使用默认加载器
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    except AttributeError:
                        # 忽略模型中不存在的权重（如某些框架无关的元数据）
                        continue
