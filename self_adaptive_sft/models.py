import os
from typing import Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_model_path(path: str) -> str:
    """
    规范化模型路径：
    - 如果是本地相对路径（以 ./ 或 ../ 开头），转换为绝对路径
    - 如果是本地绝对路径，保持原样
    - 如果是 HuggingFace 模型 ID，保持原样
    
    这样可以避免 HuggingFace 将相对路径误判为 repo id。
    """
    # 如果是绝对路径，直接返回
    if os.path.isabs(path):
        return path
    
    # 如果是相对路径（以 ./ 或 ../ 开头），转换为绝对路径
    if path.startswith("./") or path.startswith("../"):
        return os.path.abspath(path)
    
    # 如果包含 /，可能是 HuggingFace 模型 ID 或本地路径
    # 检查是否是本地存在的路径
    if "/" in path:
        if os.path.exists(path) or os.path.isdir(path):
            # 是本地路径，转换为绝对路径
            return os.path.abspath(path)
        # 否则是 HuggingFace 模型 ID，保持原样
        return path
    
    # 其他情况（不包含 /），可能是本地文件名，检查是否存在
    if os.path.exists(path) or os.path.isdir(path):
        return os.path.abspath(path)
    
    # 可能是 HuggingFace 模型 ID（不包含 /），保持原样
    return path


def load_main_model(path: str, device: str = "cuda") -> Tuple[Any, Any]:
    """
    加载主模型 M_t 和对应的tokenizer。
    
    返回: (model, tokenizer)
    """
    # 规范化路径（将相对路径转换为绝对路径，但保持 HuggingFace 模型 ID 不变）
    normalized_path = normalize_model_path(path)
    print(f"正在加载主模型: {normalized_path}")
    tokenizer = AutoTokenizer.from_pretrained(normalized_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 根据device参数决定加载方式
    if device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            normalized_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            normalized_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
    
    model.eval()  # 用于推理和reward计算
    return model, tokenizer


def load_rewriter_model(path: str, device: str = "cuda") -> Tuple[Any, Any]:
    """
    加载重写模型 m_t 和对应的tokenizer。
    
    返回: (model, tokenizer)
    """
    # 规范化路径（将相对路径转换为绝对路径，但保持 HuggingFace 模型 ID 不变）
    normalized_path = normalize_model_path(path)
    print(f"正在加载重写模型: {normalized_path}")
    tokenizer = AutoTokenizer.from_pretrained(normalized_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 根据device参数决定加载方式
    if device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            normalized_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            normalized_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
    
    model.train()  # 用于训练
    return model, tokenizer


def save_main_model(model: Any, tokenizer: Any, path: str) -> None:
    """
    保存主模型和tokenizer。
    """
    os.makedirs(path, exist_ok=True)
    print(f"正在保存主模型到: {path}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def save_rewriter_model(model: Any, tokenizer: Any, path: str) -> None:
    """
    保存重写模型和tokenizer。
    """
    os.makedirs(path, exist_ok=True)
    print(f"正在保存重写模型到: {path}")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


