"""
基于logprob的数据筛选模块。

用于筛选出主模型不align的数据（logprob较低），只对这些数据进行GRPO训练和改写。
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def compute_batch_log_prob(
    model: Any,
    tokenizer: Any,
    inputs: List[str],
    outputs: List[str],
    device: str = "cuda",
    batch_size: int = 32,  # 默认batch_size（7B模型建议16-32，3B模型可以用64-128）
) -> List[float]:
    """
    批量计算 log p(output|input) 的token概率之和。
    
    使用真正的批量计算，一次处理一个batch的数据，大幅提升速度。
    
    Args:
        model: 主模型
        tokenizer: tokenizer
        inputs: 输入文本列表
        outputs: 输出文本列表
        device: 设备
        batch_size: 批次大小（可以设置较大，如128-256）
        
    Returns:
        List[float]: 每个样本的log概率之和
    """
    model.eval()
    log_probs = []
    
    # 获取模型所在的设备
    if hasattr(model, "device"):
        model_device = model.device
    elif hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = list(model.hf_device_map.values())[0]
        if isinstance(first_device, (list, tuple)):
            model_device = first_device[0]
        else:
            model_device = first_device
    else:
        try:
            model_device = next(model.parameters()).device
        except:
            model_device = device
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="计算logprob"):
            batch_inputs = inputs[i:i+batch_size]
            batch_outputs = outputs[i:i+batch_size]
            
            # 构造完整文本列表
            batch_full_texts = [inp + out for inp, out in zip(batch_inputs, batch_outputs)]
            
            # 批量tokenize（使用padding）
            full_tokenized = tokenizer(
                batch_full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            ).to(model_device)
            
            # 批量tokenize inputs（用于确定output的起始位置）
            input_tokenized = tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            ).to(model_device)
            
            # 批量tokenize outputs（用于提取output tokens）
            output_tokenized = tokenizer(
                batch_outputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            ).to(model_device)
            
            # 批量计算logits
            model_outputs = model(**full_tokenized)
            logits = model_outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # 计算log概率
            log_probs_tensor = F.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
            
            # 获取每个样本的prompt长度和output token ids
            input_lengths = input_tokenized["attention_mask"].sum(dim=1)  # [batch_size]
            output_ids = output_tokenized["input_ids"]  # [batch_size, output_seq_len]
            output_attention_mask = output_tokenized["attention_mask"]  # [batch_size, output_seq_len]
            
            # 批量计算每个样本的output log概率之和
            batch_log_probs = []
            for batch_idx in range(len(batch_inputs)):
                prompt_len = input_lengths[batch_idx].item()
                output_seq = output_ids[batch_idx]  # [output_seq_len]
                output_mask = output_attention_mask[batch_idx]  # [output_seq_len]
                
                # 计算output部分的log概率（向量化操作）
                # 找到所有有效的output token位置
                valid_indices = torch.nonzero(output_mask, as_tuple=False).squeeze(-1)
                
                if len(valid_indices) == 0:
                    batch_log_probs.append(0.0)
                    continue
                
                # 计算每个output token在logits中的位置
                # logits[batch_idx, pos] 预测的是位置 pos+1 的token
                # output的第j个token对应logits的位置是 prompt_len + j - 1
                positions = prompt_len - 1 + valid_indices
                
                # 过滤掉超出范围的位置
                valid_positions = (positions >= 0) & (positions < log_probs_tensor.shape[1])
                if valid_positions.sum() == 0:
                    batch_log_probs.append(0.0)
                    continue
                
                positions = positions[valid_positions]
                token_ids = output_seq[valid_indices[valid_positions]]
                
                # 使用gather批量提取log概率
                # log_probs_tensor[batch_idx, positions, token_ids]
                selected_log_probs = log_probs_tensor[batch_idx, positions, token_ids]
                output_log_prob_sum = selected_log_probs.sum().item()
                
                batch_log_probs.append(output_log_prob_sum)
            
            log_probs.extend(batch_log_probs)
            
            # 显式释放中间变量，避免显存累积（7B模型需要）
            del model_outputs, logits, log_probs_tensor
            del full_tokenized, input_tokenized, output_tokenized
            del input_lengths, output_ids, output_attention_mask
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理显存缓存
    
    return log_probs


def filter_dataset_by_logprob(
    dataset: List[Dict[str, str]],
    main_model: Any,
    main_tokenizer: Any,
    device: str = "cuda",
    filter_ratio: float = 0.2,  # 筛选出最低的20%（1/5）
    batch_size: int = 32,  # 批量计算logprob（7B模型建议16-32，3B模型可以用64-128）
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    根据主模型的logprob筛选数据集。
    
    只筛选出logprob较低的数据（模型不align的数据）进行改写训练。
    
    Args:
        dataset: 原始数据集 [{"input": str, "output": str}]
        main_model: 主模型
        main_tokenizer: 主模型的tokenizer
        device: 设备
        filter_ratio: 筛选比例（0.2表示筛选出最低的20%，即1/5）
        batch_size: 批量计算logprob的批次大小
        
    Returns:
        Tuple[List[Dict], List[Dict]]: (需要改写的数据, 不需要改写的数据)
    """
    print(f"\n开始计算logprob并筛选数据（筛选比例: {filter_ratio*100:.1f}%）...")
    
    # 提取inputs和outputs
    inputs = [item["input"] for item in dataset]
    outputs = [item["output"] for item in dataset]
    
    # 批量计算logprob
    log_probs = compute_batch_log_prob(
        model=main_model,
        tokenizer=main_tokenizer,
        inputs=inputs,
        outputs=outputs,
        device=device,
        batch_size=batch_size,
    )
    
    # 计算阈值（选择最低的filter_ratio比例）
    log_probs_array = np.array(log_probs)
    threshold = np.percentile(log_probs_array, filter_ratio * 100)
    
    print(f"Logprob统计:")
    print(f"  平均值: {np.mean(log_probs_array):.2f}")
    print(f"  中位数: {np.median(log_probs_array):.2f}")
    print(f"  最小值: {np.min(log_probs_array):.2f}")
    print(f"  最大值: {np.max(log_probs_array):.2f}")
    print(f"  阈值（{filter_ratio*100:.1f}分位数）: {threshold:.2f}")
    
    # 筛选数据
    need_rewrite = []
    no_need_rewrite = []
    
    for item, log_prob in zip(dataset, log_probs):
        if log_prob <= threshold:
            need_rewrite.append(item)
        else:
            no_need_rewrite.append(item)
    
    print(f"\n数据筛选结果:")
    print(f"  总数据量: {len(dataset)}")
    print(f"  需要改写: {len(need_rewrite)} ({len(need_rewrite)/len(dataset)*100:.1f}%)")
    print(f"  不需要改写: {len(no_need_rewrite)} ({len(no_need_rewrite)/len(dataset)*100:.1f}%)")
    
    return need_rewrite, no_need_rewrite

