from typing import Dict, Any, List
import torch
import torch.nn.functional as F

from .config import SelfAdaptiveSFTConfig


def compute_log_probs(model, tokenizer, inputs: List[str], outputs: List[str], device: str = "cuda"):
    """
    计算 log p(y|x) 对每个样本，返回所有token的log概率之和。
    
    使用更健壮的方法：先对input tokenize，然后对output逐个token计算条件概率。
    
    Args:
        model: 语言模型
        tokenizer: tokenizer
        inputs: 输入文本列表
        outputs: 输出文本列表
        device: 设备
        
    Returns:
        torch.Tensor: shape [batch_size]，每个样本的log概率之和
    """
    batch_size = len(inputs)
    log_probs_list = []
    
    for i in range(batch_size):
        # Tokenize input
        input_text = inputs[i]
        output_text = outputs[i]
        
        # 对input进行tokenization
        input_tokenized = tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )
        
        # 对output进行tokenization（单独tokenize，不包含input）
        output_tokenized = tokenizer(
            output_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        )
        
        # 构造完整序列：input + output
        input_ids = input_tokenized["input_ids"].to(device)  # [1, seq_len]
        output_ids = output_tokenized["input_ids"].to(device)  # [1, seq_len]
        
        # 拼接input和output
        full_input_ids = torch.cat([input_ids, output_ids], dim=1)
        attention_mask = torch.ones_like(full_input_ids)
        
        # 计算logits（对完整序列）
        with torch.no_grad():
            outputs_model = model(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs_model.logits[0]  # [seq_len, vocab_size]
        
        # 计算output部分的log概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取output tokens的log概率
        # logits[i] 预测的是位置 i+1 的token
        input_len = input_ids.shape[1]
        output_log_prob_sum = 0.0
        
        for j, token_id in enumerate(output_ids[0]):
            # 对于output中的第j个token，它对应的logits位置是 input_len + j - 1
            # 因为logits[i]预测的是input_ids[i+1]
            pos = input_len + j - 1
            if pos >= 0 and pos < len(log_probs):
                output_log_prob_sum += log_probs[pos, token_id].item()
        
        log_probs_list.append(output_log_prob_sum)
    
    return torch.tensor(log_probs_list, device=device, dtype=torch.float32)


def compute_kl_penalty(anchor_model, anchor_tokenizer, inputs: List[str], 
                       original_outputs: List[str], rewritten_outputs: List[str],
                       device: str = "cuda"):
    """
    计算KL散度惩罚项：KL(p_anchor(y'|x) || p_anchor(y|x))
    
    这里简化为：基于anchor模型，计算改写前后output分布的KL散度。
    为了简化，我们使用输出log概率的差异作为近似。
    """
    # 计算anchor模型对原始output和改写output的log概率
    log_prob_original = compute_log_probs(anchor_model, anchor_tokenizer, inputs, original_outputs, device)
    log_prob_rewritten = compute_log_probs(anchor_model, anchor_tokenizer, inputs, rewritten_outputs, device)
    
    # 简化版的KL惩罚：使用log概率差的绝对值作为惩罚
    # 这鼓励改写不要偏离原始分布太远
    kl_penalty = -torch.abs(log_prob_rewritten - log_prob_original)
    
    return kl_penalty


def compute_rewriting_reward(
    main_model: Any,
    main_tokenizer: Any,
    anchor_model: Any,
    anchor_tokenizer: Any,
    batch: List[Dict[str, Any]],
    config: SelfAdaptiveSFTConfig,
) -> torch.Tensor:
    """
    计算单个 batch 上的 reward。

    batch 约定格式：
        {
            "input": str,
            "original_output": str,
            "rewritten_output": str,
        }
        
    返回: torch.Tensor [batch_size]，每个样本的reward
    """
    inputs = [item["input"] for item in batch]
    original_outputs = [item["original_output"] for item in batch]
    rewritten_outputs = [item["rewritten_output"] for item in batch]
    
    # 使用主模型计算log概率
    log_prob_original = compute_log_probs(
        main_model, main_tokenizer, inputs, original_outputs, config.device
    )
    log_prob_rewritten = compute_log_probs(
        main_model, main_tokenizer, inputs, rewritten_outputs, config.device
    )
    
    # 计算reward
    if config.reward_use_delta:
        # reward = log_prob(y'|x) - log_prob(y|x)
        reward = log_prob_rewritten - log_prob_original
    else:
        # reward = log_prob(y'|x)
        reward = log_prob_rewritten
    
    # 添加KL惩罚项
    if config.kl_penalty_coef > 0 and anchor_model is not None:
        kl_penalty = compute_kl_penalty(
            anchor_model, anchor_tokenizer,
            inputs, original_outputs, rewritten_outputs, config.device
        )
        reward = reward + config.kl_penalty_coef * kl_penalty
    
    return reward


