"""
基于主模型logprob的reward函数，用于open-r1的GRPO训练。
"""

from typing import List, Dict, Any
import torch
import torch.nn.functional as F


def compute_log_prob(model, tokenizer, prompt: str, completion: str, device: str = "cuda"):
    """
    计算 log p(completion|prompt)。
    
    Args:
        model: 语言模型
        tokenizer: tokenizer
        prompt: 输入提示
        completion: 完成文本
        device: 设备（如果模型使用device_map，这里应该是模型所在的设备）
        
    Returns:
        float: completion部分的log概率之和
    """
    # 获取模型所在的设备
    # 如果模型使用device_map="auto"，需要找到第一个参数所在的设备
    if hasattr(model, "device"):
        model_device = model.device
    elif hasattr(model, "hf_device_map") and model.hf_device_map:
        # 模型分布在多张卡上，找到第一个设备
        first_device = list(model.hf_device_map.values())[0]
        if isinstance(first_device, (list, tuple)):
            model_device = first_device[0]
        else:
            model_device = first_device
    else:
        # 尝试从第一个参数获取设备
        try:
            model_device = next(model.parameters()).device
        except:
            model_device = device
    
    # 构造完整文本
    full_text = prompt + completion
    
    # Tokenize
    prompt_tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    ).to(model_device)
    
    full_tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=512,  # 减小到512以节省内存
    ).to(model_device)
    
    # 计算logits（模型应该已经在eval模式）
    with torch.no_grad():
        outputs = model(**full_tokenized)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        # 立即释放不需要的中间结果
        del outputs
    
    # 计算log概率
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 找到completion部分的tokens
    prompt_len = prompt_tokenized["input_ids"].shape[1]
    completion_tokenized = tokenizer(
        completion,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    ).to(model_device)
    completion_ids = completion_tokenized["input_ids"][0]
    
    # 计算completion部分的log概率
    completion_log_prob_sum = 0.0
    for j, token_id in enumerate(completion_ids):
        # logits[i] 预测的是位置 i+1 的token
        pos = prompt_len + j - 1
        if pos >= 0 and pos < len(log_probs):
            completion_log_prob_sum += log_probs[pos, token_id].item()
    
    return completion_log_prob_sum


def create_logprob_reward_func(
    main_model: Any,
    main_tokenizer: Any,
    anchor_model: Any = None,
    anchor_tokenizer: Any = None,
    reward_use_delta: bool = True,
    kl_penalty_coef: float = 0.0,
    device: str = "cuda",
):
    """
    创建基于logprob的reward函数，兼容open-r1的GRPO。
    
    Args:
        main_model: 主模型（用于计算reward）
        main_tokenizer: 主模型的tokenizer
        anchor_model: 锚点模型（用于KL惩罚，可选）
        anchor_tokenizer: 锚点模型的tokenizer
        reward_use_delta: 是否使用delta reward（logprob(y') - logprob(y)）
        kl_penalty_coef: KL惩罚系数
        device: 设备
        
    Returns:
        reward函数，签名: (completions, solution, **kwargs) -> List[float]
    """
    
    def logprob_reward_func(completions: List[List[Dict[str, str]]], solution: List[str], **kwargs) -> List[float]:
        """
        Reward函数，兼容open-r1的GRPO接口。
        
        Args:
            completions: List of completions, 每个completion是 [{"content": str}]
            solution: List of ground truth solutions (原始output)
            **kwargs: 其他参数，通常包含prompts字段
            
        Returns:
            List[float]: 每个completion的reward
        """
        rewards = []
        
        # 从kwargs获取prompts（GRPO会传递）
        prompts = kwargs.get("prompts", [])
        
        # 如果没有prompts，尝试从其他字段获取
        if not prompts:
            # GRPO可能会将prompt作为字符串列表传递
            if "prompt" in kwargs:
                prompt = kwargs["prompt"]
                if isinstance(prompt, list):
                    prompts = [p["content"] if isinstance(p, dict) else str(p) for p in prompt]
                else:
                    prompts = [str(prompt)] * len(completions)
        
        for i, (completion_list, original_output) in enumerate(zip(completions, solution)):
            # 提取completion内容
            rewritten_output = completion_list[0]["content"] if completion_list else ""
            
            # 获取对应的prompt
            if i < len(prompts):
                prompt_text = prompts[i]
                # 如果是list（conversation格式），提取用户消息
                if isinstance(prompt_text, list):
                    # 找到最后一个user消息
                    user_messages = [msg["content"] for msg in prompt_text if msg.get("role") == "user"]
                    prompt_text = user_messages[-1] if user_messages else str(prompt_text)
                elif isinstance(prompt_text, dict):
                    prompt_text = prompt_text.get("content", str(prompt_text))
            else:
                # 如果没有prompt，使用空字符串（模型会从completion中提取input）
                prompt_text = ""
            
            # 从prompt中提取原始input（如果有的话）
            # prompt格式通常是: "Rewrite...Input: {input}\nOriginal Output: {output}\nRewritten Output:"
            # 我们需要提取input部分用于计算logprob
            original_input = ""
            if "Input:" in prompt_text and "Original Output:" in prompt_text:
                try:
                    parts = prompt_text.split("Input:")[1].split("Original Output:")[0].strip()
                    original_input = parts
                except:
                    pass
            
            # 构造用于计算logprob的prompt（只包含input部分）
            # 注意：我们需要计算的是 p(rewritten_output | original_input) vs p(original_output | original_input)
            if not original_input:
                # 如果没有提取到，使用原始prompt（去掉"Rewritten Output:"部分）
                prompt_for_logprob = prompt_text.split("Rewritten Output:")[0] if "Rewritten Output:" in prompt_text else prompt_text
            else:
                # 使用提取的原始input
                prompt_for_logprob = original_input
            
            # 计算主模型的log概率
            log_prob_rewritten = compute_log_prob(
                main_model, main_tokenizer, prompt_for_logprob, rewritten_output, device
            )
            log_prob_original = compute_log_prob(
                main_model, main_tokenizer, prompt_for_logprob, original_output, device
            )
            
            # 计算base reward
            if reward_use_delta:
                reward = log_prob_rewritten - log_prob_original
            else:
                reward = log_prob_rewritten
            
            # 添加KL惩罚
            if kl_penalty_coef > 0 and anchor_model is not None:
                log_prob_rewritten_anchor = compute_log_prob(
                    anchor_model, anchor_tokenizer, prompt_for_logprob, rewritten_output, device
                )
                log_prob_original_anchor = compute_log_prob(
                    anchor_model, anchor_tokenizer, prompt_for_logprob, original_output, device
                )
                
                # KL惩罚：惩罚偏离anchor模型分布
                kl_penalty = -abs(log_prob_rewritten_anchor - log_prob_original_anchor)
                reward = reward + kl_penalty_coef * kl_penalty
            
            rewards.append(float(reward))
        
        return rewards
    
    return logprob_reward_func

