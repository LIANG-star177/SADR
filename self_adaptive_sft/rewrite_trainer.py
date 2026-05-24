from typing import Any, List, Dict, Optional
import os
import sys
import torch
from datasets import Dataset
from tqdm import tqdm

# 添加open-r1路径
OPENR1_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "open-r1")
if OPENR1_PATH not in sys.path:
    sys.path.insert(0, OPENR1_PATH)

from .config import SelfAdaptiveSFTConfig
from .logprob_reward import create_logprob_reward_func


class RewritingDataset(Dataset):
    """用于GRPO训练的数据集"""
    def __init__(self, dataset: List[Dict[str, str]]):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def collate_fn(self, batch):
        """自定义collate函数"""
        return {
            "input": [item["input"] for item in batch],
            "output": [item["output"] for item in batch],
        }


def generate_rewritten_outputs(
    rewriter_model: Any,
    rewriter_tokenizer: Any,
    inputs: List[str],
    original_outputs: List[str] = None,
    device: str = "cuda",
    max_length: int = 512,
    max_new_tokens: int = 256,  # 优化：减小生成长度以加速推理
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """
    使用重写模型批量生成改写后的输出（真正的batch生成，提高效率）。
    
    Args:
        rewriter_model: 重写模型
        rewriter_tokenizer: tokenizer
        inputs: 输入文本列表
        original_outputs: 原始输出列表（可选，如果提供则基于原始输出改写）
        device: 设备
        max_length: 最大长度
        temperature: 温度
        top_p: top-p采样
        
    Returns:
        改写后的输出列表
    """
    rewriter_model.eval()
    
    # 构造所有prompts
    prompts = []
    for i, input_text in enumerate(inputs):
        if original_outputs is not None and i < len(original_outputs):
            original_output = original_outputs[i]
            prompt = f"""You are a data rewriting assistant. Your task is to rewrite the given output to make it better for training while maintaining correctness.

Input: {input_text}
Original Output: {original_output}

Rewrite the output above. You must ONLY output the rewritten answer directly, without any explanations, prefixes, or meta-commentary like "the rewritten output is" or "explanation is". Just provide the rewritten answer itself.

Rewritten Output:"""
        else:
            prompt = f"""You are a data generation assistant. Your task is to generate a better output for the given input.

Input: {input_text}

Generate the output. You must ONLY output the answer directly, without any explanations, prefixes, or meta-commentary. Just provide the answer itself.

Output:"""
        prompts.append(prompt)
    
    # 批量tokenize（使用padding）
    with torch.no_grad():
        inputs_tokenized = rewriter_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        # 批量生成（真正的batch processing）
        # 优化：减小max_new_tokens，启用use_cache，添加early_stopping
        outputs = rewriter_model.generate(
            **inputs_tokenized,
            max_new_tokens=max_new_tokens,  # 使用参数传入的值
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=rewriter_tokenizer.pad_token_id,
            num_return_sequences=1,
            use_cache=True,  # 启用KV cache加速
            repetition_penalty=1.1,  # 防止重复生成
        )
        
        # 批量解码
        generated_texts = rewriter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 提取改写后的输出部分
        rewritten_outputs = []
        for i, (generated_text, prompt) in enumerate(zip(generated_texts, prompts)):
            # 去掉prompt部分，只保留生成的内容
            if "Rewritten Output:" in generated_text:
                rewritten_output = generated_text.split("Rewritten Output:")[-1].strip()
            elif "Output:" in generated_text:
                rewritten_output = generated_text.split("Output:")[-1].strip()
            else:
                # 如果没有找到分隔符，尝试去掉prompt部分
                # 使用tokenizer来准确计算prompt长度
                prompt_ids = rewriter_tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0]
                output_ids = rewriter_tokenizer.encode(generated_text, add_special_tokens=False, return_tensors="pt")[0]
                if len(output_ids) > len(prompt_ids):
                    # 解码生成的部分
                    generated_ids = output_ids[len(prompt_ids):]
                    rewritten_output = rewriter_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                else:
                    rewritten_output = generated_text.strip()
            
            rewritten_outputs.append(rewritten_output)
    
    return rewritten_outputs


def prepare_dataset_for_grpo(
    dataset: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> Dataset:
    """
    将数据集转换为open-r1 GRPO需要的格式。
    
    Args:
        dataset: 原始数据集 [{"input": str, "output": str}]
        system_prompt: 系统提示（可选）
        
    Returns:
        Dataset: open-r1格式的数据集，包含"prompt"和"solution"字段
    """
    def make_conversation(example):
        """将数据转换为conversation格式"""
        prompt = []
        
        if system_prompt:
            prompt.append({"role": "system", "content": system_prompt})
        
        # 构造用户输入：要求改写output
        user_content = f"""You are a data rewriting assistant. Your task is to rewrite the given output to make it better for training while maintaining correctness.

Input: {example['input']}
Original Output: {example['output']}

Rewrite the output above. You must ONLY output the rewritten answer directly, without any explanations, prefixes, or meta-commentary like "the rewritten output is" or "explanation is". Just provide the rewritten answer itself.

Rewritten Output:"""
        prompt.append({"role": "user", "content": user_content})
        
        return {
            "prompt": prompt,
            "solution": example["output"],  # 保存原始output作为ground truth
            "input": example["input"],  # 保留原始input用于后续处理
        }
    
    # 转换为datasets.Dataset
    hf_dataset = Dataset.from_list(dataset)
    hf_dataset = hf_dataset.map(make_conversation)
    
    return hf_dataset


def train_rewriter_with_grpo(
    round_idx: int,
    dataset: List[Dict[str, str]],
    main_model: Any,
    main_tokenizer: Any,
    anchor_model: Any,
    anchor_tokenizer: Any,
    rewriter_model_path: str,  # 改为路径，因为GRPOTrainer会自己加载
    config: SelfAdaptiveSFTConfig,
    output_dir: str,
) -> str:
    """
    使用open-r1的GRPOTrainer训练重写模型。
    
    使用标准的GRPO实现，支持wandb监控。
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 导入open-r1模块
    # 确保使用open-r1的trl包，而不是系统安装的trl
    # 先导入open_r1相关模块，确保路径正确
    from open_r1.configs import GRPOConfig  # 使用open-r1的GRPOConfig，它包含chat_template属性
    from open_r1.utils import get_tokenizer
    from open_r1.utils.callbacks import get_callbacks
    from open_r1.utils.wandb_logging import init_wandb_training
    
    # 确保使用open-r1路径下的trl包
    # 如果trl已经被导入（可能是系统安装的版本），先移除它
    import importlib
    trl_modules = [k for k in sys.modules.keys() if k.startswith('trl')]
    for mod in trl_modules:
        del sys.modules[mod]
    
    # 现在导入open-r1的trl包
    from trl import GRPOTrainer, ModelConfig, get_peft_config
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 限制数据集大小
    if config.max_samples_per_round and len(dataset) > config.max_samples_per_round:
        import random
        dataset = random.sample(dataset, config.max_samples_per_round)
    
    # 准备数据集（转换为open-r1格式）
    logger.info(f"准备数据集，共 {len(dataset)} 条数据...")
    system_prompt = "You are a data rewriting assistant. Your task is to rewrite outputs to make them better for training while maintaining correctness. You must ONLY output the rewritten answer directly, without any explanations, prefixes, or meta-commentary."
    grpo_dataset = prepare_dataset_for_grpo(dataset, system_prompt=system_prompt)
    
    # 创建reward函数
    # 确保main_model和anchor_model在eval模式（用于推理，不训练）
    logger.info("准备reward函数使用的模型...")
    main_model.eval()
    if anchor_model is not None:
        anchor_model.eval()
    
    logger.info("创建logprob reward函数...")
    logprob_reward_func = create_logprob_reward_func(
        main_model=main_model,
        main_tokenizer=main_tokenizer,
        anchor_model=anchor_model,
        anchor_tokenizer=anchor_tokenizer,
        reward_use_delta=config.reward_use_delta,
        kl_penalty_coef=config.kl_penalty_coef,
        device=config.device,
    )
    
    # 准备训练参数 - 内存优化
    # 减小batch_size，增加gradient_accumulation_steps，减小num_generations
    # effective_batch_size = config.batch_size
    # per_device_batch_size = max(1, effective_batch_size // 2)  # 减小到原来的1/2
    # gradient_accumulation_steps = max(1, effective_batch_size // per_device_batch_size)
    
    # 配置WandB（如果启用）
    if config.use_wandb:
        wandb_project = getattr(config, 'wandb_project', None) or 'self_adaptive_sft'
        wandb_run_name = getattr(config, 'wandb_run_name', None) or f"round_{round_idx}"
        grpo_run_name = f"{wandb_run_name}_grpo"  # GRPO阶段添加_grpo后缀
        
        # 设置环境变量（open-r1的GRPOTrainer会读取）
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_NAME"] = grpo_run_name
        report_to_list = ["wandb"]
    else:
        report_to_list = []
    
    training_args = GRPOConfig(
        output_dir=os.path.join(output_dir, f"grpo_round_{round_idx}"),
        max_steps=config.grpo_training_steps if config.grpo_training_steps > 0 else -1,
        num_train_epochs=1 if config.grpo_training_steps > 0 else 3,
        per_device_train_batch_size=config.grpo_batch_size,  # 使用GRPO专用的batch size
        learning_rate=config.learning_rate,
        bf16=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,  # 启用梯度检查点以节省内存
        logging_steps=10,
        save_steps=500,
        save_strategy="steps",
        report_to=report_to_list,
        log_completions=True,
        max_prompt_length=config.grpo_max_prompt_length,
        max_completion_length=config.grpo_max_completion_length,
        num_generations=4,  # 减小到2以节省内存
        seed=config.seed,
    )
    
    # 初始化wandb（如果启用）
    if config.use_wandb:
        init_wandb_training(training_args)
        logger.info(f"WandB已初始化: 项目={wandb_project}, 运行名称={grpo_run_name}")
    
    # 模型参数
    model_args = ModelConfig(
        model_name_or_path=rewriter_model_path,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2" if hasattr(torch.backends.cuda, 'flash_sdp_enabled') else None,
    )
    
    # 设置model_init_kwargs，关键：use_cache=False when gradient_checkpointing
    torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype != "auto" else torch.bfloat16
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # 关键：gradient_checkpointing时需要关闭cache
    )
    # 关键：如果使用 flash_attention_2，必须确保模型在 GPU 上初始化
    # 否则会出现 "You are attempting to use Flash Attention 2.0 with a model not initialized on GPU" 错误
    if model_args.attn_implementation == "flash_attention_2" and config.device == "cuda" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"  # 确保模型在 GPU 上初始化
    training_args.model_init_kwargs = model_kwargs
    
    # 获取tokenizer
    tokenizer = get_tokenizer(model_args, training_args)
    
    # 创建GRPO trainer
    logger.info("初始化GRPOTrainer...")
    trainer = GRPOTrainer(
        model=rewriter_model_path,
        reward_funcs=[logprob_reward_func],
        args=training_args,
        train_dataset=grpo_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        callbacks=get_callbacks(training_args, model_args),
        peft_config=get_peft_config(model_args),  # 添加PEFT配置，可能使用LoRA减少内存
    )
    
    # 训练
    logger.info(f"开始GRPO训练（Round {round_idx}）...")
    trainer.train()
    
    # 保存模型
    save_path = os.path.join(output_dir, f"rewriter_model_round_{round_idx}")
    trainer.save_model(save_path)
    logger.info(f"模型已保存到: {save_path}")
    
    return save_path


def rewrite_dataset_with_model(
    rewriter_model: Any,
    rewriter_tokenizer: Any,
    dataset: List[Dict[str, str]],
    device: str = "cuda",
    batch_size: int = 64,  # 进一步增大batch_size以提高GPU利用率（从32增加到64）
    max_length: Optional[int] = None,  # 如果为None，使用config中的值
    max_new_tokens: Optional[int] = None,  # 如果为None，使用config中的值
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_vllm: bool = False,  # 是否使用vLLM加速推理
    rewriter_model_path: Optional[str] = None,  # 模型路径（用于vLLM）
    config: Optional[Any] = None,  # 配置对象（用于获取默认长度参数）
) -> List[Dict[str, str]]:
    """
    使用训练好的重写模型批量改写整个数据集。
    
    支持两种推理方式：
    1. 使用transformers的generate（默认）
    2. 使用vLLM加速推理（use_vllm=True，需要提供rewriter_model_path）
    
    Args:
        rewriter_model: 重写模型（不使用vLLM时）
        rewriter_tokenizer: tokenizer
        dataset: 原始数据集，格式为 [{"input": str, "output": str}]
        device: 设备
        batch_size: 批次大小
        max_length: 最大生成长度（如果为None，使用config中的值）
        max_new_tokens: 最大生成token数（如果为None，使用config中的值）
        temperature: 生成温度
        top_p: top-p采样
        use_vllm: 是否使用vLLM加速推理
        rewriter_model_path: 模型路径（使用vLLM时必须提供）
        config: 配置对象（用于获取默认长度参数）
        
    Returns:
        改写后的数据集，格式为 [{"input": str, "output": str}]，其中output为改写后的
    """
    # 使用config中的默认值（如果未提供）
    if max_length is None and config is not None:
        max_length = config.inference_max_length
    if max_new_tokens is None and config is not None:
        max_new_tokens = config.inference_max_new_tokens
    
    # 如果仍然为None，使用默认值
    if max_length is None:
        max_length = 512
    if max_new_tokens is None:
        max_new_tokens = 256
    
    print(f"开始批量改写数据集，共 {len(dataset)} 条数据...")
    print(f"使用参数: max_length={max_length}, max_new_tokens={max_new_tokens}")
    
    # 如果使用vLLM，使用vLLM进行推理
    if use_vllm:
        if rewriter_model_path is None:
            raise ValueError("使用vLLM时必须提供 rewriter_model_path 参数")
        return rewrite_dataset_with_vllm(
            rewriter_model_path=rewriter_model_path,
            dataset=dataset,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    
    # 使用transformers的generate方法
    rewriter_model.eval()
    rewritten_dataset = []
    
    # 批量处理
    for i in tqdm(range(0, len(dataset), batch_size), desc="Rewriting dataset"):
        batch = dataset[i:i+batch_size]
        inputs = [item["input"] for item in batch]
        
        # 批量生成改写
        original_outputs = [item["output"] for item in batch]
        rewritten_outputs = generate_rewritten_outputs(
            rewriter_model,
            rewriter_tokenizer,
            inputs,
            original_outputs=original_outputs,
            device=device,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # 构造改写后的数据
        for j, item in enumerate(batch):
            rewritten_dataset.append({
                "input": item["input"],
                "output": rewritten_outputs[j] if j < len(rewritten_outputs) else item["output"],
            })
    
    print(f"完成改写，共生成 {len(rewritten_dataset)} 条数据")
    return rewritten_dataset


def rewrite_dataset_with_vllm(
    rewriter_model_path: str,
    dataset: List[Dict[str, str]],
    batch_size: int = 128,  # vLLM可以支持更大的batch_size
    max_length: int = 512,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, str]]:
    """
    使用vLLM进行高效的批量推理。
    
    参考 LLaMA-Factory/scripts/vllm_infer.py 的实现。
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vLLM未安装。请安装vLLM: pip install vllm\n"
            "或者设置 use_vllm=False 使用transformers的generate方法"
        )
    
    print(f"使用vLLM进行推理加速...")
    
    # 1. 初始化vLLM引擎
    engine_args = {
        "model": rewriter_model_path,
        "trust_remote_code": True,
        "dtype": "bfloat16",  # 使用bfloat16
        "max_model_len": max_length + max_new_tokens,
        "tensor_parallel_size": 1,  # 可以根据GPU数量调整
        "disable_log_stats": True,
    }
    
    print("正在初始化vLLM引擎...")
    llm = LLM(**engine_args)
    
    # 2. 准备所有prompts
    prompts = []
    for item in dataset:
        input_text = item["input"]
        original_output = item["output"]
        prompt = f"""You are a data rewriting assistant. Your task is to rewrite the given output to make it better for training while maintaining correctness.

Input: {input_text}
Original Output: {original_output}

Rewrite the output above. You must ONLY output the rewritten answer directly, without any explanations, prefixes, or meta-commentary like "the rewritten output is" or "explanation is". Just provide the rewritten answer itself.

Rewritten Output:"""
        prompts.append(prompt)
    
    # 3. 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )
    
    # 4. 批量生成（vLLM会自动处理batching）
    print("开始批量生成...")
    # vLLM的generate返回结果，需要按request_id排序以确保顺序正确
    results = list(llm.generate(prompts, sampling_params))
    
    # 5. 提取生成结果（vLLM需要按request_id排序以确保顺序正确）
    # 关键：vLLM的返回结果可能不是按输入顺序的，需要根据request_id排序
    results = sorted(results, key=lambda x: int(x.request_id))
    generated_texts = [result.outputs[0].text for result in results]
    
    # 6. 提取改写后的输出部分
    rewritten_dataset = []
    for i, (generated_text, item) in enumerate(zip(generated_texts, dataset)):
        # 去掉prompt部分，只保留生成的内容
        if "Rewritten Output:" in generated_text:
            rewritten_output = generated_text.split("Rewritten Output:")[-1].strip()
        elif "Output:" in generated_text:
            rewritten_output = generated_text.split("Output:")[-1].strip()
        else:
            # 如果没有找到分隔符，使用生成的文本
            rewritten_output = generated_text.strip()
        
        rewritten_dataset.append({
            "input": item["input"],
            "output": rewritten_output if rewritten_output else item["output"],
        })
    
    print(f"完成改写，共生成 {len(rewritten_dataset)} 条数据")
    return rewritten_dataset


