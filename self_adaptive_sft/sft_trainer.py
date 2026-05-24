from typing import Any, List, Dict, Optional
import os
import json
import sys
import torch
import subprocess
import random

# 设置环境变量以跳过trl版本检查（因为GRPO需要trl==0.17.0.dev0，而SFT要求trl>=0.8.6,<=0.9.6）
# 在导入llamafactory之前设置，确保版本检查被跳过
os.environ["DISABLE_VERSION_CHECK"] = "1"

# 添加LLaMA-Factory路径
# self_adaptive_sft和LLaMA-Factory是并列文件夹
align_tax_dir = os.path.dirname(os.path.dirname(__file__))  # align_tax
LLAMAFACTORY_PATH = os.path.join(align_tax_dir, "LLaMA-Factory")
if LLAMAFACTORY_PATH not in sys.path:
    sys.path.insert(0, os.path.join(LLAMAFACTORY_PATH, "src"))

from .config import SelfAdaptiveSFTConfig


def train_main_model_with_sft(
    round_idx: int,
    base_model: Any,
    base_tokenizer: Any,
    rewritten_dataset: List[Dict[str, str]],
    config: SelfAdaptiveSFTConfig,
    output_dir: str,
    model_path_override: Optional[str] = None,  # 如果提供，强制使用此路径（用于DeepSpeed）
) -> Any:
    """
    使用LLaMA-Factory的训练框架进行SFT训练。
    
    将改写后的数据转换为LLaMA-Factory的alpaca格式，然后使用LLaMA-Factory的训练框架。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 将数据转换为LLaMA-Factory的alpaca格式
    # 添加system消息，使其与测试时的qwen25-math-cot格式一致
    # system_message = "Please reason step by step, and put your final answer within \\boxed{}."
    
    alpaca_dataset = []
    for item in rewritten_dataset:
        input_text = item["input"]
        output_text = item["output"]
        
        # 检查input_text是否已经包含system_message，避免重复叠加
        # 如果已经包含，就不添加；如果没有，则添加到开头
        # if system_message not in input_text:
        #     # 将system_message添加到input_text的开头
        #     input_text = f"{system_message}\n\n{input_text}"
        
        # 尝试分离instruction和input（如果有明显的分隔符）
        if "\n" in input_text:
            parts = input_text.split("\n", 1)
            instruction = parts[0].strip()
            input_content = parts[1].strip() if len(parts) > 1 else ""
        else:
            instruction = input_text
            input_content = ""
        
        alpaca_dataset.append({
            "instruction": instruction,
            "input": input_content,
            "output": output_text,
        })
    
    # 2. 保存为临时JSON文件
    temp_dataset_path = os.path.join(output_dir, f"temp_dataset_round_{round_idx}.json")
    with open(temp_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_dataset, f, ensure_ascii=False, indent=2)
    
    # 3. 准备LLaMA-Factory的训练参数
    # 获取模型路径（如果是模型对象，需要先保存）
    # 如果提供了model_path_override（DeepSpeed场景），直接使用
    if model_path_override:
        # 规范化路径（将相对路径转换为绝对路径，但保持 HuggingFace 模型 ID 不变）
        from .models import normalize_model_path
        model_path = normalize_model_path(model_path_override)
        print(f"使用提供的模型路径（DeepSpeed模式）: {model_path}")
    else:
        # 检查模型是否有保存的路径信息
        model_path = None
        if base_model is not None and hasattr(base_model, 'config') and hasattr(base_model.config, '_name_or_path'):
            potential_path = base_model.config._name_or_path
            # 检查是否是有效的路径（不是模型ID）
            if os.path.exists(potential_path) or os.path.isdir(potential_path):
                model_path = potential_path
            # 如果是模型ID（如 "Qwen/Qwen2.5-3B-Instruct"），也可以直接使用
            elif "/" in potential_path and not os.path.exists(potential_path):
                model_path = potential_path  # HuggingFace模型ID
        
        # 如果无法从config获取路径，需要保存模型
        # 关键：在保存模型前，先释放base_model的显存，避免重复占用
        if model_path is None or not (os.path.exists(model_path) or "/" in model_path):
            temp_model_path = os.path.join(output_dir, f"temp_model_round_{round_idx}")
            print(f"保存临时模型到: {temp_model_path}")
            print("注意：保存模型前会先释放原模型的显存，避免重复占用")
            
            # 先释放base_model的显存
            import gc
            if base_model is not None:
                del base_model
            if base_tokenizer is not None:
                del base_tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
            # 重新从路径加载并保存（避免在显存中保存两份）
            from .models import load_main_model
            if round_idx == 0:
                model_source_path = config.base_model_path
            else:
                # 转换为绝对路径，避免HuggingFace将其误判为repo id
                model_source_path = os.path.abspath(
                    os.path.join(config.output_dir, f"round_{round_idx-1}", "main_model_final")
                )
            base_model, base_tokenizer = load_main_model(
                model_source_path,
                device="cpu"  # 先加载到CPU，保存后再释放
            )
            base_model.save_pretrained(temp_model_path)
            base_tokenizer.save_pretrained(temp_model_path)
            
            # 释放CPU上的模型
            del base_model
            del base_tokenizer
            gc.collect()
            
            # 转换为绝对路径，避免HuggingFace将其误判为repo id
            model_path = os.path.abspath(temp_model_path)
    
    # 4. 构造LLaMA-Factory的训练参数
    from llamafactory.hparams.parser import get_train_args
    
    # 准备训练参数字典（完全按照LLaMA-Factory标准方式）
    # 将output_dir转换为绝对路径，避免工作目录切换后找不到文件
    output_dir_abs = os.path.abspath(output_dir)
    
    train_args_dict = {
        # Model arguments
        "model_name_or_path": model_path,
        "trust_remote_code": True,
        "flash_attn": "auto",
        
        # Data arguments
        "dataset": "custom_dataset",
        "dataset_dir": output_dir_abs,  # 使用绝对路径
        "cutoff_len": config.cutoff_len,
        "template": "qwen",
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,  # 数据预处理并行数（加速数据加载）
        "dataloader_num_workers": 4,  # DataLoader工作进程数（加速数据加载）
        
        # Training arguments
        "stage": "sft",
        "do_train": True,
        "do_eval": False,
        "output_dir": os.path.join(output_dir_abs, f"main_model_round_{round_idx}"),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": config.sft_batch_size,
        "gradient_accumulation_steps": config.sft_gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": 1,
        "max_steps": config.sft_training_steps if config.sft_training_steps > 0 else -1,
        "logging_steps": 10,
        "save_steps": config.sft_training_steps if config.sft_training_steps > 0 else 500,
        "save_total_limit": 1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "max_grad_norm": 1.0,
        "optim": "adamw_torch",
        
        # Finetuning arguments
        "finetuning_type": "full",
    }
    
    # 配置WandB（如果启用）
    if config.use_wandb:
        wandb_project = getattr(config, 'wandb_project', None) or 'self_adaptive_sft'
        wandb_run_name = getattr(config, 'wandb_run_name', None) or f"round_{round_idx}"
        sft_run_name = f"{wandb_run_name}_sft"  # SFT阶段添加_sft后缀
        
        # 设置环境变量（LLaMA-Factory和HuggingFace Trainer会读取）
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_NAME"] = sft_run_name
        
        # 添加run_name到训练参数（HuggingFace TrainingArguments支持）
        train_args_dict["run_name"] = sft_run_name
        train_args_dict["report_to"] = ["wandb"]
        
        print(f"启用WandB日志记录: 项目={wandb_project}, 运行名称={sft_run_name}")
        print(f"提示: 如果未登录WandB，请运行 'wandb login' 或在环境变量中设置 WANDB_API_KEY")
    else:
        train_args_dict["report_to"] = []
    
    # 添加DeepSpeed支持（如果启用）
    if config.use_deepspeed:
        # 确定DeepSpeed配置文件路径
        if config.deepspeed_config_path:
            ds_config_path = os.path.abspath(config.deepspeed_config_path)
        else:
            # cache在align_tax目录下
            align_tax_dir = os.path.dirname(os.path.dirname(__file__))  # align_tax
            cache_dir = os.path.join(align_tax_dir, "cache")
            # 优先使用ZeRO-2（速度更快），如果OOM再使用ZeRO-3
            # ZeRO-2: 速度更快，但显存占用稍大
            # ZeRO-3: 显存占用更小，但通信开销更大（可能慢20-30%）
            ds_config_path = os.path.abspath(os.path.join(cache_dir, "ds_z3_config.json"))
            if not os.path.exists(ds_config_path):
                # 尝试LLaMA-Factory的示例配置
                llamafactory_examples = os.path.abspath(os.path.join(LLAMAFACTORY_PATH, "examples", "deepspeed", "ds_z2_config.json"))
                if os.path.exists(llamafactory_examples):
                    ds_config_path = llamafactory_examples
                else:
                    # 回退到ZeRO-3（如果ZeRO-2不存在）
                    ds_config_path = os.path.abspath(os.path.join(cache_dir, "ds_z2_config.json"))
                    if not os.path.exists(ds_config_path):
                        llamafactory_examples = os.path.abspath(os.path.join(LLAMAFACTORY_PATH, "examples", "deepspeed", "ds_z3_config.json"))
                        if os.path.exists(llamafactory_examples):
                            ds_config_path = llamafactory_examples
        
        if ds_config_path and os.path.exists(ds_config_path):
            train_args_dict["deepspeed"] = ds_config_path
            # 关键：禁用low_cpu_mem_usage以避免使用device_map（DeepSpeed需要自己管理模型加载）
            # LLaMA-Factory只对ZeRO-3禁用low_cpu_mem_usage，但ZeRO-2也需要禁用device_map
            train_args_dict["low_cpu_mem_usage"] = False
            print(f"启用DeepSpeed: {ds_config_path}")
        else:
            print(f"警告: DeepSpeed配置文件不存在: {ds_config_path}")
    
    # 5. 创建dataset_info.json（LLaMA-Factory需要这个文件来识别数据集）
    dataset_info_path = os.path.join(output_dir_abs, "dataset_info.json")
    dataset_info = {
        "custom_dataset": {
            "file_name": os.path.basename(temp_dataset_path),
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    # 6. 设置环境变量，让LLaMA-Factory能找到dataset_info.json
    original_dataset_dir = os.environ.get("LLAMAFACTORY_DATASET_DIR", None)
    os.environ["LLAMAFACTORY_DATASET_DIR"] = output_dir_abs
    
    # 7. 使用LLaMA-Factory的标准训练流程（通过torchrun启动）
    try:
        from llamafactory.extras.misc import get_device_count
        num_gpus = get_device_count()
    except:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    try:
        if num_gpus > 1:
            # 多卡：使用torchrun启动LLaMA-Factory的标准训练入口
            print(f"检测到 {num_gpus} 个GPU，使用 torchrun 启动分布式训练...")
            
            # 保存训练参数到JSON文件（LLaMA-Factory支持JSON配置文件）
            train_args_json_path = os.path.join(output_dir_abs, f"train_args_round_{round_idx}.json")
            # 转换为绝对路径，因为后面会切换工作目录
            train_args_json_path = os.path.abspath(train_args_json_path)
            with open(train_args_json_path, 'w', encoding='utf-8') as f:
                json.dump(train_args_dict, f, ensure_ascii=False, indent=2)
            
            # 使用LLaMA-Factory的标准训练脚本
            train_script_path = os.path.join(LLAMAFACTORY_PATH, 'src', 'train.py')
            
            # 使用torchrun运行，传递JSON配置文件
            master_port = random.randint(20001, 29999)
            cmd = [
                "torchrun",
                "--nproc_per_node", str(num_gpus),
                "--master_port", str(master_port),
                train_script_path,
                train_args_json_path  # 使用绝对路径
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            # 确保环境变量被传递到子进程（虽然默认会继承，但显式传递更保险）
            env = os.environ.copy()
            env["DISABLE_VERSION_CHECK"] = "1"
            result = subprocess.run(cmd, check=True, cwd=LLAMAFACTORY_PATH, env=env)
        else:
            # 单卡：直接调用
            print("单卡训练，直接调用训练函数...")
            from llamafactory.train.tuner import run_exp
            run_exp(args=train_args_dict, callbacks=None)
        
        # 8. 重新加载训练后的模型
        from .models import load_main_model
        # 已经是绝对路径（output_dir_abs），但为了确保一致性，再次转换为绝对路径
        saved_model_path = os.path.abspath(os.path.join(output_dir_abs, f"main_model_round_{round_idx}"))
        saved_model, saved_tokenizer = load_main_model(
            saved_model_path,
            device=config.device
        )
        
        return saved_model, saved_tokenizer
        
    finally:
        # 恢复环境变量
        if original_dataset_dir is not None:
            os.environ["LLAMAFACTORY_DATASET_DIR"] = original_dataset_dir
        elif "LLAMAFACTORY_DATASET_DIR" in os.environ:
            del os.environ["LLAMAFACTORY_DATASET_DIR"]
        
        # 清理临时文件（可选）
        # os.remove(temp_dataset_path)  # 保留数据集文件以便调试


