import os
from typing import Tuple
import torch

from .config import SelfAdaptiveSFTConfig
from .data import load_supervised_dataset, save_rewritten_dataset
from .models import (
    load_main_model, load_rewriter_model,
    save_main_model, save_rewriter_model
)
from .rewrite_trainer import (
    train_rewriter_with_grpo,
    rewrite_dataset_with_model,
)
from .sft_trainer import train_main_model_with_sft
from .data_filter import filter_dataset_by_logprob


def run_self_adaptive_sft_loop(config: SelfAdaptiveSFTConfig) -> Tuple[object, object]:
    """
    自适应 SFT 主训练循环。

    流程：
    1. 加载数据 D = {(x, y)}
    2. 加载初始模型 M0, m0
    3. 迭代训练 n 轮：
       - 用 GRPO 训练重写模型 m_t -> m_{t+1}
       - 使用 m_{t+1} 推理生成改写数据 D_{t+1}
       - 使用改写数据对主模型做 SFT: M_t -> M_{t+1}
       - 保存模型和数据
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    print("=" * 80)
    print("开始自适应SFT训练流程")
    print("=" * 80)
    
    # 1. 加载数据 D = {(x, y)}
    print(f"\n[步骤1] 加载数据: {config.data_path}")
    dataset = load_supervised_dataset(config.data_path)
    print(f"加载完成，共 {len(dataset)} 条数据")
    
    # 2. 加载初始模型 M0, m0
    print(f"\n[步骤2] 加载初始模型")
    print(f"  主模型: {config.base_model_path}")
    main_model, main_tokenizer = load_main_model(
        config.base_model_path, device=config.device
    )
    print(f"  重写模型: {config.base_rewriter_path}")
    rewriter_model, rewriter_tokenizer = load_rewriter_model(
        config.base_rewriter_path, device=config.device
    )
    
    # anchor_model 作为 KL 惩罚的锚点（始终使用初始的 M0）
    anchor_model, anchor_tokenizer = main_model, main_tokenizer
    print("  Anchor模型已设置为初始主模型（用于KL惩罚）")
    
    # 3. 迭代训练
    print(f"\n[步骤3] 开始迭代训练，共 {config.num_rounds} 轮")
    
    for t in range(config.num_rounds):
        print("\n" + "=" * 80)
        print(f"第 {t+1}/{config.num_rounds} 轮训练")
        print("=" * 80)
        
        round_dir = os.path.join(config.output_dir, f"round_{t}")
        os.makedirs(round_dir, exist_ok=True)
        
        # 如果使用初始重写模型对所有数据重写一次，然后只执行SFT（消融实验模式）
        if config.rewrite_once_with_initial_model:
            print(f"\n[轮次 {t+1}] 消融实验模式：使用初始重写模型对所有数据重写一次，然后只执行SFT...")
            
            # 使用初始重写模型对所有数据进行一次重写
            print(f"\n[轮次 {t+1}.0] 使用初始重写模型对所有 {len(dataset)} 条数据进行改写...")
            rewritten_data = rewrite_dataset_with_model(
                rewriter_model=rewriter_model,
                rewriter_tokenizer=rewriter_tokenizer,
                dataset=dataset,  # 对所有数据进行改写
                device=config.device,
                batch_size=max(64, config.grpo_batch_size * 4),  # 推理时使用更大的batch_size
                max_length=config.inference_max_length,
                max_new_tokens=config.inference_max_new_tokens,
                use_vllm=config.use_vllm,
                rewriter_model_path=config.base_rewriter_path if config.use_vllm else None,  # vLLM需要模型路径
                config=config,
            )
            
            # 保存改写后的数据
            rewritten_data_path = os.path.join(round_dir, "rewritten_dataset.jsonl")
            save_rewritten_dataset(rewritten_data, rewritten_data_path)
            print(f"改写数据已保存到: {rewritten_data_path}")
            
            final_training_dataset = rewritten_data
            print(f"使用改写后的数据：{len(final_training_dataset)} 条")
            
            # 保存最终训练数据
            final_data_path = os.path.join(round_dir, "final_training_dataset.jsonl")
            save_rewritten_dataset(final_training_dataset, final_data_path)
            print(f"训练数据已保存到: {final_data_path}")
            
            # 进行SFT训练
            print(f"\n[轮次 {t+1}.1] SFT训练主模型...")
            
            # 确定模型路径（用于LLaMA-Factory加载）
            if t == 0:
                model_path_for_sft = config.base_model_path
            else:
                model_path_for_sft = os.path.abspath(
                    os.path.join(config.output_dir, f"round_{t-1}", "main_model_final")
                )
            
            # 如果使用DeepSpeed，先释放模型显存
            if config.use_deepspeed:
                print("使用DeepSpeed，先释放模型显存，让DeepSpeed重新加载...")
                import gc
                del main_model
                del main_tokenizer
                torch.cuda.empty_cache()
                gc.collect()
                main_model_for_sft = None
                main_tokenizer_for_sft = None
            else:
                main_model_for_sft = main_model
                main_tokenizer_for_sft = main_tokenizer
            
            main_model, main_tokenizer = train_main_model_with_sft(
                round_idx=t,
                base_model=main_model_for_sft,
                base_tokenizer=main_tokenizer_for_sft,
                rewritten_dataset=final_training_dataset,
                config=config,
                output_dir=round_dir,
                model_path_override=model_path_for_sft if config.use_deepspeed else None,
            )
            
            # 保存模型
            print(f"\n[轮次 {t+1}.2] 保存模型...")
            save_main_model(
                main_model, main_tokenizer,
                os.path.join(round_dir, "main_model_final")
            )
            
            print(f"第 {t+1} 轮训练完成！")
            continue
        
        # 如果跳过GRPO，直接使用原始数据进行SFT
        if config.skip_grpo:
            print(f"\n[轮次 {t+1}] 跳过GRPO训练和改写，直接使用原始数据进行SFT...")
            final_training_dataset = dataset
            print(f"使用原始数据：{len(final_training_dataset)} 条")
            
            # 保存数据
            final_data_path = os.path.join(round_dir, "final_training_dataset.jsonl")
            save_rewritten_dataset(final_training_dataset, final_data_path)
            print(f"训练数据已保存到: {final_data_path}")
            
            # 直接进行SFT训练
            print(f"\n[轮次 {t+1}.1] SFT训练主模型...")
            main_model, main_tokenizer = train_main_model_with_sft(
                round_idx=t,
                base_model=main_model,
                base_tokenizer=main_tokenizer,
                rewritten_dataset=final_training_dataset,
                config=config,
                output_dir=round_dir,
            )
            
            # 保存模型
            print(f"\n[轮次 {t+1}.2] 保存模型...")
            save_main_model(
                main_model, main_tokenizer,
                os.path.join(round_dir, "main_model_final")
            )
            
            print(f"第 {t+1} 轮训练完成！")
            continue
        
        # 3.0 根据logprob筛选数据（只对不align的数据进行改写）
        if config.filter_by_logprob:
            print(f"\n[轮次 {t+1}.0] 根据logprob筛选需要改写的数据...")
            need_rewrite_data, no_need_rewrite_data = filter_dataset_by_logprob(
                dataset=dataset,
                main_model=main_model,
                main_tokenizer=main_tokenizer,
                device=config.device,
                filter_ratio=config.filter_ratio,
                batch_size=16,  # 批量计算logprob（7B模型建议16-32，3B模型可以用64）
            )
            print(f"筛选完成：{len(need_rewrite_data)} 条需要改写，{len(no_need_rewrite_data)} 条不需要改写")
        else:
            # 如果不筛选，所有数据都需要改写
            need_rewrite_data = dataset
            no_need_rewrite_data = []
            print(f"\n[轮次 {t+1}.0] 跳过数据筛选，所有 {len(dataset)} 条数据都将进行改写")
        
        # 3.1 用 GRPO 训练重写模型 m_t -> m_{t+1}（只使用需要改写的数据）
        if len(need_rewrite_data) > 0:
            print(f"\n[轮次 {t+1}.1] GRPO训练重写模型（使用 {len(need_rewrite_data)} 条数据）...")
            # 第一轮使用初始模型路径，后续使用上一轮训练后的模型
            if t == 0:
                current_rewriter_path = config.base_rewriter_path
            else:
                # 转换为绝对路径，避免HuggingFace将其误判为repo id
                current_rewriter_path = os.path.abspath(
                    os.path.join(config.output_dir, f"round_{t-1}", f"rewriter_model_round_{t-1}")
                )
            
            # 使用open-r1的GRPOTrainer训练（只使用需要改写的数据）
            trained_model_path = train_rewriter_with_grpo(
                round_idx=t,
                dataset=need_rewrite_data,  # 只使用需要改写的数据
                main_model=main_model,
                main_tokenizer=main_tokenizer,
                anchor_model=anchor_model,
                anchor_tokenizer=anchor_tokenizer,
                rewriter_model_path=current_rewriter_path,
                config=config,
                output_dir=round_dir,
            )
            
            # 重新加载训练后的模型用于后续推理
            rewriter_model, rewriter_tokenizer = load_rewriter_model(
                trained_model_path, device=config.device
            )
            
            # 获取训练后的模型路径（用于vLLM）
            # 转换为绝对路径，避免HuggingFace将其误判为repo id
            trained_model_path = os.path.abspath(
                os.path.join(config.output_dir, f"round_{t}", f"rewriter_model_round_{t}")
            )
            
            # 3.2 使用 m_{t+1} 生成改写数据 D_{t+1}（只改写需要改写的数据）
            print(f"\n[轮次 {t+1}.2] 使用重写模型生成改写数据（{len(need_rewrite_data)} 条）...")
            rewritten_data = rewrite_dataset_with_model(
                rewriter_model=rewriter_model,
                rewriter_tokenizer=rewriter_tokenizer,
                dataset=need_rewrite_data,  # 只改写需要改写的数据
                device=config.device,
                batch_size=max(64, config.grpo_batch_size * 4),  # 推理时使用更大的batch_size（小模型可以用大batch）
                max_length=config.inference_max_length,  # 使用配置中的统一参数
                max_new_tokens=config.inference_max_new_tokens,  # 使用配置中的统一参数
                use_vllm=config.use_vllm,  # 使用配置中的vLLM选项
                rewriter_model_path=trained_model_path,  # vLLM需要模型路径
                config=config,  # 传入config以便使用默认值
            )
            
            # 保存改写后的数据
            rewritten_data_path = os.path.join(round_dir, "rewritten_dataset.jsonl")
            save_rewritten_dataset(rewritten_data, rewritten_data_path)
            print(f"改写数据已保存到: {rewritten_data_path}")
        else:
            # 如果没有需要改写的数据，跳过GRPO训练和改写
            print(f"\n[轮次 {t+1}.1-2] 跳过GRPO训练和改写（没有需要改写的数据）")
            rewritten_data = []
            # 如果没有训练重写模型，使用上一轮的模型（或初始模型）
            if t == 0:
                rewriter_model, rewriter_tokenizer = load_rewriter_model(
                    config.base_rewriter_path, device=config.device
                )
            else:
                # 转换为绝对路径，避免HuggingFace将其误判为repo id
                previous_model_path = os.path.abspath(
                    os.path.join(config.output_dir, f"round_{t-1}", f"rewriter_model_round_{t-1}")
                )
                rewriter_model, rewriter_tokenizer = load_rewriter_model(
                    previous_model_path, device=config.device
                )
        
        # 3.3 合并数据：改写后的数据 + 不需要改写的数据（使用原始数据）
        print(f"\n[轮次 {t+1}.3] 合并训练数据...")
        final_training_dataset = rewritten_data + no_need_rewrite_data
        print(f"最终训练数据：{len(rewritten_data)} 条改写数据 + {len(no_need_rewrite_data)} 条原始数据 = {len(final_training_dataset)} 条")
        
        # 保存合并后的数据
        final_data_path = os.path.join(round_dir, "final_training_dataset.jsonl")
        save_rewritten_dataset(final_training_dataset, final_data_path)
        print(f"合并后的训练数据已保存到: {final_data_path}")
        
        # 3.4 使用合并后的数据对主模型做 SFT: M_t -> M_{t+1}
        print(f"\n[轮次 {t+1}.4] SFT训练主模型...")
        
        # 关键：如果使用DeepSpeed，需要先释放模型显存，让LLaMA-Factory在DeepSpeed初始化后重新加载
        # 确定模型路径（用于LLaMA-Factory加载）
        if t == 0:
            # 第一轮使用初始模型路径
            model_path_for_sft = config.base_model_path
        else:
            # 后续轮次使用上一轮训练后的模型
            # 转换为绝对路径，避免HuggingFace将其误判为repo id
            model_path_for_sft = os.path.abspath(
                os.path.join(config.output_dir, f"round_{t-1}", "main_model_final")
            )
        
        # 如果使用DeepSpeed，先释放模型显存
        if config.use_deepspeed:
            print("使用DeepSpeed，先释放模型显存，让DeepSpeed重新加载...")
            import gc
            del main_model
            del main_tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            # 传递None，让sft_trainer从路径加载
            main_model_for_sft = None
            main_tokenizer_for_sft = None
        else:
            # 不使用DeepSpeed，可以传递模型对象
            main_model_for_sft = main_model
            main_tokenizer_for_sft = main_tokenizer
        
        main_model, main_tokenizer = train_main_model_with_sft(
            round_idx=t,
            base_model=main_model_for_sft,
            base_tokenizer=main_tokenizer_for_sft,
            rewritten_dataset=final_training_dataset,  # 使用合并后的数据
            config=config,
            output_dir=round_dir,
            model_path_override=model_path_for_sft if config.use_deepspeed else None,  # DeepSpeed时强制使用路径
        )
        
        # 3.5 保存每一轮的模型
        print(f"\n[轮次 {t+1}.5] 保存模型...")
        save_rewriter_model(
            rewriter_model, rewriter_tokenizer,
            os.path.join(round_dir, "rewriter_model_final")
        )
        save_main_model(
            main_model, main_tokenizer,
            os.path.join(round_dir, "main_model_final")
        )
        
        print(f"第 {t+1} 轮训练完成！")
    
    # 保存最终模型
    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    save_main_model(main_model, main_tokenizer, os.path.join(final_dir, "main_model"))
    save_rewriter_model(rewriter_model, rewriter_tokenizer, os.path.join(final_dir, "rewriter_model"))
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最终模型保存在: {final_dir}")
    print("=" * 80)
    
    return main_model, rewriter_model


