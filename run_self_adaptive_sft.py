#!/usr/bin/env python
"""
自适应SFT训练主入口脚本

使用方法:
    python run_self_adaptive_sft.py --data_path <数据路径> --output_dir <输出目录> [其他参数]
    
或者直接修改下面的配置然后运行:
    python run_self_adaptive_sft.py
"""

import sys
import os
import argparse

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from self_adaptive_sft.config import SelfAdaptiveSFTConfig
from self_adaptive_sft.main import run_self_adaptive_sft


def parse_args():
    parser = argparse.ArgumentParser(description="自适应SFT训练")
    
    parser.add_argument("--data_path", type=str, default="training_data/training_data_100k.json",
                        help="原始监督数据路径（JSONL格式）")
    parser.add_argument("--output_dir", type=str, default="./output/SFT_3b_r3b",
                        help="输出目录")
    
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="初始主模型路径（3B或7B）")
    parser.add_argument("--base_rewriter_path", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="初始重写模型路径（1.5B）")
    
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="迭代轮数（建议3-5轮）")
    parser.add_argument("--grpo_training_steps", type=int, default=100,
                        help="每轮GRPO训练步数（默认52步，6卡训练：20k筛选数据 / 有效batch 384）")
    parser.add_argument("--sft_training_steps", type=int, default=1042,
                        help="每轮SFT训练步数（默认1041步，6卡训练：100k数据 / 有效batch 96）")
    parser.add_argument("--grpo_batch_size", type=int, default=4,
                        help="GRPO训练小模型的batch size（可以设置较大，如4-8，因为小模型显存占用小）")
    parser.add_argument("--grpo_gradient_accumulation_steps", type=int, default=16,
                        help="GRPO梯度累积步数（参考SOTA配置，建议16-32，有效batch size = grpo_batch_size * grpo_gradient_accumulation_steps * num_gpus）")
    parser.add_argument("--sft_batch_size", type=int, default=2,
                        help="SFT训练大模型的batch size（需要设置较小，如1-2，因为大模型显存占用大。如果显存充足，可以尝试增加到2以提升速度）")
    parser.add_argument("--sft_gradient_accumulation_steps", type=int, default=8,
                        help="SFT梯度累积步数（通过累积梯度保持有效batch size，显存占用取决于sft_batch_size，有效batch size = sft_batch_size * sft_gradient_accumulation_steps * num_gpus）")
    parser.add_argument("--cutoff_len", type=int, default=2048,
                        help="SFT训练的总序列长度（input+output），建议覆盖95%%数据，先用analyze_data_length.py分析")
    parser.add_argument("--grpo_max_prompt_length", type=int, default=256,
                        help="GRPO训练的prompt最大长度")
    parser.add_argument("--grpo_max_completion_length", type=int, default=1024,
                        help="GRPO训练的completion最大长度")
    parser.add_argument("--inference_max_length", type=int, default=256,
                        help="推理时的最大输入长度（input部分）")
    parser.add_argument("--inference_max_new_tokens", type=int, default=2048,
                        help="推理时的最大生成token数（output部分）")
    parser.add_argument("--logprob_max_length", type=int, default=2048,
                        help="logprob计算时的最大长度")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率（GRPO建议2e-5，SFT建议5e-5，当前为统一设置）")
    parser.add_argument("--grpo_learning_rate", type=float, default=2e-5,
                        help="GRPO专用学习率（如果设置，将覆盖统一的learning_rate）")
    parser.add_argument("--sft_learning_rate", type=float, default=5e-5,
                        help="SFT专用学习率（如果设置，将覆盖统一的learning_rate）")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup比例（建议0.05-0.1，SOTA常用0.1）")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="学习率调度器类型（cosine/cosine_with_min_lr/linear，SOTA常用cosine）")
    
    parser.add_argument("--reward_use_delta", action="store_true", default=False,
                        help="使用delta reward（logprob(y') - logprob(y)）")
    parser.add_argument("--kl_penalty_coef", type=float, default=0.1,
                        help="KL惩罚系数")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="使用wandb监控训练（默认启用）")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false",
                        help="禁用wandb监控训练")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB项目名称（默认: self_adaptive_sft）")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB运行名称（默认: 自动生成）")
    parser.add_argument("--max_samples_per_round", type=int, default=None,
                        help="每轮训练的最大样本数（None表示使用全部）")
    parser.add_argument("--use_vllm", action="store_true", default=True,
                        help="使用vLLM加速推理（需要安装vllm包，速度可提升10-50倍）")
    parser.add_argument("--use_deepspeed", action="store_true", default=True,
                        help="使用DeepSpeed ZeRO优化（强烈建议开启，可大幅节省显存，之前能跑7B就是因为这个）")
    parser.add_argument("--deepspeed_config_path", type=str, default="cache/ds_z2_config.json",
                        help="DeepSpeed配置文件路径（默认使用cache/ds_z2_config.json，ZeRO-2速度更快。如果OOM，可尝试ds_z3_config.json或ds_z3_offload_config.json）")
    
    parser.add_argument("--filter_by_logprob", action="store_true", default=True,
                        help="根据logprob筛选需要改写的数据（只对不align的数据进行改写）")
    parser.add_argument("--filter_ratio", type=float, default=0.2,
                        help="数据筛选比例（0.2表示筛选出最低的20%，即1/5的数据进行改写，SOTA常用0.2）")
    parser.add_argument("--skip_grpo", action="store_true", default=False,
                        help="跳过GRPO训练和改写，直接使用原始数据进行SFT（用于测试，正式训练应设为False）")
    parser.add_argument("--rewrite_once_with_initial_model", action="store_true", default=False,
                        help="使用初始重写模型对所有数据重写一次，然后只执行SFT（用于消融实验，通常配合--num_rounds 1使用）")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备（cuda/cpu）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建配置
    config = SelfAdaptiveSFTConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path,
        base_rewriter_path=args.base_rewriter_path,
        num_rounds=args.num_rounds,
        grpo_training_steps=args.grpo_training_steps,
        sft_training_steps=args.sft_training_steps,
        grpo_batch_size=args.grpo_batch_size,
        grpo_gradient_accumulation_steps=getattr(args, 'grpo_gradient_accumulation_steps', 16),
        sft_batch_size=args.sft_batch_size,
        sft_gradient_accumulation_steps=args.sft_gradient_accumulation_steps,
        cutoff_len=args.cutoff_len,
        grpo_max_prompt_length=args.grpo_max_prompt_length,
        grpo_max_completion_length=args.grpo_max_completion_length,
        inference_max_length=args.inference_max_length,
        inference_max_new_tokens=args.inference_max_new_tokens,
        logprob_max_length=args.logprob_max_length,
        learning_rate=args.learning_rate,
        grpo_learning_rate=getattr(args, 'grpo_learning_rate', None),
        sft_learning_rate=getattr(args, 'sft_learning_rate', None),
        warmup_ratio=getattr(args, 'warmup_ratio', 0.1),
        lr_scheduler_type=getattr(args, 'lr_scheduler_type', 'cosine'),
        reward_use_delta=args.reward_use_delta,
        kl_penalty_coef=args.kl_penalty_coef,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        max_samples_per_round=args.max_samples_per_round,
        device=args.device,
        seed=args.seed,
        use_vllm=args.use_vllm,
        use_deepspeed=args.use_deepspeed,
        deepspeed_config_path=args.deepspeed_config_path,
        filter_by_logprob=args.filter_by_logprob,
        filter_ratio=args.filter_ratio,
        skip_grpo=args.skip_grpo,
        rewrite_once_with_initial_model=args.rewrite_once_with_initial_model,
    )
    
    print("=" * 80)
    print("自适应SFT训练配置")
    print("=" * 80)
    print(f"数据路径: {config.data_path}")
    print(f"输出目录: {config.output_dir}")
    print(f"主模型: {config.base_model_path}")
    print(f"重写模型: {config.base_rewriter_path}")
    print(f"迭代轮数: {config.num_rounds}")
    print(f"GRPO训练步数/轮: {config.grpo_training_steps}")
    print(f"SFT训练步数/轮: {config.sft_training_steps}")
    print(f"GRPO批次大小: {config.grpo_batch_size} (训练小模型)")
    print(f"GRPO梯度累积步数: {getattr(config, 'grpo_gradient_accumulation_steps', 16)}")
    print(f"SFT批次大小: {config.sft_batch_size} (训练大模型)")
    print(f"SFT梯度累积步数: {config.sft_gradient_accumulation_steps}")
    print(f"\n序列长度设置:")
    print(f"  SFT cutoff_len: {config.cutoff_len} (input+output总长度)")
    print(f"  GRPO prompt长度: {config.grpo_max_prompt_length}")
    print(f"  GRPO completion长度: {config.grpo_max_completion_length}")
    print(f"  推理输入长度: {config.inference_max_length}")
    print(f"  推理生成长度: {config.inference_max_new_tokens}")
    print(f"  logprob计算长度: {config.logprob_max_length}")
    grpo_effective_bs = config.grpo_batch_size * getattr(config, 'grpo_gradient_accumulation_steps', 16)
    sft_effective_bs = config.sft_batch_size * config.sft_gradient_accumulation_steps
    print(f"\n有效批次大小 (单卡):")
    print(f"  GRPO: {grpo_effective_bs}")
    print(f"  SFT: {sft_effective_bs}")
    grpo_lr = getattr(config, 'grpo_learning_rate', None) or config.learning_rate
    sft_lr = getattr(config, 'sft_learning_rate', None) or config.learning_rate
    print(f"\n学习率设置:")
    print(f"  GRPO: {grpo_lr} {'(专用)' if getattr(config, 'grpo_learning_rate', None) else '(统一)'}")
    print(f"  SFT: {sft_lr} {'(专用)' if getattr(config, 'sft_learning_rate', None) else '(统一)'}")
    print(f"  Warmup比例: {getattr(config, 'warmup_ratio', 0.1)}")
    print(f"  学习率调度器: {getattr(config, 'lr_scheduler_type', 'cosine')}")
    print(f"Reward类型: {'Delta' if config.reward_use_delta else 'Absolute'}")
    print(f"KL惩罚系数: {config.kl_penalty_coef}")
    if config.use_wandb:
        wandb_project = getattr(config, 'wandb_project', None) or 'self_adaptive_sft'
        print(f"WandB监控: 启用 (项目: {wandb_project})")
    else:
        print(f"WandB监控: 禁用")
    print(f"每轮最大样本数: {config.max_samples_per_round or '全部'}")
    print(f"使用vLLM加速推理: {config.use_vllm}")
    print(f"使用DeepSpeed优化: {config.use_deepspeed} {'(关键！可大幅节省显存)' if config.use_deepspeed else '(建议开启)'}")
    if config.use_deepspeed:
        if config.deepspeed_config_path:
            print(f"DeepSpeed配置文件: {config.deepspeed_config_path}")
        else:
            print(f"DeepSpeed配置文件: 默认使用ZeRO-2 (ds_z2_config.json，速度更快。如果OOM，可尝试ZeRO-3)")
    print(f"根据logprob筛选数据: {config.filter_by_logprob}")
    if config.filter_by_logprob:
        print(f"数据筛选比例: {config.filter_ratio*100:.1f}% (只对最低的{config.filter_ratio*100:.1f}%数据进行改写)")
    print(f"跳过GRPO训练: {config.skip_grpo} {'(测试模式：直接使用原始数据进行SFT)' if config.skip_grpo else ''}")
    print(f"使用初始模型重写一次: {config.rewrite_once_with_initial_model} {'(消融实验模式：使用初始重写模型对所有数据重写一次，然后只执行SFT)' if config.rewrite_once_with_initial_model else ''}")
    print("=" * 80)
    print()
    
    # 运行训练
    try:
        final_main_model, final_rewriter_model = run_self_adaptive_sft(config)
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print(f"最终模型保存在: {config.output_dir}/final/")
        print("=" * 80)
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

