"""
示例运行脚本：自适应SFT训练

使用方法:
    python run_example.py
"""

from .config import SelfAdaptiveSFTConfig
from .main import run_self_adaptive_sft


def main():
    # 配置参数
    config = SelfAdaptiveSFTConfig(
        # 数据路径
        data_path="../../gsm8k_aligned_answers.jsonl",  # 修改为你的数据路径
        
        # 输出目录
        output_dir="./output/self_adaptive_sft",
        
        # 模型路径（使用HuggingFace模型名称或本地路径）
        base_model_path="Qwen/Qwen2.5-3B-Instruct",  # 主模型（3B或7B）
        base_rewriter_path="Qwen/Qwen2.5-1.5B-Instruct",  # 重写模型（1.5B）
        
        # 训练超参数
        num_rounds=3,  # 迭代轮数
        grpo_training_steps=100,  # 每轮GRPO训练步数
        sft_training_steps=500,  # 每轮SFT训练步数
        batch_size=4,
        learning_rate=1e-5,
        
        # Reward相关
        reward_use_delta=True,  # True: logprob(y') - logprob(y); False: logprob(y')
        kl_penalty_coef=0.1,  # KL惩罚系数
        
        # 其他
        device="cuda",
        seed=42,
    )
    
    # 运行训练
    final_main_model, final_rewriter_model = run_self_adaptive_sft(config)
    
    print("\n训练完成！")
    print(f"最终模型保存在: {config.output_dir}/final/")


if __name__ == "__main__":
    main()

