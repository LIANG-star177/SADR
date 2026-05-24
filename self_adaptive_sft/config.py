from dataclasses import dataclass
from typing import Optional


@dataclass
class SelfAdaptiveSFTConfig:
    """
    自适应 SFT 主流程配置。
    后续如果需要，可以再根据实际的 Open-R1 / LLaMA-Factory 接口逐步扩展。
    """

    # 数据与输出
    data_path: str               # 原始监督数据 (x, y) 路径
    output_dir: str              # 实验输出根目录

    # 模型路径
    base_model_path: str         # 初始主模型 M0 (例如: Qwen/Qwen2.5-3B-Instruct)
    base_rewriter_path: str      # 初始重写模型 m0 (例如: Qwen/Qwen2.5-1.5B-Instruct)

    # 训练超参
    num_rounds: int = 3
    grpo_training_steps: int = 52   # 6卡训练：确保训练完所有筛选数据（20k数据 / 有效batch 384）
    sft_training_steps: int = 1041   # 6卡训练：确保训练完所有数据（100k数据 / 有效batch 96）
    grpo_batch_size: int = 4  # GRPO训练小模型的batch size（可以设置较大，如4-8）
    grpo_gradient_accumulation_steps: int = 16  # GRPO梯度累积步数（SOTA常用16-32）
    sft_batch_size: int = 1  # SFT训练大模型的batch size（需要设置较小，如1-2）
    sft_gradient_accumulation_steps: int = 8  # SFT梯度累积步数，增大可以减小显存
    learning_rate: float = 2e-5  # 统一学习率（如果grpo_learning_rate或sft_learning_rate未设置则使用此值）
    grpo_learning_rate: Optional[float] = None  # GRPO专用学习率（SOTA常用2e-5）
    sft_learning_rate: Optional[float] = None  # SFT专用学习率（SOTA常用5e-5）
    warmup_ratio: float = 0.1  # Warmup比例（SOTA常用0.1）
    lr_scheduler_type: str = "cosine"  # 学习率调度器类型（cosine/cosine_with_min_lr/linear）
    
    # 序列长度设置（统一管理，建议先用analyze_data_length.py分析数据后设置）
    cutoff_len: int = 512  # SFT训练的总序列长度（input+output），建议覆盖95%数据
    grpo_max_prompt_length: int = 256  # GRPO训练的prompt最大长度
    grpo_max_completion_length: int = 256  # GRPO训练的completion最大长度
    inference_max_length: int = 512  # 推理时的最大输入长度（input部分）
    inference_max_new_tokens: int = 256  # 推理时的最大生成token数（output部分）
    logprob_max_length: int = 512  # logprob计算时的最大长度

    # reward 相关
    reward_use_delta: bool = True   # True: logprob(y') - logprob(y); False: logprob(y')
    kl_penalty_coef: float = 0.1
    max_samples_per_round: Optional[int] = None
    
    # 数据筛选相关
    filter_by_logprob: bool = True  # 是否根据logprob筛选数据
    filter_ratio: float = 0.2  # 筛选比例（0.2表示筛选出最低的20%，即1/5的数据进行改写）

    # 其它
    device: str = "cuda"
    seed: int = 42
    use_wandb: bool = True  # 是否使用wandb监控训练
    wandb_project: Optional[str] = None  # WandB项目名称（默认使用"self_adaptive_sft"）
    wandb_run_name: Optional[str] = None  # WandB运行名称（默认使用"round_{round_idx}"）
    use_vllm: bool = False  # 是否使用vLLM加速推理（需要安装vllm包）
    skip_grpo: bool = False  # 是否跳过GRPO训练和改写，直接使用原始数据进行SFT（用于测试）
    rewrite_once_with_initial_model: bool = False  # 是否使用初始重写模型对所有数据重写一次，然后只执行SFT（用于消融实验）
    use_deepspeed: bool = True  # 是否使用DeepSpeed ZeRO优化（强烈建议开启，可大幅节省显存）
    deepspeed_config_path: Optional[str] = None  # DeepSpeed配置文件路径，如果为None，默认使用ds_z2_config.json（ZeRO-2，速度更快。如果OOM，可尝试ZeRO-3）


