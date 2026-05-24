from typing import Tuple

from .config import SelfAdaptiveSFTConfig
from .loop import run_self_adaptive_sft_loop


def run_self_adaptive_sft(config: SelfAdaptiveSFTConfig) -> Tuple[object, object]:
    """
    对外主入口：运行自适应 SFT 完整流程。

    返回最终的大模型 M_n 与重写模型 m_n（这里暂时用 object，占位）。
    具体类型会在与底层模型库（如 transformers / vllm 等）对接时再细化。
    """
    final_main_model, final_rewriter_model = run_self_adaptive_sft_loop(config)
    return final_main_model, final_rewriter_model


