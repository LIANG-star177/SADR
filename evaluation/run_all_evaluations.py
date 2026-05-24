#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键运行所有测试集的评估脚本。

使用方法：
    python evaluation/run_all_evaluations.py \
        --predictions_dir predictions/ \
        --testsets_dir tests/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_prediction_file(pred_dir: Path, base_name: str, model_id: str = None) -> Path:
    """查找预测文件，支持多种文件名模式"""
    patterns = []
    
    # 如果提供了模型标识符，优先使用它来匹配
    if model_id:
        patterns.extend([
            f"{base_name}_{model_id}_vllm_preds.jsonl",  # 如: alpaca_DR_3b_vllm_vllm_preds.jsonl
            f"{base_name}_{model_id}_local_preds.jsonl",  # 如: alpaca_DR_3b_vllm_local_preds.jsonl
            f"{base_name}_{model_id}_preds.jsonl",  # 如: alpaca_DR_3b_vllm_preds.jsonl
        ])
    
    # 尝试多种文件名模式（按优先级）
    patterns.extend([
        f"{base_name}_local_preds.jsonl",  # 本地推理（优先）
        f"{base_name}_api_preds.jsonl",    # API推理
        f"{base_name}_preds.jsonl",        # 标准格式（兼容旧格式）
    ])
    
    # 先尝试精确匹配
    for pattern in patterns:
        pred_file = pred_dir / pattern
        if pred_file.exists():
            return pred_file
    
    # 如果没有找到且没有提供模型标识符，尝试通配符匹配（作为后备方案）
    if not model_id:
        import glob
        wildcard_pattern = str(pred_dir / f"{base_name}_*_preds.jsonl")
        matching_files = glob.glob(wildcard_pattern)
        if matching_files:
            # 返回第一个匹配的文件（按字母顺序）
            return Path(sorted(matching_files)[0])
    
    return None


def run_evaluation(script_name: str, args: list) -> bool:
    """运行评估脚本"""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*60}")
    print(f"运行：{' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="运行所有测试集的评估")
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="./predictions",
        help="预测结果目录（默认 ./predictions）",
    )
    parser.add_argument(
        "--testsets_dir",
        type=str,
        default="./tests",
        help="测试集目录（默认 ./tests）",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="模型标识符，用于匹配预测文件（如 DR_3b_vllm）",
    )
    parser.add_argument(
        "--skip_mt_bench",
        action="store_true",
        help="（已废弃）MT-Bench 请使用专门的 LLM as Judge 评测流程",
    )

    args = parser.parse_args()
    pred_dir = Path(args.predictions_dir)
    test_dir = Path(args.testsets_dir)
    model_id = args.model_id

    if model_id:
        print(f"使用模型标识符匹配预测文件: {model_id}")

    evaluations = []

    # 1. HotpotQA
    hotpotqa_pred = find_prediction_file(pred_dir, "hotpotqa", model_id)
    hotpotqa_test = test_dir / "hotpotqa" / "hotpotqa_test.jsonl"
    if hotpotqa_pred and hotpotqa_test.exists():
        evaluations.append(
            (
                "eval_hotpotqa.py",
                [
                    "--gold_file", str(hotpotqa_test),
                    "--pred_file", str(hotpotqa_pred),
                ],
            )
        )

    # 2. MMLU
    mmlu_pred = find_prediction_file(pred_dir, "mmlu", model_id)
    mmlu_test = test_dir / "mmlu" / "mmlu_test.jsonl"
    if mmlu_pred and mmlu_test.exists():
        evaluations.append(
            (
                "eval_mmlu.py",
                [
                    "--gold_file", str(mmlu_test),
                    "--pred_file", str(mmlu_pred),
                ],
            )
        )

    # 3. Alpaca
    alpaca_pred = find_prediction_file(pred_dir, "alpaca", model_id)
    alpaca_test = test_dir / "alpaca" / "alpaca_robust_test.jsonl"
    if alpaca_pred and alpaca_test.exists():
        evaluations.append(
            (
                "eval_alpaca_robust.py",
                [
                    "--gold_file", str(alpaca_test),
                    "--pred_file", str(alpaca_pred),
                ],
            )
        )

    # 注意: MT-Bench 和 PKU-SafeRLHF 请使用专门的 LLM as Judge 评测流程
    # bash evaluation/run_llm_judge_evaluation.sh

    if len(evaluations) == 0:
        print("未找到可评估的预测文件，请先运行推理脚本生成预测。")
        return

    print(f"将运行 {len(evaluations)} 个评估任务\n")

    results = []
    for script_name, script_args in evaluations:
        success = run_evaluation(script_name, script_args)
        results.append((script_name, success))

    print("\n" + "=" * 60)
    print("评估结果汇总：")
    print("=" * 60)
    for script_name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{status}: {script_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

