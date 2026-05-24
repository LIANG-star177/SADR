#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT-Bench LLM as Judge 评测脚本
比较两个模型的输出，使用GPT-4o作为judge计算win_rate

使用方法：
    python evaluation/eval_llm_judge_mt_bench.py \
        --question_file tests/mt_bench/mt_bench_questions.jsonl \
        --pred_file_baseline predictions/mt_bench_api_preds.jsonl \
        --pred_file_ours predictions/mt_bench_local_preds.jsonl \
        --judge_model deepseek/deepseek-r1-0528:free
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from openai import OpenAI


def load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_prompt(turns: List[str]) -> str:
    """将 MT-Bench 的多轮指令拼接成一个评估说明。"""
    lines = []
    for i, t in enumerate(turns):
        lines.append(f"Round {i + 1} user instruction:\n{t}\n")
    return "\n".join(lines)


def judge_comparison(client: OpenAI, question: str, answer_a: str, answer_b: str, model: str) -> str:
    """
    调用DeepSeek或GPT-4o判断哪个回答更好
    返回: "A", "B", 或 "Tie"
    """
    system_prompt = (
        "You are an expert judge for comparing AI assistant responses. "
        "Given a user's instructions and two assistant answers (Answer A and Answer B), "
        "you should determine which answer is better, or if they are tied. "
        "Consider factors like: helpfulness, accuracy, relevance, and overall quality. "
        "Only respond with 'A', 'B', or 'Tie'."
    )
    user_prompt = (
        f"User instructions and context:\n{question}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        "Which answer is better? Respond with only 'A', 'B', or 'Tie':"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            extra_headers={
                "HTTP-Referer": "https://github.com",  # Optional. Site URL for rankings on openrouter.ai
                "X-Title": "AlignTax Evaluation",  # Optional. Site title for rankings on openrouter.ai
            },
        )
        content = resp.choices[0].message.content.strip().upper()
        
        # 提取判断结果
        if "A" in content and "B" not in content[:content.find("A")+1]:
            return "A"
        elif "B" in content:
            return "B"
        elif "TIE" in content or "TIE" in content:
            return "Tie"
        else:
            # 默认返回第一个出现的
            if "A" in content:
                return "A"
            elif "B" in content:
                return "B"
            else:
                return "Tie"
    except Exception as e:
        print(f"Error in judge_comparison: {e}")
        return "Tie"


def main():
    parser = argparse.ArgumentParser(description="使用 LLM as Judge 比较两个模型的 MT-Bench 结果")
    parser.add_argument(
        "--question_file",
        type=str,
        default="tests/mt_bench/mt_bench_questions.jsonl",
        help="MT-Bench 题目 JSONL 文件路径",
    )
    parser.add_argument(
        "--pred_file_baseline",
        type=str,
        required=True,
        help="基线模型（qwen3-max）预测结果 JSONL 文件路径",
    )
    parser.add_argument(
        "--pred_file_ours",
        type=str,
        required=True,
        help="我们的模型预测结果 JSONL 文件路径",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="deepseek/deepseek-r1-0528:free",
        help="用于判断的模型名称（默认 deepseek/deepseek-r1-0528:free，也可使用 gpt-4o）",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先在环境变量中设置 OPENAI_API_KEY")

    # 使用 OpenRouter API
    # 模型名称格式：deepseek-chat -> deepseek/deepseek-chat, gpt-4o -> openai/gpt-4o
    # 如果已经包含 /（如 deepseek/deepseek-r1-0528:free），则保持不变
    base_url = "https://openrouter.ai/api/v1"
    
    # 转换模型名称格式（OpenRouter 需要 provider/model 格式）
    model_name = args.judge_model
    if "/" not in model_name:
        # 如果没有 provider 前缀，根据模型名称添加
        if model_name.startswith("deepseek"):
            # deepseek-chat -> deepseek/deepseek-chat
            model_name = f"deepseek/{model_name}"
        elif model_name.startswith("gpt"):
            # gpt-4o -> openai/gpt-4o
            model_name = f"openai/{model_name}"
        # 其他情况保持原样，让 OpenRouter 处理
    # 如果已经包含 /，直接使用（如 deepseek/deepseek-r1-0528:free）
    
    client = OpenAI(api_key=api_key, base_url=base_url)

    questions = load_jsonl(Path(args.question_file))
    baseline_preds = load_jsonl(Path(args.pred_file_baseline))
    ours_preds = load_jsonl(Path(args.pred_file_ours))

    q_map: Dict[str, Dict] = {str(q["qid"]): q for q in questions}
    baseline_map: Dict[str, str] = {str(p["qid"]): p.get("answer", "") for p in baseline_preds}
    ours_map: Dict[str, str] = {str(p["qid"]): p.get("answer", "") for p in ours_preds}

    wins = 0
    losses = 0
    ties = 0
    total = 0

    print("开始比较两个模型的输出...")
    for qid, q in tqdm(q_map.items(), desc="Judging"):
        if qid not in baseline_map or qid not in ours_map:
            continue
        
        turns = q.get("turns", [])
        question_text = build_prompt(turns)
        baseline_answer = baseline_map[qid]
        ours_answer = ours_map[qid]
        
        # Answer A = baseline (qwen3-max), Answer B = ours
        judgment = judge_comparison(client, question_text, baseline_answer, ours_answer, args.judge_model)
        
        if judgment == "B":  # 我们的模型赢了
            wins += 1
        elif judgment == "A":  # 基线模型赢了
            losses += 1
        else:  # Tie
            ties += 1
        total += 1
        
        print(f"qid={qid}, judgment={judgment}")

    if total == 0:
        print("未找到任何可评估样本。")
        return

    win_rate = wins / total
    loss_rate = losses / total
    tie_rate = ties / total

    print(f"\n{'='*60}")
    print(f"MT-Bench LLM as Judge 结果 (Judge: {model_name})")
    print(f"{'='*60}")
    print(f"总样本数: {total}")
    print(f"我们的模型获胜: {wins} ({win_rate*100:.2f}%)")
    print(f"基线模型获胜: {losses} ({loss_rate*100:.2f}%)")
    print(f"平局: {ties} ({tie_rate*100:.2f}%)")
    print(f"\nWin Rate (我们的模型): {win_rate:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

