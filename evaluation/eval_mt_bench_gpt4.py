#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT-Bench 评估脚本（GPT-4 judge 版本）

思路（简化版单模型打分）：
- 使用 tests/mt_bench/mt_bench_questions.jsonl 中的题目；
- pred 文件为模型在这些题目上的回答：
    - JSONL，每行字段：
        - qid: 与题目文件中的 qid 对应
        - answer: 模型对整个多轮对话的回答（可以是拼接后的回答，或仅最后一轮回答）
- 本脚本调用 OpenAI 的 GPT-4 模型，对每个 (question, answer) 打 1-10 分；
- 最终输出平均分。

注意：
- 需要环境变量 OPENAI_API_KEY
- 需要安装 openai 官方 SDK（>=1.0.0）：pip install openai
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

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


def ask_gpt4_score(client: OpenAI, question: str, answer: str, model: str) -> float:
    """
    调用 GPT-4 对 (question, answer) 打分，返回 1-10 之间的浮点数。
    """
    system_prompt = (
        "You are an expert judge for instruction-following ability of AI assistants. "
        "Given a user's instructions and an assistant's answer, you should rate the answer "
        "on a scale from 1 to 10, where 1 is very poor and 10 is excellent. "
        "Only output a single number."
    )
    user_prompt = (
        f"User instructions and context:\n{question}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        "Please provide a single integer score from 1 to 10:"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content.strip()
    try:
        score = float(content)
    except ValueError:
        # 简单鲁棒性处理：提取第一个数字
        digits = "".join(ch for ch in content if ch.isdigit() or ch == ".")
        try:
            score = float(digits)
        except Exception:
            score = 0.0
    # 截断到 [1,10]
    score = max(1.0, min(10.0, score))
    return score


def main():
    parser = argparse.ArgumentParser(description="使用 GPT-4 对 MT-Bench 结果打分")
    parser.add_argument(
        "--question_file",
        type=str,
        default="tests/mt_bench/mt_bench_questions.jsonl",
        help="MT-Bench 题目 JSONL 文件路径",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="模型预测结果 JSONL 文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="用于打分的 GPT-4 模型名称（如 gpt-4.1, gpt-4.1-mini 等）",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先在环境变量中设置 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    questions = load_jsonl(Path(args.question_file))
    preds = load_jsonl(Path(args.pred_file))

    q_map: Dict[str, Dict] = {str(q["qid"]): q for q in questions}
    pred_map: Dict[str, str] = {str(p["qid"]): p.get("answer", "") for p in preds}

    total = 0
    score_sum = 0.0

    for qid, q in q_map.items():
        if qid not in pred_map:
            continue
        turns = q.get("turns", [])
        question_text = build_prompt(turns)
        answer_text = pred_map[qid]
        score = ask_gpt4_score(client, question_text, answer_text, args.model)
        total += 1
        score_sum += score
        print(f"qid={qid}, score={score:.2f}")

    if total == 0:
        print("未找到任何可评估样本。")
        return

    avg_score = score_sum / total
    print(f"\nMT-Bench 平均得分: {avg_score:.4f} / 10")


if __name__ == "__main__":
    main()


