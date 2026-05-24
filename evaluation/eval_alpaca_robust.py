#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpaca 鲁棒性测试集评估脚本（简单自动指标版本）

由于 Alpaca 任务多为开放式指令，本脚本提供一个非常轻量级的自动评估方式：
- gold 文件：tests/alpaca/alpaca_robust_test.jsonl
    - id, instruction, input, output
- pred 文件：模型预测 JSONL
    - id: 与 gold 对应
    - answer: 模型对 (instruction, input) 的输出

评估指标：
- 与参考 output 的 token F1（与 HotpotQA/PKU 中类似的粗略相似度度量）
这不是完美的鲁棒性衡量方式，但可以提供一个大致的自动参考。
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def normalize_text(s: str) -> str:
    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    def remove_punc(text):
        return re.sub(r"[^\w\s]", " ", text)

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(
        description="评估 Alpaca 鲁棒性测试集表现（token F1 近似指标）"
    )
    parser.add_argument(
        "--gold_file",
        type=str,
        default="tests/alpaca/alpaca_robust_test.jsonl",
        help="gold JSONL 文件路径",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="模型预测 JSONL 文件路径",
    )
    args = parser.parse_args()

    gold_data = load_jsonl(Path(args.gold_file))
    pred_data = load_jsonl(Path(args.pred_file))

    gold_map: Dict[str, str] = {}
    for ex in gold_data:
        gid = str(ex.get("id"))
        gold_map[gid] = ex.get("output", "")

    pred_map: Dict[str, str] = {}
    for ex in pred_data:
        pid = str(ex.get("id"))
        pred_map[pid] = ex.get("answer", "")

    total = 0
    f1_sum = 0.0

    for gid, gold_ans in gold_map.items():
        if gid not in pred_map:
            continue
        pred_ans = pred_map[gid]
        total += 1
        f1_sum += f1_score(pred_ans, gold_ans)

    if total == 0:
        print("未找到可评估样本。")
        return

    avg_f1 = f1_sum / total
    print(f"Alpaca Robustness F1 (approx): {avg_f1:.4f}")


if __name__ == "__main__":
    main()


