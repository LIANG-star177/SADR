#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HotpotQA 测试集评估脚本

评估指标：
- EM（exact match）
- F1（token 级别 F1，与官方脚本一致思路的简化实现）

输入：
- gold_file: tests/hotpotqa/hotpotqa_test.jsonl
- pred_file: 模型预测结果 JSONL，每行字段：
    - id: 与 gold 中的 id 对应
    - answer: 模型生成的答案字符串
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return re.sub(r"[^\w\s]", " ", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def evaluate(gold_file: Path, pred_file: Path) -> Tuple[float, float]:
    gold_data = load_jsonl(gold_file)
    pred_data = load_jsonl(pred_file)

    pred_map: Dict[str, str] = {}
    for ex in pred_data:
        qid = str(ex.get("id"))
        ans = ex.get("answer", "")
        pred_map[qid] = ans

    total = 0
    em_sum = 0.0
    f1_sum = 0.0

    for ex in gold_data:
        qid = str(ex.get("id"))
        gold_ans = ex.get("answer", "")
        if qid not in pred_map:
            continue
        pred_ans = pred_map[qid]
        total += 1
        em_sum += float(exact_match_score(pred_ans, gold_ans))
        f1_sum += f1_score(pred_ans, gold_ans)

    if total == 0:
        return 0.0, 0.0
    return em_sum / total, f1_sum / total


def main():
    parser = argparse.ArgumentParser(description="评估 HotpotQA 测试集表现")
    parser.add_argument(
        "--gold_file",
        type=str,
        default="tests/hotpotqa/hotpotqa_test.jsonl",
        help="带标准答案的 JSONL 文件路径",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        required=True,
        help="模型预测结果 JSONL 文件路径",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold_file)
    pred_path = Path(args.pred_file)

    em, f1 = evaluate(gold_path, pred_path)
    print(f"HotpotQA EM: {em:.4f}")
    print(f"HotpotQA F1: {f1:.4f}")


if __name__ == "__main__":
    main()


