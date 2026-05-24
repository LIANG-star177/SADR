#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMLU 测试集评估脚本

配合 tests/mmlu/mmlu_test.jsonl 使用（由 tests/build_test_sets.py 生成）

gold 文件字段：
    - id
    - subject
    - question
    - choices: List[str]
    - answer: 正确选项索引（0-based）

pred 文件格式（JSONL）：
    - id: 与 gold 中的 id 对应
    - answer: 可以是：
        - 整数索引（0,1,2,3）
        - 字母 'A'/'B'/'C'/'D'

输出：
    - overall accuracy
    - 按 subject 的平均 accuracy
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def parse_pred_answer(raw) -> int:
    """将预测值解析为 0-based 索引。"""
    if raw is None:
        return -1
    # 已经是 int
    if isinstance(raw, int):
        return raw
    s = str(raw).strip().upper()
    # 可能是 'A'/'B'/'C'/'D'
    if len(s) == 1 and s in "ABCD":
        return "ABCD".index(s)
    # 可能是 "0"/"1"/"2"/"3"
    try:
        return int(s)
    except ValueError:
        return -1


def evaluate(gold_file: Path, pred_file: Path) -> Tuple[float, Dict[str, float]]:
    gold_data = load_jsonl(gold_file)
    pred_data = load_jsonl(pred_file)

    pred_map: Dict[str, int] = {}
    for ex in pred_data:
        qid = str(ex.get("id"))
        ans_idx = parse_pred_answer(ex.get("answer"))
        pred_map[qid] = ans_idx

    total = 0
    correct = 0
    per_subject_total = defaultdict(int)
    per_subject_correct = defaultdict(int)

    for ex in gold_data:
        qid = str(ex.get("id"))
        subject = ex.get("subject", "unknown")
        gold_ans = ex.get("answer")
        if qid not in pred_map:
            continue
        pred_ans = pred_map[qid]
        if pred_ans < 0:
            continue

        total += 1
        per_subject_total[subject] += 1
        if pred_ans == gold_ans:
            correct += 1
            per_subject_correct[subject] += 1

    overall_acc = correct / total if total > 0 else 0.0
    per_subject_acc = {
        s: (per_subject_correct[s] / per_subject_total[s])
        for s in per_subject_total
        if per_subject_total[s] > 0
    }
    return overall_acc, per_subject_acc


def main():
    parser = argparse.ArgumentParser(description="评估 MMLU 测试集表现")
    parser.add_argument(
        "--gold_file",
        type=str,
        default="tests/mmlu/mmlu_test.jsonl",
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

    overall, per_subject = evaluate(gold_path, pred_path)
    print(f"MMLU Overall Accuracy: {overall:.4f}")
    print("Per-subject accuracy:")
    for subject, acc in sorted(per_subject.items(), key=lambda x: x[0]):
        print(f"  {subject}: {acc:.4f}")


if __name__ == "__main__":
    main()


