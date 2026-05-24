#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在测试集上进行批量推理，生成预测文件供后续评估使用。

支持所有测试集：
- MT-Bench（多轮对话）
- HotpotQA（问答）
- MMLU（多选题）
- PKU-SafeRLHF（对齐）
- Alpaca（指令跟随）

使用方法：
    python evaluation/inference_on_testsets.py \
        --testset mt_bench \
        --test_file tests/mt_bench/mt_bench_questions.jsonl \
        --output_file predictions/mt_bench_preds.jsonl \
        --model qwen2.5-turbo \
        --tasktype qwen
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# 导入你的推理接口
from simple_thread import stable_chat


def load_testset(test_file: Path) -> List[Dict[str, Any]]:
    """加载测试集文件"""
    records = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def format_mt_bench_prompt(turns: List[str]) -> str:
    """格式化 MT-Bench 多轮对话为单次 prompt"""
    # MT-Bench 是多轮对话，需要模拟对话历史
    prompt_parts = []
    for i, turn in enumerate(turns):
        if i == 0:
            prompt_parts.append(f"问题：{turn}")
        else:
            prompt_parts.append(f"\n请继续回答：{turn}")
    return "\n".join(prompt_parts)


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """格式化 MMLU 多选题为 prompt"""
    prompt = f"问题：{question}\n\n选项：\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "\n请只回答选项字母（A/B/C/D），不要其他内容。"
    return prompt


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    """格式化 Alpaca 格式为 prompt"""
    if input_text and input_text.strip():
        return f"指令：{instruction}\n\n输入：{input_text}\n\n请回答："
    else:
        return f"指令：{instruction}\n\n请回答："


def inference_mt_bench(
    records: List[Dict],
    model: str,
    tasktype: str,
    max_rounds: int = 5,
    requests_per_minite: int = 10,
) -> List[Dict[str, Any]]:
    """MT-Bench 推理：多轮对话"""
    messages = []
    for rec in records:
        qid = rec.get("qid") or rec.get("id")
        turns = rec.get("turns", [])
        if not turns:
            continue
        prompt = format_mt_bench_prompt(turns)
        messages.append({"id": qid, "content": prompt, "label": None})

    results = stable_chat(
        tasktype=tasktype,
        messages=messages,
        model=model,
        max_rounds=max_rounds,
        requests_per_minite=requests_per_minite,
    )

    # 转换为评估脚本需要的格式
    preds = []
    for res in results:
        preds.append({"qid": res["id"], "answer": res.get("response", "")})
    return preds


def inference_hotpotqa(
    records: List[Dict],
    model: str,
    tasktype: str,
    max_rounds: int = 5,
    requests_per_minite: int = 10,
) -> List[Dict[str, Any]]:
    """HotpotQA 推理：问答"""
    messages = []
    for rec in records:
        qid = rec.get("id")
        question = rec.get("question", "")
        if not question:
            continue
        prompt = f"问题：{question}\n\n请直接回答，不要其他解释。"
        messages.append({"id": qid, "content": prompt, "label": None})

    results = stable_chat(
        tasktype=tasktype,
        messages=messages,
        model=model,
        max_rounds=max_rounds,
        requests_per_minite=requests_per_minite,
    )

    preds = []
    for res in results:
        preds.append({"id": res["id"], "answer": res.get("response", "")})
    return preds


def inference_mmlu(
    records: List[Dict],
    model: str,
    tasktype: str,
    max_rounds: int = 5,
    requests_per_minite: int = 10,
) -> List[Dict[str, Any]]:
    """MMLU 推理：多选题"""
    messages = []
    for rec in records:
        qid = rec.get("id")
        question = rec.get("question", "")
        choices = rec.get("choices", [])
        if not question or not choices:
            continue
        prompt = format_mmlu_prompt(question, choices)
        messages.append({"id": qid, "content": prompt, "label": None})

    results = stable_chat(
        tasktype=tasktype,
        messages=messages,
        model=model,
        max_rounds=max_rounds,
        requests_per_minite=requests_per_minite,
    )

    preds = []
    for res in results:
        # 尝试从回答中提取选项字母或索引
        answer_text = res.get("response", "").strip().upper()
        answer_idx = None
        
        # 尝试匹配 A/B/C/D
        if answer_text.startswith("A"):
            answer_idx = 0
        elif answer_text.startswith("B"):
            answer_idx = 1
        elif answer_text.startswith("C"):
            answer_idx = 2
        elif answer_text.startswith("D"):
            answer_idx = 3
        else:
            # 尝试匹配数字 0/1/2/3
            import re
            match = re.search(r"\b([0-3])\b", answer_text)
            if match:
                answer_idx = int(match.group(1))
        
        preds.append({"id": res["id"], "answer": answer_idx if answer_idx is not None else answer_text})
    return preds


def inference_pku_saferlhf(
    records: List[Dict],
    model: str,
    tasktype: str,
    max_rounds: int = 5,
    requests_per_minite: int = 10,
) -> List[Dict[str, Any]]:
    """PKU-SafeRLHF 推理：对齐测试"""
    messages = []
    for idx, rec in enumerate(records):
        # PKU数据集的id可能为null，使用索引作为id
        qid = rec.get("id") or rec.get("sample_id") or idx
        prompt = rec.get("prompt", "")
        if not prompt:
            continue
        messages.append({"id": qid, "content": prompt, "label": None})

    results = stable_chat(
        tasktype=tasktype,
        messages=messages,
        model=model,
        max_rounds=max_rounds,
        requests_per_minite=requests_per_minite,
    )

    preds = []
    for res in results:
        preds.append({"id": res["id"], "answer": res.get("response", "")})
    return preds


def inference_alpaca(
    records: List[Dict],
    model: str,
    tasktype: str,
    max_rounds: int = 5,
    requests_per_minite: int = 10,
) -> List[Dict[str, Any]]:
    """Alpaca 推理：指令跟随"""
    messages = []
    for rec in records:
        qid = rec.get("id")
        instruction = rec.get("instruction", "")
        input_text = rec.get("input", "")
        if not instruction:
            continue
        prompt = format_alpaca_prompt(instruction, input_text)
        messages.append({"id": qid, "content": prompt, "label": None})

    results = stable_chat(
        tasktype=tasktype,
        messages=messages,
        model=model,
        max_rounds=max_rounds,
        requests_per_minite=requests_per_minite,
    )

    preds = []
    for res in results:
        preds.append({"id": res["id"], "answer": res.get("response", "")})
    return preds


def main():
    parser = argparse.ArgumentParser(description="在测试集上进行批量推理")
    parser.add_argument(
        "--testset",
        type=str,
        required=True,
        choices=["mt_bench", "hotpotqa", "mmlu", "pku_saferlhf", "alpaca"],
        help="测试集名称",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="测试集文件路径（JSONL格式）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="预测结果输出文件路径（JSONL格式）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-turbo",
        help="模型名称（默认 qwen2.5-turbo）",
    )
    parser.add_argument(
        "--tasktype",
        type=str,
        default="qwen",
        choices=["qwen", "openai"],
        help="推理接口类型（默认 qwen）",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=5,
        help="每个请求的最大重试次数（默认 5）",
    )
    parser.add_argument(
        "--requests_per_minite",
        type=int,
        default=10,
        help="并发请求数（默认 10）",
    )

    args = parser.parse_args()

    test_file = Path(args.test_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载测试集：{test_file}")
    records = load_testset(test_file)
    print(f"共 {len(records)} 条测试样本")

    # 根据测试集类型选择推理函数
    inference_funcs = {
        "mt_bench": inference_mt_bench,
        "hotpotqa": inference_hotpotqa,
        "mmlu": inference_mmlu,
        "pku_saferlhf": inference_pku_saferlhf,
        "alpaca": inference_alpaca,
    }

    inference_func = inference_funcs[args.testset]
    print(f"\n开始推理（模型：{args.model}，接口：{args.tasktype}）...")
    preds = inference_func(
        records,
        model=args.model,
        tasktype=args.tasktype,
        max_rounds=args.max_rounds,
        requests_per_minite=args.requests_per_minite,
    )

    # 保存预测结果
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"\n预测结果已保存到：{output_file}")
    print(f"共生成 {len(preds)} 条预测")


if __name__ == "__main__":
    main()

