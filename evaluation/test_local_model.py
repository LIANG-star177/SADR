#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试本地训练的模型

使用方法：
    python evaluation/test_local_model.py \
        --model_path output/self_adaptive_sft/round_0/main_model_round_0 \
        --testset mmlu \
        --test_file tests/mmlu/mmlu_test.jsonl \
        --output_file predictions/mmlu_local_preds.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接定义需要的函数（避免导入依赖）
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


class LocalModelInference:
    """本地模型推理类"""
    
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 2048):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        
        print(f"加载本地模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device == "cuda" and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(device)
        
        self.model.eval()
        print("模型加载完成")
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """生成回答"""
        # 格式化prompt（Qwen格式）
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # 使用tokenizer的apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode - 只解码新生成的部分
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()


def inference_mt_bench(
    records: List[Dict],
    model_inference: LocalModelInference,
) -> List[Dict[str, Any]]:
    """MT-Bench 推理：多轮对话"""
    preds = []
    for rec in tqdm(records, desc="MT-Bench推理"):
        qid = rec.get("qid") or rec.get("id")
        turns = rec.get("turns", [])
        if not turns:
            continue
        prompt = format_mt_bench_prompt(turns)
        answer = model_inference.generate(prompt)
        preds.append({"qid": qid, "answer": answer})
    return preds


def inference_hotpotqa(
    records: List[Dict],
    model_inference: LocalModelInference,
) -> List[Dict[str, Any]]:
    """HotpotQA 推理：问答"""
    preds = []
    for rec in tqdm(records, desc="HotpotQA推理"):
        qid = rec.get("id")
        question = rec.get("question", "")
        if not question:
            continue
        prompt = f"问题：{question}\n\n请直接回答，不要其他解释。"
        answer = model_inference.generate(prompt)
        preds.append({"id": qid, "answer": answer})
    return preds


def inference_mmlu(
    records: List[Dict],
    model_inference: LocalModelInference,
) -> List[Dict[str, Any]]:
    """MMLU 推理：多选题"""
    preds = []
    for rec in tqdm(records, desc="MMLU推理"):
        qid = rec.get("id")
        question = rec.get("question", "")
        choices = rec.get("choices", [])
        if not question or not choices:
            continue
        prompt = format_mmlu_prompt(question, choices)
        answer_text = model_inference.generate(prompt, max_new_tokens=10)
        
        # 尝试提取选项
        answer_text = answer_text.strip().upper()
        answer_idx = None
        
        if answer_text.startswith("A"):
            answer_idx = 0
        elif answer_text.startswith("B"):
            answer_idx = 1
        elif answer_text.startswith("C"):
            answer_idx = 2
        elif answer_text.startswith("D"):
            answer_idx = 3
        else:
            import re
            match = re.search(r"\b([0-3])\b", answer_text)
            if match:
                answer_idx = int(match.group(1))
        
        preds.append({"id": qid, "answer": answer_idx if answer_idx is not None else answer_text})
    return preds


def inference_pku_saferlhf(
    records: List[Dict],
    model_inference: LocalModelInference,
) -> List[Dict[str, Any]]:
    """PKU-SafeRLHF 推理：对齐测试"""
    preds = []
    for idx, rec in enumerate(tqdm(records, desc="PKU-SafeRLHF推理")):
        # PKU数据集的id可能为null，使用索引作为id
        qid = rec.get("id") or rec.get("sample_id") or idx
        prompt = rec.get("prompt", "")
        if not prompt:
            continue
        answer = model_inference.generate(prompt)
        preds.append({"id": qid, "answer": answer})
    return preds


def inference_alpaca(
    records: List[Dict],
    model_inference: LocalModelInference,
) -> List[Dict[str, Any]]:
    """Alpaca 推理：指令跟随"""
    preds = []
    for rec in tqdm(records, desc="Alpaca推理"):
        qid = rec.get("id")
        instruction = rec.get("instruction", "")
        input_text = rec.get("input", "")
        if not instruction:
            continue
        prompt = format_alpaca_prompt(instruction, input_text)
        answer = model_inference.generate(prompt)
        preds.append({"id": qid, "answer": answer})
    return preds


def main():
    parser = argparse.ArgumentParser(description="测试本地训练的模型")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="本地模型路径",
    )
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
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备（默认 cuda）",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="最大输入长度（默认 2048）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成长度（默认 512）",
    )

    args = parser.parse_args()

    test_file = Path(args.test_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载测试集：{test_file}")
    records = load_testset(test_file)
    print(f"共 {len(records)} 条测试样本")

    # 加载本地模型
    model_inference = LocalModelInference(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length
    )

    # 根据测试集类型选择推理函数
    inference_funcs = {
        "mt_bench": inference_mt_bench,
        "hotpotqa": inference_hotpotqa,
        "mmlu": inference_mmlu,
        "pku_saferlhf": inference_pku_saferlhf,
        "alpaca": inference_alpaca,
    }

    inference_func = inference_funcs[args.testset]
    print(f"\n开始推理（模型：{args.model_path}）...")
    
    # 修改generate函数以使用max_new_tokens参数
    original_generate = model_inference.generate
    model_inference.generate = lambda prompt, **kwargs: original_generate(
        prompt, 
        max_new_tokens=kwargs.get("max_new_tokens", args.max_new_tokens),
        **{k: v for k, v in kwargs.items() if k != "max_new_tokens"}
    )
    
    preds = inference_func(records, model_inference)

    # 保存预测结果
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"\n预测结果已保存到：{output_file}")
    print(f"共生成 {len(preds)} 条预测")


if __name__ == "__main__":
    main()

