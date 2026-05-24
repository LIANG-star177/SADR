#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 VLLM 加速的本地模型推理脚本

使用方法：
    python evaluation/test_local_model_vllm.py \
        --model_path output/self_adaptive_sft/round_0/main_model_round_0 \
        --testset mmlu \
        --test_file tests/mmlu/mmlu_test.jsonl \
        --output_file predictions/mmlu_local_preds.jsonl \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.9
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    print("错误: 请安装 vllm 和 transformers")
    print("pip install vllm transformers")
    sys.exit(1)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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


class VLLMModelInference:
    """使用 VLLM 的模型推理类"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        
        print(f"加载 VLLM 模型: {model_path}")
        print(f"Tensor Parallel Size: {tensor_parallel_size}")
        print(f"GPU Memory Utilization: {gpu_memory_utilization}")
        
        # 初始化 VLLM 引擎
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        
        # 加载 tokenizer（用于格式化 prompt）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("VLLM 模型加载完成")
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """批量生成回答，确保顺序匹配"""
        # 格式化 prompts（Qwen 格式）
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # 使用 tokenizer 的 apply_chat_template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(text)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
        )
        
        # 批量生成（VLLM 通常保证返回顺序与输入顺序一致）
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        # 如果返回结果有 request_id，按 request_id 排序以确保顺序正确
        # （某些情况下 VLLM 可能因为连续批处理导致顺序变化）
        if len(outputs) > 0 and hasattr(outputs[0], 'request_id') and outputs[0].request_id:
            try:
                # 尝试按 request_id 排序（如果 request_id 是数字）
                outputs = sorted(outputs, key=lambda x: int(x.request_id) if str(x.request_id).isdigit() else x.request_id)
            except (ValueError, TypeError):
                # 如果 request_id 不是数字或无法排序，保持原顺序
                pass
        
        # 提取生成的文本
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        if len(responses) != len(prompts):
            raise ValueError(f"返回结果数量 ({len(responses)}) 与输入数量 ({len(prompts)}) 不匹配")
        
        return responses


def inference_mt_bench(
    records: List[Dict],
    model_inference: VLLMModelInference,
    batch_size: int = 32,
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """MT-Bench 推理：多轮对话"""
    preds = []
    
    # 准备所有 prompts
    prompts = []
    qids = []
    for rec in records:
        qid = rec.get("qid") or rec.get("id")
        turns = rec.get("turns", [])
        if not turns:
            continue
        prompt = format_mt_bench_prompt(turns)
        prompts.append(prompt)
        qids.append(qid)
    
    # 批量推理
    print(f"开始批量推理 {len(prompts)} 条样本...")
    for i in tqdm(range(0, len(prompts), batch_size), desc="MT-Bench推理"):
        batch_prompts = prompts[i:i+batch_size]
        batch_qids = qids[i:i+batch_size]
        
        answers = model_inference.generate_batch(
            batch_prompts,
            max_new_tokens=max_new_tokens
        )
        
        for qid, answer in zip(batch_qids, answers):
            preds.append({"qid": qid, "answer": answer})
    
    return preds


def inference_hotpotqa(
    records: List[Dict],
    model_inference: VLLMModelInference,
    batch_size: int = 32,
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """HotpotQA 推理：问答"""
    preds = []
    
    # 准备所有 prompts
    prompts = []
    ids = []
    for rec in records:
        qid = rec.get("id")
        question = rec.get("question", "")
        if not question:
            continue
        prompt = f"问题：{question}\n\n请直接回答，不要其他解释。"
        prompts.append(prompt)
        ids.append(qid)
    
    # 批量推理
    print(f"开始批量推理 {len(prompts)} 条样本...")
    for i in tqdm(range(0, len(prompts), batch_size), desc="HotpotQA推理"):
        batch_prompts = prompts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        answers = model_inference.generate_batch(
            batch_prompts,
            max_new_tokens=max_new_tokens
        )
        
        for qid, answer in zip(batch_ids, answers):
            preds.append({"id": qid, "answer": answer})
    
    return preds


def inference_mmlu(
    records: List[Dict],
    model_inference: VLLMModelInference,
    batch_size: int = 32,
    max_new_tokens: int = 10,
) -> List[Dict[str, Any]]:
    """MMLU 推理：多选题"""
    preds = []
    
    # 准备所有 prompts
    prompts = []
    ids = []
    for rec in records:
        qid = rec.get("id")
        question = rec.get("question", "")
        choices = rec.get("choices", [])
        if not question or not choices:
            continue
        prompt = format_mmlu_prompt(question, choices)
        prompts.append(prompt)
        ids.append(qid)
    
    # 批量推理
    print(f"开始批量推理 {len(prompts)} 条样本...")
    for i in tqdm(range(0, len(prompts), batch_size), desc="MMLU推理"):
        batch_prompts = prompts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        answers = model_inference.generate_batch(
            batch_prompts,
            max_new_tokens=max_new_tokens
        )
        
        # 处理答案
        for qid, answer_text in zip(batch_ids, answers):
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
    model_inference: VLLMModelInference,
    batch_size: int = 32,
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """PKU-SafeRLHF 推理：对齐测试"""
    preds = []
    
    # 准备所有 prompts
    prompts = []
    ids = []
    for idx, rec in enumerate(records):
        qid = rec.get("id") or rec.get("sample_id") or idx
        prompt = rec.get("prompt", "")
        if not prompt:
            continue
        prompts.append(prompt)
        ids.append(qid)
    
    # 批量推理
    print(f"开始批量推理 {len(prompts)} 条样本...")
    for i in tqdm(range(0, len(prompts), batch_size), desc="PKU-SafeRLHF推理"):
        batch_prompts = prompts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        answers = model_inference.generate_batch(
            batch_prompts,
            max_new_tokens=max_new_tokens
        )
        
        for qid, answer in zip(batch_ids, answers):
            preds.append({"id": qid, "answer": answer})
    
    return preds


def inference_alpaca(
    records: List[Dict],
    model_inference: VLLMModelInference,
    batch_size: int = 32,
    max_new_tokens: int = 512,
) -> List[Dict[str, Any]]:
    """Alpaca 推理：指令跟随"""
    preds = []
    
    # 准备所有 prompts
    prompts = []
    ids = []
    for rec in records:
        qid = rec.get("id")
        instruction = rec.get("instruction", "")
        input_text = rec.get("input", "")
        if not instruction:
            continue
        prompt = format_alpaca_prompt(instruction, input_text)
        prompts.append(prompt)
        ids.append(qid)
    
    # 批量推理
    print(f"开始批量推理 {len(prompts)} 条样本...")
    for i in tqdm(range(0, len(prompts), batch_size), desc="Alpaca推理"):
        batch_prompts = prompts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        answers = model_inference.generate_batch(
            batch_prompts,
            max_new_tokens=max_new_tokens
        )
        
        for qid, answer in zip(batch_ids, answers):
            preds.append({"id": qid, "answer": answer})
    
    return preds


def main():
    parser = argparse.ArgumentParser(description="使用 VLLM 测试本地训练的模型")
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
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor 并行大小（默认 1）",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU 内存利用率（默认 0.9）",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="最大模型长度（默认 4096）",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成长度（默认 512）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批量大小（默认 32）",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="数据类型（默认 bfloat16）",
    )

    args = parser.parse_args()

    test_file = Path(args.test_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载测试集：{test_file}")
    records = load_testset(test_file)
    print(f"共 {len(records)} 条测试样本")

    # 加载 VLLM 模型
    model_inference = VLLMModelInference(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
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
    print(f"\n开始推理（模型：{args.model_path}，使用 VLLM 加速）...")
    
    # 根据数据集设置不同的 max_new_tokens
    if args.testset == "mmlu":
        max_new_tokens = 10
    else:
        max_new_tokens = args.max_new_tokens
    
    preds = inference_func(
        records,
        model_inference,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
    )

    # 保存预测结果
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"\n预测结果已保存到：{output_file}")
    print(f"共生成 {len(preds)} 条预测")


if __name__ == "__main__":
    main()
