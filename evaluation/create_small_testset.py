#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建小测试集 - 随机采样10%的数据用于快速测试

使用方法：
    python evaluation/create_small_testset.py \
        --input_file tests/alpaca/alpaca_robust_test.jsonl \
        --output_file tests/alpaca/alpaca_robust_test_small.jsonl \
        --sample_ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path


def create_small_testset(input_file: str, output_file: str, sample_ratio: float = 0.1, seed: int = 42):
    """从原始测试集中随机采样指定比例的数据"""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 设置随机种子以确保可复现
    random.seed(seed)
    
    # 读取所有数据
    print(f"读取文件: {input_file}")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    total_count = len(records)
    sample_count = max(1, int(total_count * sample_ratio))
    
    print(f"原始数据量: {total_count}")
    print(f"采样比例: {sample_ratio * 100:.1f}%")
    print(f"采样数量: {sample_count}")
    
    # 随机采样
    sampled_records = random.sample(records, sample_count)
    
    # 按原始顺序排序（保持id顺序，处理None值）
    if sampled_records and "id" in sampled_records[0]:
        def get_sort_key(x):
            id_val = x.get("id")
            # 处理None值：将None放在最后
            if id_val is None:
                return float('inf')
            # 处理非数字id
            try:
                return int(id_val)
            except (ValueError, TypeError):
                return str(id_val) if id_val else float('inf')
        
        sampled_records.sort(key=get_sort_key)
    
    # 保存采样结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in sampled_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"小测试集已保存到: {output_file}")
    print(f"实际采样数量: {len(sampled_records)}")


def main():
    parser = argparse.ArgumentParser(description="创建小测试集（随机采样）")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="原始测试集文件路径（JSONL格式）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出小测试集文件路径（JSONL格式）",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.1,
        help="采样比例（默认 0.1，即10%%）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42，确保可复现）",
    )
    
    args = parser.parse_args()
    
    create_small_testset(
        input_file=args.input_file,
        output_file=args.output_file,
        sample_ratio=args.sample_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

