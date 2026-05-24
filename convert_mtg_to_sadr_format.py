#!/usr/bin/env python3
"""
将 MTG 训练数据转换为 SADR 期望的格式

MTG 格式：
{
    "extra_info.question": "...",
    "extra_info.answer": "...",
    ...
}

SADR 格式：
{
    "input": "...",  # 问题
    "output": "..."  # 答案
}
"""
import json
import os
from tqdm import tqdm

def convert_mtg_to_sadr(input_file, output_file, max_samples=None):
    """
    转换 MTG 数据格式为 SADR 格式
    
    Args:
        input_file: MTG 训练数据文件路径 (JSONL)
        output_file: 输出文件路径 (JSONL)
        max_samples: 最大转换样本数（None 表示全部）
    """
    print(f"正在读取 MTG 数据: {input_file}")
    
    converted_data = []
    total_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="转换数据"):
            if max_samples and total_count >= max_samples:
                break
                
            if not line.strip():
                continue
                
            try:
                item = json.loads(line)
                total_count += 1
                
                # 提取问题和答案
                # 优先使用 extra_info 中的字段
                if "extra_info" in item and isinstance(item["extra_info"], dict):
                    question = item["extra_info"].get("question", "")
                    answer = item["extra_info"].get("answer", "")
                else:
                    # 如果没有 extra_info，尝试使用扁平化的字段
                    question = item.get("extra_info.question", item.get("question", ""))
                    answer = item.get("extra_info.answer", item.get("answer", ""))
                
                # 如果都没有，跳过这条数据
                if not question or not answer:
                    skipped_count += 1
                    continue
                
                # 转换为 SADR 格式
                converted_item = {
                    "input": question.strip(),
                    "output": answer.strip()
                }
                
                converted_data.append(converted_item)
                
            except json.JSONDecodeError as e:
                print(f"警告: 跳过无效的 JSON 行: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"警告: 处理数据时出错: {e}")
                skipped_count += 1
                continue
    
    # 保存转换后的数据
    print(f"\n正在保存转换后的数据到: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n转换完成！")
    print(f"  总读取: {total_count} 条")
    print(f"  成功转换: {len(converted_data)} 条")
    print(f"  跳过: {skipped_count} 条")
    print(f"  输出文件: {output_file}")
    
    # 显示前几条样本
    if converted_data:
        print(f"\n前 3 条转换后的样本:")
        print("="*80)
        for i, item in enumerate(converted_data[:3]):
            print(f"\n样本 {i+1}:")
            print(f"Input (前200字符): {item['input'][:200]}...")
            print(f"Output (前200字符): {item['output'][:200]}...")
    
    return len(converted_data)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="将 MTG 训练数据转换为 SADR 格式")
    parser.add_argument("--input", type=str, default="data_mtg/train/train.jsonl",
                        help="MTG 训练数据文件路径")
    parser.add_argument("--output", type=str, default="data_mtg/train/train_sadr_format.jsonl",
                        help="输出文件路径")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大转换样本数（None 表示全部）")
    
    args = parser.parse_args()
    
    convert_mtg_to_sadr(args.input, args.output, args.max_samples)


if __name__ == "__main__":
    main()
