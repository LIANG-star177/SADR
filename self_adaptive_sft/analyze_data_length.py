"""
分析训练数据的实际长度分布，用于确定合理的序列长度设置
"""
import json
import sys
from typing import List, Dict
from transformers import AutoTokenizer

def analyze_data_length(data_path: str, model_path: str = "Qwen/Qwen2.5-3B-Instruct", sample_size: int = 1000):
    """
    分析数据集中输入和输出的token长度分布
    
    Args:
        data_path: 数据文件路径
        model_path: 模型路径（用于tokenizer）
        sample_size: 采样数量（如果数据量大，只分析前N条）
    """
    # 加载tokenizer
    print(f"加载tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载数据
    print(f"加载数据: {data_path}")
    dataset = []
    if data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                dataset.append(json.loads(line.strip()))
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset = data[:sample_size] if isinstance(data, list) else [data]
    
    print(f"分析 {len(dataset)} 条数据...")
    
    # 计算长度
    input_lengths = []
    output_lengths = []
    total_lengths = []
    
    for item in dataset:
        # 获取input和output
        input_text = item.get("input", item.get("question", ""))
        output_text = item.get("output", item.get("original_answer", item.get("answer", "")))
        
        # Tokenize并计算长度
        input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
        output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
        
        input_lengths.append(len(input_tokens))
        output_lengths.append(len(output_tokens))
        total_lengths.append(len(input_tokens) + len(output_tokens))
    
    # 统计信息
    def print_stats(name, lengths):
        import numpy as np
        lengths = np.array(lengths)
        print(f"\n{name}:")
        print(f"  平均: {lengths.mean():.1f} tokens")
        print(f"  中位数: {np.median(lengths):.1f} tokens")
        print(f"  最小: {lengths.min()} tokens")
        print(f"  最大: {lengths.max()} tokens")
        print(f"  75%分位: {np.percentile(lengths, 75):.1f} tokens")
        print(f"  90%分位: {np.percentile(lengths, 90):.1f} tokens")
        print(f"  95%分位: {np.percentile(lengths, 95):.1f} tokens")
        print(f"  99%分位: {np.percentile(lengths, 99):.1f} tokens")
        
        # 计算覆盖率
        for cutoff in [128, 256, 512, 1024, 2048]:
            coverage = (lengths <= cutoff).sum() / len(lengths) * 100
            print(f"  {cutoff} tokens覆盖率: {coverage:.1f}%")
    
    print("=" * 60)
    print("数据长度分析结果")
    print("=" * 60)
    print_stats("Input长度", input_lengths)
    print_stats("Output长度", output_lengths)
    print_stats("总长度 (Input+Output)", total_lengths)
    
    # 建议
    import numpy as np
    total_lengths = np.array(total_lengths)
    print("\n" + "=" * 60)
    print("建议的序列长度设置")
    print("=" * 60)
    
    # 覆盖95%的数据
    cutoff_95 = int(np.percentile(total_lengths, 95))
    cutoff_99 = int(np.percentile(total_lengths, 99))
    
    print(f"\n1. SFT训练 cutoff_len:")
    print(f"   - 保守（覆盖95%数据）: {cutoff_95} tokens")
    print(f"   - 完整（覆盖99%数据）: {cutoff_99} tokens")
    print(f"   - 平衡（推荐）: {min(512, cutoff_95)} tokens")
    
    # GRPO设置
    output_lengths = np.array(output_lengths)
    max_output_95 = int(np.percentile(output_lengths, 95))
    max_output_99 = int(np.percentile(output_lengths, 99))
    
    print(f"\n2. GRPO训练:")
    print(f"   - max_prompt_length: {int(np.percentile(input_lengths, 95))} tokens (覆盖95%输入)")
    print(f"   - max_completion_length: {max_output_95} tokens (覆盖95%输出)")
    
    print(f"\n3. GRPO推理生成:")
    print(f"   - max_new_tokens: {max_output_95} tokens (覆盖95%输出长度)")
    
    print(f"\n4. 其他计算（logprob等）:")
    print(f"   - max_length: {min(512, cutoff_95)} tokens")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python analyze_data_length.py <data_path> [model_path] [sample_size]")
        print("示例: python analyze_data_length.py training_data/training_data_100k.json Qwen/Qwen2.5-3B-Instruct 1000")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-3B-Instruct"
    sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    analyze_data_length(data_path, model_path, sample_size)

