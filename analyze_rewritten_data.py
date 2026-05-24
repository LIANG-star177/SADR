#!/usr/bin/env python
"""
分析改写后的数据质量
对比原始数据和改写后的数据，评估改写的效果
"""

import json
import sys
import os
from typing import List, Dict

# 添加self_adaptive_sft路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_adaptive_sft.data import load_supervised_dataset

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def analyze_rewritten_data(original_data_path: str, rewritten_data_path: str, num_examples: int = 10):
    """分析改写后的数据"""
    
    print("=" * 80)
    print("改写数据分析")
    print("=" * 80)
    
    # 加载原始数据（使用与训练时相同的数据加载函数，确保格式一致）
    print(f"\n1. 加载原始数据: {original_data_path}")
    try:
        # 使用与训练时相同的数据加载函数，确保格式一致（会合并instruction和input）
        original_data = load_supervised_dataset(original_data_path)
        print(f"   原始数据条数: {len(original_data)} (使用load_supervised_dataset加载)")
    except Exception as e:
        print(f"   警告: 使用load_supervised_dataset加载失败: {e}")
        print(f"   回退到直接加载JSON/JSONL...")
        # 回退到直接加载
        if original_data_path.endswith('.jsonl'):
            original_data = load_jsonl(original_data_path)
        else:
            with open(original_data_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
        print(f"   原始数据条数: {len(original_data)} (直接加载)")
    
    # 加载改写后的数据
    print(f"\n2. 加载改写后的数据: {rewritten_data_path}")
    rewritten_data = load_jsonl(rewritten_data_path)
    print(f"   改写数据条数: {len(rewritten_data)}")
    
    # 统计信息
    print("\n3. 数据统计:")
    original_lengths = [len(item.get('output', '')) for item in original_data[:len(rewritten_data)]]
    rewritten_lengths = [len(item.get('output', '')) for item in rewritten_data]
    
    print(f"   原始输出平均长度: {sum(original_lengths) / len(original_lengths):.1f} 字符")
    print(f"   改写输出平均长度: {sum(rewritten_lengths) / len(rewritten_lengths):.1f} 字符")
    print(f"   长度变化: {((sum(rewritten_lengths) / len(rewritten_lengths)) / (sum(original_lengths) / len(original_lengths)) - 1) * 100:.1f}%")
    
    # 显示示例对比
    print(f"\n4. 示例对比 (显示前 {num_examples} 个):")
    print("=" * 80)
    
    for i in range(min(num_examples, len(rewritten_data))):
        print(f"\n【示例 {i+1}】")
        print("-" * 80)
        
        # 获取原始数据（必须通过input匹配，不能使用索引，因为改写数据是筛选后的子集）
        original_item = None
        rewritten_item = rewritten_data[i]
        rewritten_input = rewritten_item.get('input', '')
        
        # 尝试找到对应的原始数据（精确匹配或去除空白后匹配）
        for orig in original_data:
            orig_input = orig.get('input', '')
            if orig_input == rewritten_input or orig_input.strip() == rewritten_input.strip():
                original_item = orig
                break
        
        if original_item is None:
            print(f"   ⚠️  无法找到对应的原始数据（input不匹配）")
            print(f"   改写数据的input: {rewritten_input[:200]}...")
            continue
        
        if original_item:
            print(f"📝 输入 (Input):")
            print(f"   {rewritten_input[:200]}{'...' if len(rewritten_input) > 200 else ''}")
            
            print(f"\n📄 原始输出 (Original Output):")
            original_output = original_item.get('output', '')
            print(f"   {original_output[:300]}{'...' if len(original_output) > 300 else ''}")
            print(f"   [长度: {len(original_output)} 字符]")
            
            print(f"\n✨ 改写输出 (Rewritten Output):")
            rewritten_output = rewritten_item.get('output', '')
            print(f"   {rewritten_output[:300]}{'...' if len(rewritten_output) > 300 else ''}")
            print(f"   [长度: {len(rewritten_output)} 字符]")
            
            # 分析改写质量
            print(f"\n📊 改写分析:")
            if len(rewritten_output) > len(original_output) * 1.5:
                print("   ⚠️  改写后长度显著增加（可能过度扩展）")
            elif len(rewritten_output) < len(original_output) * 0.5:
                print("   ⚠️  改写后长度显著减少（可能过度压缩）")
            else:
                print("   ✅ 长度变化合理")
            
            # 检查是否有重复
            if rewritten_output.count(rewritten_output[:50]) > 1:
                print("   ⚠️  检测到可能的重复内容")
            
            # 检查是否包含原始内容
            if original_output[:100] in rewritten_output or rewritten_output[:100] in original_output:
                print("   ℹ️  改写输出与原始输出有部分重叠")
            else:
                print("   ✅ 改写输出与原始输出有明显差异（可能是好的改写）")
        else:
            print("   ⚠️  无法找到对应的原始数据")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python analyze_rewritten_data.py <原始数据路径> <改写数据路径> [示例数量]")
        print("\n示例:")
        print("  python analyze_rewritten_data.py training_data/training_data_100k.json output/self_adaptive_sft/round_0/rewritten_dataset.jsonl 10")
        sys.exit(1)
    
    original_path = sys.argv[1]
    rewritten_path = sys.argv[2]
    num_examples = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    if not os.path.exists(original_path):
        print(f"错误: 原始数据文件不存在: {original_path}")
        sys.exit(1)
    
    if not os.path.exists(rewritten_path):
        print(f"错误: 改写数据文件不存在: {rewritten_path}")
        sys.exit(1)
    
    analyze_rewritten_data(original_path, rewritten_path, num_examples)
