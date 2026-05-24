#!/usr/bin/env python3
"""
比较 MTG 训练数据和现有训练数据格式
"""
import json

def load_jsonl(file_path, n=3):
    """加载 JSONL 文件的前 n 条数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data.append(json.loads(line))
    return data

def load_json(file_path, n=3):
    """加载 JSON 文件的前 n 条数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return data[:n]
        elif isinstance(data, dict):
            # 如果是字典，尝试找到列表字段
            for key in data:
                if isinstance(data[key], list):
                    return data[key][:n]
            return [data]
        return [data]

def print_sample(title, sample, idx=0):
    """打印样本信息"""
    print(f"\n{'='*80}")
    print(f"{title} - 样本 {idx+1}")
    print('='*80)
    print(f"字段列表: {list(sample.keys())}")
    print(f"\n完整内容:")
    print(json.dumps(sample, indent=2, ensure_ascii=False))

def analyze_structure(data, name):
    """分析数据结构"""
    print(f"\n{'='*80}")
    print(f"{name} 数据结构分析")
    print('='*80)
    
    if len(data) == 0:
        print("数据为空")
        return
    
    # 分析第一个样本
    first_sample = data[0]
    print(f"\n总样本数: {len(data)}")
    print(f"第一个样本的字段: {list(first_sample.keys())}")
    
    # 检查关键字段
    key_fields = ['messages', 'instruction', 'input', 'output', 'question', 'answer', 'prompt', 'response']
    found_fields = {field: field in first_sample for field in key_fields}
    print(f"\n关键字段检查:")
    for field, exists in found_fields.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {field}: {exists}")
    
    # 显示第一个样本的完整内容（截断）
    print(f"\n第一个样本内容（前2000字符）:")
    sample_str = json.dumps(first_sample, indent=2, ensure_ascii=False)
    print(sample_str[:2000])
    if len(sample_str) > 2000:
        print("... (内容已截断)")

def main():
    # 读取 MTG 训练数据
    print("\n" + "="*80)
    print("读取 MTG 训练数据 (data_mtg/train/train.jsonl)")
    print("="*80)
    try:
        mtg_data = load_jsonl('data_mtg/train/train.jsonl', n=3)
        print(f"✓ 成功读取 {len(mtg_data)} 条数据")
        analyze_structure(mtg_data, "MTG 训练数据")
        
        # 显示前3条样本
        for i, sample in enumerate(mtg_data):
            print_sample("MTG 训练数据", sample, i)
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 读取现有训练数据
    print("\n\n" + "="*80)
    print("读取现有训练数据 (training_data/training_data_100k.json)")
    print("="*80)
    try:
        existing_data = load_json('training_data/training_data_100k.json', n=3)
        print(f"✓ 成功读取 {len(existing_data)} 条数据")
        analyze_structure(existing_data, "现有训练数据")
        
        # 显示前3条样本
        for i, sample in enumerate(existing_data):
            print_sample("现有训练数据", sample, i)
    except Exception as e:
        print(f"✗ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 比较格式
    print("\n\n" + "="*80)
    print("格式比较")
    print("="*80)
    
    if len(mtg_data) > 0 and len(existing_data) > 0:
        mtg_keys = set(mtg_data[0].keys())
        existing_keys = set(existing_data[0].keys())
        
        print(f"\nMTG 数据字段: {mtg_keys}")
        print(f"现有数据字段: {existing_keys}")
        
        common_keys = mtg_keys & existing_keys
        mtg_only = mtg_keys - existing_keys
        existing_only = existing_keys - mtg_keys
        
        print(f"\n共同字段: {common_keys}")
        print(f"仅 MTG 有: {mtg_only}")
        print(f"仅现有数据有: {existing_only}")
        
        # 检查关键字段
        if 'messages' in common_keys:
            print("\n✓ 两者都有 'messages' 字段，格式可能兼容")
        elif 'messages' in mtg_keys and 'messages' not in existing_keys:
            print("\n⚠ MTG 有 'messages' 字段，但现有数据没有")
        elif 'messages' in existing_keys and 'messages' not in mtg_keys:
            print("\n⚠ 现有数据有 'messages' 字段，但 MTG 没有")
        
        # 检查 messages 格式
        if 'messages' in mtg_data[0]:
            print("\nMTG messages 格式示例:")
            print(json.dumps(mtg_data[0]['messages'], indent=2, ensure_ascii=False)[:500])
        
        if 'messages' in existing_data[0]:
            print("\n现有数据 messages 格式示例:")
            print(json.dumps(existing_data[0]['messages'], indent=2, ensure_ascii=False)[:500])

if __name__ == "__main__":
    main()
