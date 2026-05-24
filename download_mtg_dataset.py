#!/usr/bin/env python3
"""
下载 MTG 数据集并保存到本地文件夹
使用 streaming 模式避免 schema 问题
"""
from datasets import load_dataset
import json
import os
import time

def download_mtg_dataset(output_dir):
    """
    下载 MTG 数据集并保存为 JSONL 格式
    
    Args:
        output_dir: 输出目录路径
    """
    print("正在下载 MTG 数据集...")
    
    # 使用 streaming 模式加载数据集，避免 schema 问题
    max_retries = 5
    ds = None
    for attempt in range(max_retries):
        try:
            print(f"尝试下载 (第 {attempt + 1}/{max_retries} 次)...")
            # 尝试使用 streaming 模式，并忽略验证
            ds = load_dataset(
                "xychao/MTG", 
                streaming=True,
                verification_mode="no_checks"  # 忽略验证
            )
            print("下载成功！")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 15
                print(f"下载失败: {str(e)[:200]}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"下载失败，已重试 {max_retries} 次")
                raise
    
    if ds is None:
        raise RuntimeError("无法下载数据集")
    
    print(f"数据集信息: {ds}")
    print(f"数据集分割: {list(ds.keys())}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个分割的数据
    for split_name in ds.keys():
        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        print(f"正在保存 {split_name} 到 {output_file}...")
        
        dataset = ds[split_name]
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            try:
                for item in dataset:
                    # 将 item 转换为字典
                    try:
                        if isinstance(item, dict):
                            item_dict = item
                        elif hasattr(item, '__dict__'):
                            item_dict = item.__dict__
                        else:
                            item_dict = dict(item)
                        
                        # 处理嵌套结构
                        cleaned_dict = {}
                        for k, v in item_dict.items():
                            # 跳过 None 值或无法序列化的值
                            if v is not None:
                                try:
                                    json.dumps(v)  # 测试是否可序列化
                                    cleaned_dict[k] = v
                                except (TypeError, ValueError):
                                    # 如果无法序列化，转换为字符串
                                    cleaned_dict[k] = str(v)
                        
                        f.write(json.dumps(cleaned_dict, ensure_ascii=False) + '\n')
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"  已处理 {count} 条数据...")
                    except Exception as e:
                        print(f"  处理第 {count+1} 条数据时出错: {str(e)[:100]}")
                        continue
            except Exception as e:
                print(f"  遍历数据集时出错: {str(e)[:200]}")
                # 继续，至少保存已处理的数据
        
        print(f"已保存 {count} 条数据到 {output_file}")
    
    print("下载完成！")

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "train")
    download_mtg_dataset(output_dir)
