#!/usr/bin/env python3
"""
整理测试数据从 Off-Policy-SFT-main/math_evaluation/data 到 data_mtg/test
"""
import os
import shutil
import json

def organize_test_data(source_dir, target_dir):
    """
    整理测试数据
    
    Args:
        source_dir: 源目录路径 (Off-Policy-SFT-main/math_evaluation/data)
        target_dir: 目标目录路径 (data_mtg/test)
    """
    print(f"正在从 {source_dir} 整理测试数据到 {target_dir}...")
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有测试数据集目录
    if not os.path.exists(source_dir):
        print(f"错误: 源目录 {source_dir} 不存在")
        return
    
    # 列出所有子目录
    subdirs = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"找到 {len(subdirs)} 个测试数据集目录")
    
    # 创建汇总信息
    summary = {
        "datasets": [],
        "total_files": 0
    }
    
    # 复制每个数据集
    for subdir in sorted(subdirs):
        source_subdir = os.path.join(source_dir, subdir)
        target_subdir = os.path.join(target_dir, subdir)
        
        # 创建目标子目录
        os.makedirs(target_subdir, exist_ok=True)
        
        # 复制所有文件
        files_copied = []
        for file in os.listdir(source_subdir):
            source_file = os.path.join(source_subdir, file)
            if os.path.isfile(source_file):
                target_file = os.path.join(target_subdir, file)
                shutil.copy2(source_file, target_file)
                files_copied.append(file)
                summary["total_files"] += 1
        
        if files_copied:
            dataset_info = {
                "name": subdir,
                "files": files_copied,
                "file_count": len(files_copied)
            }
            summary["datasets"].append(dataset_info)
            print(f"  已复制 {subdir}: {len(files_copied)} 个文件")
    
    # 保存汇总信息
    summary_file = os.path.join(target_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n整理完成！")
    print(f"  共整理 {len(summary['datasets'])} 个数据集")
    print(f"  共复制 {summary['total_files']} 个文件")
    print(f"  汇总信息已保存到 {summary_file}")

if __name__ == "__main__":
    project_root = "/home/u12321044/share/liang_52/align_tax"
    source_dir = os.path.join(project_root, "Off-Policy-SFT-main/math_evaluation/data")
    target_dir = os.path.join(project_root, "data_mtg/test")
    
    organize_test_data(source_dir, target_dir)
