#!/bin/bash
# 创建小测试集脚本 - 为alpaca和pku数据集创建10%采样的小测试集
# 使用方法：
#   bash evaluation/create_small_testsets.sh

set -e

# 设置项目根目录
PROJECT_ROOT="/home/u12321044/share/liang_52/align_tax"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "创建小测试集（10%采样）"
echo "=========================================="

# 1. Alpaca小测试集
echo ""
echo ">>> 创建 Alpaca 小测试集..."
python evaluation/create_small_testset.py \
    --input_file tests/alpaca/alpaca_robust_test.jsonl \
    --output_file tests/alpaca/alpaca_robust_test_small.jsonl \
    --sample_ratio 0.1 \
    --seed 42

# 2. PKU-SafeRLHF小测试集
echo ""
echo ">>> 创建 PKU-SafeRLHF 小测试集..."
python evaluation/create_small_testset.py \
    --input_file tests/pku_saferlhf/pku_benchmark.jsonl \
    --output_file tests/pku_saferlhf/pku_benchmark_small.jsonl \
    --sample_ratio 0.1 \
    --seed 42

echo ""
echo "=========================================="
echo "小测试集创建完成！"
echo "=========================================="
echo ""
echo "生成的文件："
echo "  - tests/alpaca/alpaca_robust_test_small.jsonl"
echo "  - tests/pku_saferlhf/pku_benchmark_small.jsonl"
echo ""
echo "使用小测试集进行推理："
echo "  bash evaluation/run_local_inference.sh <模型路径> cuda 2048 512"
echo "  (脚本会自动检测并使用小测试集)"
echo ""

