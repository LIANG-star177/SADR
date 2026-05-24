#!/bin/bash
# API模型推理脚本 - 对所有测试集进行推理
# 使用方法：
#   bash evaluation/run_api_inference.sh [模型名] [接口类型] [并发数]
# 示例：
#   bash evaluation/run_api_inference.sh qwen2.5-turbo qwen 10
#   bash evaluation/run_api_inference.sh gpt-4 openai 5

set -e  # 遇到错误立即退出

# 设置项目根目录
PROJECT_ROOT="/home/u12321044/share/liang_52/align_tax"
cd "$PROJECT_ROOT"

# 创建预测结果目录
mkdir -p predictions

# 解析参数
MODEL="${1:-qwen2.5-turbo}"  # 第一个参数为模型名，默认 qwen2.5-turbo
TASKTYPE="${2:-qwen}"         # 第二个参数为接口类型，默认 qwen
REQUESTS_PER_MINUTE="${3:-10}" # 第三个参数为并发数，默认 10

echo "=========================================="
echo "开始API模型推理"
echo "模型: $MODEL"
echo "接口类型: $TASKTYPE"
echo "并发数: $REQUESTS_PER_MINUTE"
echo ""
echo "注意: MT-Bench 和 PKU-SafeRLHF 请使用专门的 LLM as Judge 评测流程"
echo "      bash evaluation/run_llm_judge_evaluation.sh"
echo "=========================================="

# 1. HotpotQA
echo ""
echo ">>> [1/3] 推理 HotpotQA..."
python evaluation/inference_on_testsets.py \
    --testset hotpotqa \
    --test_file tests/hotpotqa/hotpotqa_test.jsonl \
    --output_file predictions/hotpotqa_api_preds.jsonl \
    --model "$MODEL" \
    --tasktype "$TASKTYPE" \
    --requests_per_minite "$REQUESTS_PER_MINUTE" || {
    echo "警告: HotpotQA 推理失败，继续执行其他数据集..."
}

# 2. MMLU
echo ""
echo ">>> [2/3] 推理 MMLU..."
python evaluation/inference_on_testsets.py \
    --testset mmlu \
    --test_file tests/mmlu/mmlu_test.jsonl \
    --output_file predictions/mmlu_api_preds.jsonl \
    --model "$MODEL" \
    --tasktype "$TASKTYPE" \
    --requests_per_minite "$REQUESTS_PER_MINUTE" || {
    echo "警告: MMLU 推理失败，继续执行其他数据集..."
}

# 3. Alpaca
echo ""
echo ">>> [3/3] 推理 Alpaca..."
python evaluation/inference_on_testsets.py \
    --testset alpaca \
    --test_file tests/alpaca/alpaca_robust_test.jsonl \
    --output_file predictions/alpaca_api_preds.jsonl \
    --model "$MODEL" \
    --tasktype "$TASKTYPE" \
    --requests_per_minite "$REQUESTS_PER_MINUTE" || {
    echo "警告: Alpaca 推理失败，继续执行..."
}

echo ""
echo "=========================================="
echo "API模型推理完成！"
echo "预测文件保存在 predictions/ 目录（文件名包含 '_api_preds' 后缀）"
echo "=========================================="
echo ""
echo "生成的预测文件："
echo "  - predictions/hotpotqa_api_preds.jsonl"
echo "  - predictions/mmlu_api_preds.jsonl"
echo "  - predictions/alpaca_api_preds.jsonl"
echo ""
echo "注意: MT-Bench 和 PKU-SafeRLHF 请使用专门的 LLM as Judge 评测流程"
echo ""
echo "运行评测脚本："
echo "  bash evaluation/run_evaluation.sh predictions/ tests/ [--skip_mt_bench]"
echo ""

