#!/bin/bash
# 本地模型推理脚本 - 对所有测试集进行推理
# 使用方法：
#   bash evaluation/run_local_inference.sh [模型路径] [设备] [最大输入长度] [最大生成长度]
# 示例：
#   bash evaluation/run_local_inference.sh output/self_adaptive_sft/round_0/main_model_round_0
#   bash evaluation/run_local_inference.sh output/self_adaptive_sft/round_0/main_model_round_0 cuda 2048 512

set -e  # 遇到错误立即退出

# 设置项目根目录
PROJECT_ROOT="/home/u12321044/share/liang_52/align_tax"
cd "$PROJECT_ROOT"

# 创建预测结果目录
mkdir -p predictions

# 解析参数
MODEL_PATH="${1:-}"  # 第一个参数为模型路径（必需）
DEVICE="${2:-cuda}"  # 第二个参数为设备，默认 cuda
MAX_LENGTH="${3:-2048}"  # 第三个参数为最大输入长度，默认 2048
MAX_NEW_TOKENS="${4:-512}"  # 第四个参数为最大生成长度，默认 512

# 检查模型路径
if [ -z "$MODEL_PATH" ]; then
    echo "错误: 请提供模型路径"
    echo "使用方法: bash evaluation/run_local_inference.sh <模型路径> [设备] [最大输入长度] [最大生成长度]"
    echo "示例: bash evaluation/run_local_inference.sh output/self_adaptive_sft/round_0/main_model_round_0"
    exit 1
fi

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 从模型路径中提取标识符（用于区分不同模型的预测结果）
# 提取 output 文件夹下一级的文件夹名，例如：
# output/self_adaptive_sft/round_0/main_model_round_0 -> self_adaptive_sft
# output/self_adaptive_sft/round_1/main_model_round_1 -> self_adaptive_sft
if [[ "$MODEL_PATH" == output/* ]]; then
    # 移除 output/ 前缀，然后提取第一部分
    RELATIVE_PATH="${MODEL_PATH#output/}"
    MODEL_ID=$(echo "$RELATIVE_PATH" | cut -d'/' -f1)
else
    # 如果路径不是以 output/ 开头，使用 basename 作为后备方案
    MODEL_ID=$(basename "$MODEL_PATH")
fi
# 如果标识符为空或太短，使用整个路径的哈希值
if [ -z "$MODEL_ID" ] || [ ${#MODEL_ID} -lt 3 ]; then
    MODEL_ID=$(echo "$MODEL_PATH" | md5sum | cut -d' ' -f1 | cut -c1-8)
    echo "警告: 模型路径标识符为空，使用哈希值: $MODEL_ID"
fi

echo "=========================================="
echo "开始本地模型推理"
echo "模型路径: $MODEL_PATH"
echo "模型标识符: $MODEL_ID"
echo "设备: $DEVICE"
echo "最大输入长度: $MAX_LENGTH"
echo "最大生成长度: $MAX_NEW_TOKENS"
echo ""
echo "注意: MT-Bench 和 PKU-SafeRLHF 请使用专门的 LLM as Judge 评测流程"
echo "      bash evaluation/run_llm_judge_evaluation.sh"
echo "=========================================="

# 1. HotpotQA
echo ""
echo ">>> [1/3] 推理 HotpotQA..."
python evaluation/test_local_model.py \
    --model_path "$MODEL_PATH" \
    --testset hotpotqa \
    --test_file tests/hotpotqa/hotpotqa_test.jsonl \
    --output_file predictions/hotpotqa_${MODEL_ID}_local_preds.jsonl \
    --device "$DEVICE" \
    --max_length "$MAX_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" || {
    echo "警告: HotpotQA 推理失败，继续执行其他数据集..."
}

# 2. MMLU
echo ""
echo ">>> [2/3] 推理 MMLU..."
python evaluation/test_local_model.py \
    --model_path "$MODEL_PATH" \
    --testset mmlu \
    --test_file tests/mmlu/mmlu_test.jsonl \
    --output_file predictions/mmlu_${MODEL_ID}_local_preds.jsonl \
    --device "$DEVICE" \
    --max_length "$MAX_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" || {
    echo "警告: MMLU 推理失败，继续执行其他数据集..."
}

# 3. Alpaca（优先使用小测试集）
echo ""
echo ">>> [3/3] 推理 Alpaca..."
# 检查是否存在小测试集
if [ -f "tests/alpaca/alpaca_robust_test_small.jsonl" ]; then
    TEST_FILE="tests/alpaca/alpaca_robust_test_small.jsonl"
    echo "  使用小测试集: $TEST_FILE"
else
    TEST_FILE="tests/alpaca/alpaca_robust_test.jsonl"
    echo "  使用完整测试集: $TEST_FILE"
fi
python evaluation/test_local_model.py \
    --model_path "$MODEL_PATH" \
    --testset alpaca \
    --test_file "$TEST_FILE" \
    --output_file predictions/alpaca_${MODEL_ID}_local_preds.jsonl \
    --device "$DEVICE" \
    --max_length "$MAX_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" || {
    echo "警告: Alpaca 推理失败，继续执行..."
}

echo ""
echo "=========================================="
echo "本地模型推理完成！"
echo "预测文件保存在 predictions/ 目录（文件名包含模型标识符: $MODEL_ID）"
echo "=========================================="
echo ""
echo "生成的预测文件："
echo "  - predictions/hotpotqa_${MODEL_ID}_local_preds.jsonl"
echo "  - predictions/mmlu_${MODEL_ID}_local_preds.jsonl"
echo "  - predictions/alpaca_${MODEL_ID}_local_preds.jsonl"
echo ""
echo "注意: MT-Bench 和 PKU-SafeRLHF 请使用专门的 LLM as Judge 评测流程"
echo ""
echo "运行评测脚本："
echo "  bash evaluation/run_evaluation.sh predictions/ tests/ [--skip_mt_bench]"
echo ""

