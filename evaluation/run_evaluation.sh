#!/bin/bash
# 评测脚本 - 根据推理结果评测指标
# 使用方法：
#   方式1（推荐）：直接输入模型标识符
#     bash evaluation/run_evaluation.sh [模型标识符]
#   方式2：完整参数
#     bash evaluation/run_evaluation.sh [预测目录] [测试集目录] [模型标识符] [是否跳过MT-Bench]
# 示例：
#   bash evaluation/run_evaluation.sh DR_3b_vllm
#   bash evaluation/run_evaluation.sh predictions/ tests/
#   bash evaluation/run_evaluation.sh predictions/ tests/ DR_3b_vllm
#   bash evaluation/run_evaluation.sh predictions/ tests/ DR_3b_vllm --skip_mt_bench

set -e  # 遇到错误立即退出

# 设置项目根目录
PROJECT_ROOT="/home/u12321044/share/liang_52/align_tax"
cd "$PROJECT_ROOT"

# 解析参数
# 智能识别参数：如果第一个参数看起来不像目录路径，则视为模型标识符
if [ $# -gt 0 ] && [[ ! "$1" =~ ^(predictions|tests|\./|/).* ]] && [ "$1" != "--skip_mt_bench" ]; then
    # 第一个参数是模型标识符
    MODEL_ID="$1"
    PREDICTIONS_DIR="${2:-predictions}"  # 第二个参数为预测目录，默认 predictions/
    TESTSETS_DIR="${3:-tests}"            # 第三个参数为测试集目录，默认 tests/
    SKIP_MT_BENCH="${4:-}"                # 第四个参数，如果为 --skip_mt_bench 则跳过MT-Bench评测
else
    # 标准用法：第一个参数是预测目录
    PREDICTIONS_DIR="${1:-predictions}"  # 第一个参数为预测目录，默认 predictions/
    TESTSETS_DIR="${2:-tests}"            # 第二个参数为测试集目录，默认 tests/
    MODEL_ID="${3:-}"                     # 第三个参数为模型标识符（可选），如 DR_3b_vllm
    SKIP_MT_BENCH="${4:-}"                # 第四个参数，如果为 --skip_mt_bench 则跳过MT-Bench评测
fi

# 检查参数是否是 --skip_mt_bench（兼容旧用法）
if [ "$MODEL_ID" == "--skip_mt_bench" ]; then
    SKIP_MT_BENCH="--skip_mt_bench"
    MODEL_ID=""
fi
if [ "$TESTSETS_DIR" == "--skip_mt_bench" ]; then
    SKIP_MT_BENCH="--skip_mt_bench"
    TESTSETS_DIR="tests"
fi

echo "=========================================="
echo "开始评测所有数据集"
echo "预测目录: $PREDICTIONS_DIR"
echo "测试集目录: $TESTSETS_DIR"
if [ -n "$MODEL_ID" ]; then
    echo "模型标识符: $MODEL_ID"
fi
if [ "$SKIP_MT_BENCH" == "--skip_mt_bench" ]; then
    echo "将跳过 MT-Bench 评测（需要 GPT-4 judge）"
fi
echo "=========================================="

# 构建评测命令
EVAL_CMD="python evaluation/run_all_evaluations.py --predictions_dir $PREDICTIONS_DIR --testsets_dir $TESTSETS_DIR"
if [ -n "$MODEL_ID" ]; then
    EVAL_CMD="$EVAL_CMD --model_id $MODEL_ID"
fi
if [ "$SKIP_MT_BENCH" == "--skip_mt_bench" ]; then
    EVAL_CMD="$EVAL_CMD --skip_mt_bench"
fi

echo ""
echo "运行评测命令: $EVAL_CMD"
echo ""

# 运行评测
$EVAL_CMD || {
    echo "警告: 评测过程中出现错误，请检查日志"
    exit 1
}

echo ""
echo "=========================================="
echo "所有评测完成！"
echo "=========================================="
echo ""
echo "评测结果已显示在上方输出中"
echo "如需查看详细结果，请检查各评测脚本的输出"
echo ""

