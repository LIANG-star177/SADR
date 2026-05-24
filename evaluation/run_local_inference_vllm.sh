#!/bin/bash
# 使用 VLLM 加速的本地模型推理脚本 - 对所有测试集进行推理
# 使用方法：
#   bash evaluation/run_local_inference_vllm.sh [模型路径] [tensor_parallel_size] [gpu_memory_utilization] [batch_size] [max_new_tokens]
# 示例：
#   bash evaluation/run_local_inference_vllm.sh output/self_adaptive_sft/round_0/main_model_round_0
#   bash evaluation/run_local_inference_vllm.sh output/self_adaptive_sft/round_0/main_model_round_0 1 0.9 32 512

set -e  # 遇到错误立即退出

# 设置项目根目录
PROJECT_ROOT="/home/u12321044/share/liang_52/align_tax"
cd "$PROJECT_ROOT"

# 创建预测结果目录
mkdir -p predictions

# 解析参数
MODEL_PATH="${1:-}"  # 第一个参数为模型路径（必需）
TENSOR_PARALLEL_SIZE="${2:-1}"  # 第二个参数为 tensor parallel size，默认 1
GPU_MEMORY_UTILIZATION="${3:-0.9}"  # 第三个参数为 GPU 内存利用率，默认 0.9
BATCH_SIZE="${4:-32}"  # 第四个参数为批量大小，默认 32
MAX_NEW_TOKENS="${5:-512}"  # 第五个参数为最大生成长度，默认 512

# 检查模型路径
if [ -z "$MODEL_PATH" ]; then
    echo "错误: 请提供模型路径或 HuggingFace 模型名称"
    echo "使用方法: bash evaluation/run_local_inference_vllm.sh <模型路径或HuggingFace模型名> [tensor_parallel_size] [gpu_memory_utilization] [batch_size] [max_new_tokens]"
    echo "示例（本地模型）: bash evaluation/run_local_inference_vllm.sh output/self_adaptive_sft/round_0/main_model_round_0"
    echo "示例（HuggingFace模型）: bash evaluation/run_local_inference_vllm.sh Qwen/Qwen2.5-3B-Instruct"
    exit 1
fi

# 判断是本地路径还是 HuggingFace 模型名称
# HuggingFace 模型名称通常包含 "/" 且不以 "./" 或 "/" 开头，也不是相对路径
IS_HF_MODEL=false
if [[ "$MODEL_PATH" =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+$ ]] || [[ "$MODEL_PATH" =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+/.*$ ]]; then
    # 检查是否不是本地路径（不以 ./ 或 / 开头，且不是 output/ 开头）
    if [[ ! "$MODEL_PATH" =~ ^\./ ]] && [[ ! "$MODEL_PATH" =~ ^/ ]] && [[ ! "$MODEL_PATH" =~ ^output/ ]]; then
        IS_HF_MODEL=true
    fi
fi

# 如果是本地路径，检查是否存在
if [ "$IS_HF_MODEL" = false ] && [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 从模型路径中提取标识符（用于区分不同模型的预测结果）
if [ "$IS_HF_MODEL" = true ]; then
    # HuggingFace 模型名称：将 "/" 替换为 "_"，例如：
    # Qwen/Qwen2.5-3B-Instruct -> Qwen_Qwen2.5-3B-Instruct
    MODEL_ID=$(echo "$MODEL_PATH" | sed 's/\//_/g')
    echo "检测到 HuggingFace 模型: $MODEL_PATH"
elif [[ "$MODEL_PATH" == output/* ]]; then
    # 提取 output 文件夹下一级的文件夹名，例如：
    # output/self_adaptive_sft/round_0/main_model_round_0 -> self_adaptive_sft
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
echo "开始使用 VLLM 进行模型推理（加速版）"
if [ "$IS_HF_MODEL" = true ]; then
    echo "模型类型: HuggingFace 模型"
    echo "模型名称: $MODEL_PATH"
else
    echo "模型类型: 本地模型"
    echo "模型路径: $MODEL_PATH"
fi
echo "模型标识符: $MODEL_ID"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Batch Size: $BATCH_SIZE"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo ""
echo "注意: MT-Bench 和 PKU-SafeRLHF 请使用专门的 LLM as Judge 评测流程"
echo "      bash evaluation/run_llm_judge_evaluation.sh"
echo "=========================================="

# 1. HotpotQA
echo ""
echo ">>> [1/3] 推理 HotpotQA..."
python evaluation/test_local_model_vllm.py \
    --model_path "$MODEL_PATH" \
    --testset hotpotqa \
    --test_file tests/hotpotqa/hotpotqa_test.jsonl \
    --output_file predictions/hotpotqa_${MODEL_ID}_vllm_preds.jsonl \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" || {
    echo "警告: HotpotQA 推理失败，继续执行其他数据集..."
}

# 2. MMLU
echo ""
echo ">>> [2/3] 推理 MMLU..."
python evaluation/test_local_model_vllm.py \
    --model_path "$MODEL_PATH" \
    --testset mmlu \
    --test_file tests/mmlu/mmlu_test.jsonl \
    --output_file predictions/mmlu_${MODEL_ID}_vllm_preds.jsonl \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens 10 || {
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
python evaluation/test_local_model_vllm.py \
    --model_path "$MODEL_PATH" \
    --testset alpaca \
    --test_file "$TEST_FILE" \
    --output_file predictions/alpaca_${MODEL_ID}_vllm_preds.jsonl \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" || {
    echo "警告: Alpaca 推理失败，继续执行..."
}

echo ""
echo "=========================================="
echo "VLLM 本地模型推理完成！"
echo "预测文件保存在 predictions/ 目录（文件名包含模型标识符: $MODEL_ID 和 '_vllm_preds' 后缀）"
echo "=========================================="
echo ""
echo "生成的预测文件："
echo "  - predictions/hotpotqa_${MODEL_ID}_vllm_preds.jsonl"
echo "  - predictions/mmlu_${MODEL_ID}_vllm_preds.jsonl"
echo "  - predictions/alpaca_${MODEL_ID}_vllm_preds.jsonl"
echo ""
echo "注意: MT-Bench 和 PKU-SafeRLHF 请使用专门的 LLM as Judge 评测流程"
echo ""
echo "运行评测脚本："
echo "  bash evaluation/run_evaluation.sh predictions/ tests/ [--skip_mt_bench]"
echo ""
