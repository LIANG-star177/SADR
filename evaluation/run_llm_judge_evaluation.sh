#!/bin/bash
# LLM as Judge 评测脚本
# 1. 使用qwen3-max API推理MT-bench和PKU测试集（作为基准，可复用）
# 2. 使用本地训练的模型推理这两个测试集
# 3. 使用DeepSeek R1作为judge比较两个模型，计算win_rate（默认 deepseek/deepseek-r1-0528:free，也可使用GPT-4o）
#    通过 OpenRouter API 调用（base_url: https://openrouter.ai/api/v1）
#
# 使用方法：
#   bash evaluation/run_llm_judge_evaluation.sh [模型路径] [judge模型] [tensor_parallel_size] [gpu_memory_utilization] [batch_size] [--small] [--skip-baseline] [--skip-inference] [--only-judge]
#
# 参数说明：
#   tensor_parallel_size: VLLM tensor parallel size，默认 1
#   gpu_memory_utilization: VLLM GPU 内存利用率，默认 0.9
#   batch_size: VLLM 批量大小，默认 32
#   --skip-baseline: 跳过qwen3-max基准推理（如果基准结果已存在）
#   --skip-inference: 跳过本地模型推理（如果本地模型结果已存在）
#   --only-judge: 只运行judge评测（如果所有预测文件都已存在）

set -e

# 设置项目根目录
PROJECT_ROOT="/home/u12321044/share/liang_52/align_tax"
cd "$PROJECT_ROOT"

# 设置 OPENAI_API_KEY
export OPENAI_API_KEY="***

# 解析参数
USE_SMALL_TEST=""          # 是否使用小测试集
SKIP_BASELINE=false       # 是否跳过基准推理
SKIP_INFERENCE=false      # 是否跳过本地模型推理
ONLY_JUDGE=false          # 是否只运行judge

# 先解析可选参数，并收集位置参数
POSITIONAL_ARGS=()
for arg in "$@"; do
    case $arg in
        --small)
            USE_SMALL_TEST="--small"
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            ;;
        --skip-inference)
            SKIP_INFERENCE=true
            ;;
        --only-judge)
            ONLY_JUDGE=true
            ;;
        *)
            # 不是可选参数，作为位置参数
            POSITIONAL_ARGS+=("$arg")
            ;;
    esac
done

# 从位置参数中提取模型路径、judge模型和VLLM参数
MODEL_PATH="${POSITIONAL_ARGS[0]:-output/self_adaptive_sft/round_0/main_model_round_0}"  # 第一个位置参数为本地模型路径
JUDGE_MODEL="${POSITIONAL_ARGS[1]:-deepseek/deepseek-r1-0528:free}"  # 第二个位置参数为judge模型，默认 deepseek/deepseek-r1-0528:free（免费版本，也可使用 gpt-4o）
TENSOR_PARALLEL_SIZE="${POSITIONAL_ARGS[2]:-1}"  # 第三个位置参数为 tensor parallel size，默认 1
GPU_MEMORY_UTILIZATION="${POSITIONAL_ARGS[3]:-0.9}"  # 第四个位置参数为 GPU 内存利用率，默认 0.9
BATCH_SIZE="${POSITIONAL_ARGS[4]:-32}"  # 第五个位置参数为批量大小，默认 32

echo "=========================================="
echo "LLM as Judge 评测流程（使用 VLLM 推理）"
echo "=========================================="
echo "本地模型路径: $MODEL_PATH"
echo "Judge模型: $JUDGE_MODEL"
echo "VLLM Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "VLLM GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "VLLM Batch Size: $BATCH_SIZE"
if [ "$USE_SMALL_TEST" == "--small" ]; then
    echo "将使用小测试集（10%采样）"
fi
if [ "$SKIP_BASELINE" == true ]; then
    echo "将跳过qwen3-max基准推理（使用已有结果）"
fi
if [ "$SKIP_INFERENCE" == true ]; then
    echo "将跳过本地模型推理（使用已有结果）"
fi
if [ "$ONLY_JUDGE" == true ]; then
    echo "只运行judge评测"
fi
echo "=========================================="

# 创建预测结果目录
mkdir -p predictions

# 检查OPENAI_API_KEY（用于 OpenRouter API）
if [ -z "$OPENAI_API_KEY" ]; then
    echo "警告: 未设置 OPENAI_API_KEY 环境变量（OpenRouter API Key），judge功能可能无法使用"
fi

# ============================================
# 第一部分：使用qwen3-max API推理（基准，可复用）
# ============================================
if [ "$ONLY_JUDGE" != true ]; then
    echo ""
    echo "=========================================="
    echo "第一部分：使用 qwen3-max API 推理（基准）"
    echo "=========================================="
    echo "注意: qwen3-max的结果作为基准，可以复用，无需重复推理"
fi

# 1. MT-Bench
if [ "$ONLY_JUDGE" != true ] && [ "$SKIP_BASELINE" != true ]; then
    echo ""
    echo ">>> [1/2] 推理 MT-Bench (qwen3-max 基准)..."
    if [ -f "predictions/mt_bench_api_preds.jsonl" ]; then
        echo "  ✓ 检测到已存在的基准文件，跳过推理"
        echo "  文件: predictions/mt_bench_api_preds.jsonl"
        echo "  （基准结果可复用，无需重复推理）"
    else
        echo "  生成qwen3-max基准结果..."
        python evaluation/inference_on_testsets.py \
            --testset mt_bench \
            --test_file tests/mt_bench/mt_bench_questions.jsonl \
            --output_file predictions/mt_bench_api_preds.jsonl \
            --model qwen3-max \
            --tasktype qwen \
            --requests_per_minite 10 || {
            echo "错误: MT-Bench API推理失败"
            exit 1
        }
        echo "  ✓ qwen3-max基准结果已生成"
    fi
fi

# 2. PKU-SafeRLHF
if [ "$ONLY_JUDGE" != true ] && [ "$SKIP_BASELINE" != true ]; then
    echo ""
    echo ">>> [2/2] 推理 PKU-SafeRLHF (qwen3-max 基准)..."
    if [ -f "predictions/pku_api_preds.jsonl" ]; then
        echo "  ✓ 检测到已存在的基准文件，跳过推理"
        echo "  文件: predictions/pku_api_preds.jsonl"
        echo "  （基准结果可复用，无需重复推理）"
    else
        if [ "$USE_SMALL_TEST" == "--small" ] && [ -f "tests/pku_saferlhf/pku_benchmark_small.jsonl" ]; then
            TEST_FILE="tests/pku_saferlhf/pku_benchmark_small.jsonl"
            echo "  使用小测试集: $TEST_FILE"
        else
            TEST_FILE="tests/pku_saferlhf/pku_benchmark.jsonl"
            echo "  使用完整测试集: $TEST_FILE"
        fi
        echo "  生成qwen3-max基准结果..."
        python evaluation/inference_on_testsets.py \
            --testset pku_saferlhf \
            --test_file "$TEST_FILE" \
            --output_file predictions/pku_api_preds.jsonl \
            --model qwen3-max \
            --tasktype qwen \
            --requests_per_minite 10 || {
            echo "错误: PKU-SafeRLHF API推理失败"
            exit 1
        }
        echo "  ✓ qwen3-max基准结果已生成"
    fi
    echo ""
    echo "基准推理完成！"
fi

# ============================================
# 第二部分：使用本地模型推理（与基准对比）
# ============================================
if [ "$ONLY_JUDGE" != true ]; then
    echo ""
    echo "=========================================="
    echo "第二部分：使用本地模型推理（与基准对比）"
    echo "=========================================="
fi

if [ "$ONLY_JUDGE" != true ] && [ "$SKIP_INFERENCE" != true ]; then
    # 检查模型路径
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

    echo "模型标识符: $MODEL_ID"
    echo ""

    # 1. MT-Bench
    echo ""
    echo ">>> [1/2] 推理 MT-Bench (本地模型: $MODEL_PATH, 使用 VLLM)..."
    MT_BENCH_OUTPUT_FILE="predictions/mt_bench_${MODEL_ID}_local_preds.jsonl"
    if [ -f "$MT_BENCH_OUTPUT_FILE" ]; then
        echo "  ✓ 检测到已存在的预测文件，跳过推理"
        echo "  文件: $MT_BENCH_OUTPUT_FILE"
        echo "  提示: 如需重新推理，请删除该文件后重新运行"
    else
        python evaluation/test_local_model_vllm.py \
            --model_path "$MODEL_PATH" \
            --testset mt_bench \
            --test_file tests/mt_bench/mt_bench_questions.jsonl \
            --output_file "$MT_BENCH_OUTPUT_FILE" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --batch_size "$BATCH_SIZE" \
            --max_new_tokens 512 || {
            echo "错误: MT-Bench 本地推理失败"
            exit 1
        }
    fi

    # 2. PKU-SafeRLHF
    echo ""
    echo ">>> [2/2] 推理 PKU-SafeRLHF (本地模型: $MODEL_PATH, 使用 VLLM)..."
    PKU_OUTPUT_FILE="predictions/pku_${MODEL_ID}_local_preds.jsonl"
    if [ -f "$PKU_OUTPUT_FILE" ]; then
        echo "  ✓ 检测到已存在的预测文件，跳过推理"
        echo "  文件: $PKU_OUTPUT_FILE"
        echo "  提示: 如需重新推理，请删除该文件后重新运行"
    else
        if [ "$USE_SMALL_TEST" == "--small" ] && [ -f "tests/pku_saferlhf/pku_benchmark_small.jsonl" ]; then
            TEST_FILE="tests/pku_saferlhf/pku_benchmark_small.jsonl"
            echo "  使用小测试集: $TEST_FILE"
        else
            TEST_FILE="tests/pku_saferlhf/pku_benchmark.jsonl"
            echo "  使用完整测试集: $TEST_FILE"
        fi
        python evaluation/test_local_model_vllm.py \
            --model_path "$MODEL_PATH" \
            --testset pku_saferlhf \
            --test_file "$TEST_FILE" \
            --output_file "$PKU_OUTPUT_FILE" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --batch_size "$BATCH_SIZE" \
            --max_new_tokens 512 || {
            echo "错误: PKU-SafeRLHF 本地推理失败"
            exit 1
        }
    fi

    echo ""
    echo "本地模型推理完成！"
fi

# ============================================
# 第三部分：LLM as Judge 评测
# ============================================
echo ""
echo "=========================================="
echo "第三部分：LLM as Judge 评测"
echo "=========================================="

# 如果已经提取了模型标识符，使用它；否则尝试从模型路径提取
if [ -z "$MODEL_ID" ]; then
    if [[ "$MODEL_PATH" == output/* ]]; then
        RELATIVE_PATH="${MODEL_PATH#output/}"
        MODEL_ID=$(echo "$RELATIVE_PATH" | cut -d'/' -f1)
    else
        MODEL_ID=$(basename "$MODEL_PATH")
    fi
    if [ -z "$MODEL_ID" ] || [ ${#MODEL_ID} -lt 3 ]; then
        MODEL_ID=$(echo "$MODEL_PATH" | md5sum | cut -d' ' -f1 | cut -c1-8)
    fi
fi

# 设置输出文件名（包含模型标识符）
MT_BENCH_OUTPUT_FILE="predictions/mt_bench_${MODEL_ID}_local_preds.jsonl"
PKU_OUTPUT_FILE="predictions/pku_${MODEL_ID}_local_preds.jsonl"

# 检查预测文件是否存在
if [ ! -f "predictions/mt_bench_api_preds.jsonl" ] || [ ! -f "$MT_BENCH_OUTPUT_FILE" ]; then
    echo "错误: MT-Bench 预测文件缺失"
    echo "  基准文件: predictions/mt_bench_api_preds.jsonl"
    echo "  本地模型文件: $MT_BENCH_OUTPUT_FILE"
    exit 1
fi

if [ ! -f "predictions/pku_api_preds.jsonl" ] || [ ! -f "$PKU_OUTPUT_FILE" ]; then
    echo "错误: PKU-SafeRLHF 预测文件缺失"
    echo "  基准文件: predictions/pku_api_preds.jsonl"
    echo "  本地模型文件: $PKU_OUTPUT_FILE"
    exit 1
fi

# 1. MT-Bench Win Rate
echo ""
echo ">>> [1/2] 评测 MT-Bench (Win Rate)..."
python evaluation/eval_llm_judge_mt_bench.py \
    --question_file tests/mt_bench/mt_bench_questions.jsonl \
    --pred_file_baseline predictions/mt_bench_api_preds.jsonl \
    --pred_file_ours "$MT_BENCH_OUTPUT_FILE" \
    --judge_model "$JUDGE_MODEL" || {
    echo "警告: MT-Bench judge评测失败"
}

# 2. PKU-SafeRLHF Win Better & Win Safer
echo ""
echo ">>> [2/2] 评测 PKU-SafeRLHF (Win Better & Win Safer)..."
if [ "$USE_SMALL_TEST" == "--small" ] && [ -f "tests/pku_saferlhf/pku_benchmark_small.jsonl" ]; then
    GOLD_FILE="tests/pku_saferlhf/pku_benchmark_small.jsonl"
else
    GOLD_FILE="tests/pku_saferlhf/pku_benchmark.jsonl"
fi
python evaluation/eval_llm_judge_pku.py \
    --gold_file "$GOLD_FILE" \
    --pred_file_baseline predictions/pku_api_preds.jsonl \
    --pred_file_ours "$PKU_OUTPUT_FILE" \
    --judge_model "$JUDGE_MODEL" || {
    echo "警告: PKU-SafeRLHF judge评测失败"
}

echo ""
echo "=========================================="
echo "LLM as Judge 评测完成！"
echo "=========================================="
echo ""
echo "预测文件位置:"
echo "  - predictions/mt_bench_api_preds.jsonl (qwen3-max)"
if [ -n "$MODEL_ID" ]; then
    echo "  - predictions/mt_bench_${MODEL_ID}_local_preds.jsonl (我们的模型: $MODEL_ID)"
    echo "  - predictions/pku_api_preds.jsonl (qwen3-max)"
    echo "  - predictions/pku_${MODEL_ID}_local_preds.jsonl (我们的模型: $MODEL_ID)"
else
    echo "  - predictions/mt_bench_local_preds.jsonl (我们的模型)"
    echo "  - predictions/pku_api_preds.jsonl (qwen3-max)"
    echo "  - predictions/pku_local_preds.jsonl (我们的模型)"
fi
echo ""

