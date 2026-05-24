from transformers import AutoTokenizer
import re
import jsonlines
from tqdm import tqdm
from evaluate import load
from datasets import load_dataset
from vllm import LLM, SamplingParams


# 加载数据集
dataset = load_dataset("AI-MO/NuminaMath-TIR", "default")['train']

# 初始化vLLM模型和采样参数
model_name = "Qwen/Qwen2.5-3B-Instruct"
llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.7)
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义增强版翻译函数（使用vLLM）
def robust_translate(texts, src_lang, tgt_lang):
    """批量翻译函数"""
    prompts = [
        f"保持数学符号和数字不变,将以下{src_lang}推理过程翻译为{tgt_lang}：\n{text}"
        for text in texts
    ]
    format_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        format_prompts.append(text)
    print(format_prompts[0])
    outputs = llm.generate(format_prompts, sampling_params)
    print(outputs[0].outputs[0].text)
    translated_texts = [output.outputs[0].text.strip() for output in outputs]
    return translated_texts

def align_style_via_translation(original_answers_en):
    # 批量英→中翻译
    chinese_versions = robust_translate(original_answers_en, "英文", "中文")
    # 批量中→英翻译（风格对齐）
    aligned_english = robust_translate(chinese_versions, "中文", "英文")
    return aligned_english

# 语义一致性验证
bertscore = load("bertscore")

# 保存结果到JSONL文件
aligned_answers = []
batch_size = 24  # 批量大小，可根据硬件调整
with jsonlines.open('Tir_cot_aligned_answers_vllm.jsonl', mode='w') as writer:
    # for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
    #     batch_end = min(batch_start + batch_size, len(dataset))
        # batch_samples = dataset[batch_start:batch_end]
        batch_samples = dataset[:8000]
        original_answers = batch_samples["solution"]
        questions = batch_samples['problem']
        # gsm8k_ans = batch_samples['answer']
        
        try:
            # 批量执行翻译对齐
            aligned_answers_batch = align_style_via_translation(original_answers)
            
            # 批量计算语义相似度
            scores = bertscore.compute(
                predictions=aligned_answers_batch,
                references=original_answers,
                model_type="distilbert-base-uncased"
            )['f1']
            
            # 构建数据字典并写入文件
            for i, (question, original_answer, aligned_answer, score) in enumerate(zip(
                questions, original_answers, aligned_answers_batch, scores
            )):
                data = {
                    "question": question,
                    "original_cot": original_answer,
                    "aligned_cot": aligned_answer,
                    "bertscore_f1": round(score, 4),
                    # "gsm8k_ans": ans
                }
                writer.write(data)
                
        except Exception as e:
            # print(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")
            # 记录错误信息
            for i, (question, original_answer) in enumerate(zip(questions, original_answers)):
                error_data = {
                    "question": question,
                    "original_answer": original_answer,
                    "error": str(e)
                }
                writer.write(error_data)