from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import jsonlines
from tqdm import tqdm
from evaluate import load
from datasets import load_dataset
from vllm import LLM, SamplingParams


# 加载Qwen2.5-1.5B-Instruct模型
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def robust_translate(text, src_lang, tgt_lang, max_retry=3):
    """增加容错机制的翻译函数"""
    prompt = f"保持数学符号和数字不变，仅转换语言风格：将{src_lang}翻译为{tgt_lang}：\n{text}"
    # 不要缩短计算步骤，完全保持原有计算过程
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    for _ in range(max_retry):
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 检查关键元素是否保留
        if all(num in response for num in re.findall(r'\d+\.?\d*', text)):
            return response
    return response  # 超过重试次数返回最后结果

def align_style_via_translation(original_answer_en):
    # 英→中翻译
    chinese_version = robust_translate(original_answer_en, "英文", "中文")
    # 中→英翻译（风格对齐）
    aligned_english = robust_translate(chinese_version, "中文", "英文")
    return aligned_english

bertscore = load("bertscore")

def check_semantic_equivalence(original, translated):
    """通过BERTScore评估语义一致性"""
    results = bertscore.compute(
        predictions=translated,
        references=original,
        model_type="distilbert-base-uncased"
    )
    return results  # F1分数>0.9视为安全

# 5. 结果分析
aligned_answers = []
dataset = load_dataset("openai/gsm8k", "main")
# samples = dataset['train']
# for answer_en in samples['answer']:
#     aligned_english = align_style_via_translation(answer_en)
#     aligned_answers.append(aligned_english)

# scores = check_semantic_equivalence(samples['answer'], aligned_answers)

# for i in range(10):
#     print(f"Sample {i+1}:")
#     print(f"Original Answer: {samples['answer'][i]}")
#     print(f"Aligned Answer: {aligned_answers[i]}")
#     print(f"BERTScore F1: {scores['f1'][i]:.2f}")
#     print("-"*40)

with jsonlines.open('gsm8k_aligned_answers.jsonl', mode='w') as writer:
    for idx in tqdm(range(len(dataset['train'])), desc="Processing samples"):
        sample = dataset['train'][idx]
        original_answer = sample['answer']
        
        try:
            # 执行翻译对齐
            aligned_answer = align_style_via_translation(original_answer)
            
            # 计算语义相似度
            score = bertscore.compute(
                predictions=[aligned_answer],
                references=[original_answer],
                model_type="distilbert-base-uncased"
            )['f1'][0]
            
            # 构建数据字典
            data = {
                "question": sample['question'],
                "original_answer": original_answer,
                "aligned_answer": aligned_answer,
                "bertscore_f1": round(score, 4)
            }
            
            # 写入文件
            writer.write(data)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # 记录错误信息
            error_data = {
                "question": sample['question'],
                "original_answer": original_answer,
                "error": str(e)
            }
            writer.write(error_data)