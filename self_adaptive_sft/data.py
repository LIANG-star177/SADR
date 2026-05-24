import json
from typing import List, Dict, Any


def load_supervised_dataset(path: str) -> List[Dict[str, Any]]:
    """
    加载原始监督数据 D = {(x, y)}。
    
    支持多种数据格式：
    1. JSONL: 每行一个JSON对象
    2. JSON: 整个文件是一个JSON数组
    
    数据字段支持：
    - {"input": str, "output": str}
    - {"question": str, "original_answer": str}
    - {"instruction": str, "input": str, "output": str}  (会将instruction和input合并)
    - {"instruction": str, "output": str}  (input为空时)
    - {"messages": [...]} 或 {"conversations": [...]}  (sharegpt格式，提取user和assistant消息)
    """
    dataset = []
    
    # 判断文件格式
    with open(path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # JSON数组格式
            data_list = json.load(f)
            items = data_list
        else:
            # JSONL格式
            items = []
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))
    
    # 转换数据格式
    for item in items:
        if "input" in item and "output" in item:
            # 标准格式：{"input": str, "output": str}
            input_text = item["input"]
            if "instruction" in item and item["instruction"]:
                # 如果有instruction，将其合并到input前面
                input_text = f"{item['instruction']}\n\n{item['input']}" if input_text else item["instruction"]
            dataset.append({"input": input_text, "output": item["output"]})
        elif "instruction" in item and "output" in item:
            # 只有instruction和output
            input_text = item.get("input", "")
            if input_text:
                full_input = f"{item['instruction']}\n\n{input_text}"
            else:
                full_input = item["instruction"]
            dataset.append({"input": full_input, "output": item["output"]})
        elif "question" in item and "original_answer" in item:
            dataset.append({"input": item["question"], "output": item["original_answer"]})
        elif "messages" in item or "conversations" in item:
            # ShareGPT格式：{"messages": [...]} 或 {"conversations": [...]}
            messages = item.get("messages", item.get("conversations", []))
            if not isinstance(messages, list) or len(messages) == 0:
                continue
            
            # 提取user和assistant消息
            # 支持多种格式：
            # 1. {"role": "user", "content": "..."} 或 {"from": "human", "value": "..."}
            # 2. {"role": "assistant", "content": "..."} 或 {"from": "gpt", "value": "..."}
            user_content = None
            assistant_content = None
            
            for msg in messages:
                # 检查是否是OpenAI格式 (role/content)
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("from", "").lower()
                    content = msg.get("content") or msg.get("value", "")
                    
                    if role in ["user", "human"]:
                        if user_content is None:
                            user_content = content
                        else:
                            user_content += "\n" + content
                    elif role in ["assistant", "gpt"]:
                        if assistant_content is None:
                            assistant_content = content
                        else:
                            assistant_content += "\n" + content
            
            if user_content and assistant_content:
                dataset.append({"input": user_content, "output": assistant_content})
            elif user_content:
                # 只有user消息，没有assistant消息（可能是未完成的对话）
                dataset.append({"input": user_content, "output": ""})
        else:
            raise ValueError(f"无法识别的数据格式。可用字段: {list(item.keys())}, 需要的字段: input/output 或 question/original_answer 或 instruction/output 或 messages/conversations")
    
    return dataset


def save_rewritten_dataset(examples: List[Dict[str, Any]], path: str) -> None:
    """
    保存改写后的数据集到JSONL文件。
    
    格式：每行 {"input": str, "output": str}
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


