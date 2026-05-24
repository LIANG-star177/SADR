import re
from tqdm import tqdm
import threading
import json
from openai import OpenAI
import dashscope
from dashscope import Generation

def openai_res(query, model="gpt-3.5"):
    client = OpenAI(api_key="sk-RA5dsNrMxVPvtxCfA2B5587c4c6044A492E330Dc8cF41f4d", base_url="https://api.xeduapi.com")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    # 兼容不同代理/SDK的返回结构
    try:
        if hasattr(completion, "choices"):
            return completion.choices[0].message.content
        if isinstance(completion, dict):
            if "choices" in completion:
                first = completion["choices"][0]
                if isinstance(first, dict):
                    if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                        return first["message"]["content"]
                    if "text" in first:
                        return first["text"]
        if isinstance(completion, str):
            return completion
    except Exception:
        pass
    return json.dumps(completion, ensure_ascii=False)

def qwen_res(query, model="farui-plus"):
    dashscope.api_key = 'sk-7cb34adf99ff42c3b6e89ef47f8db813'
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': query}]
    response = dashscope.Generation.call(model, messages=messages, result_format='message')
    return response['output']['choices'][0]['message']['content']

mapping = {
    "openai": openai_res,
    # "gemini": GeminiTask,
    "qwen": qwen_res
}

def stable_chat(tasktype, messages, model, max_rounds=1, requests_per_minite=10):
    num_threads = requests_per_minite
    samples = messages
    path = "/home/u12321044/share/liang_52/LLM_API/tmp.json"
    # 清空文件内容
    with open(path, 'w') as f:
        pass  # 不需要写入任何内容，只是为了清空文件
    def wewrite(rank):
        enume_obj = tqdm(enumerate(samples), total=len(samples)) if rank == 0 else enumerate(samples)
        for idx, sample in enume_obj:
            if idx % num_threads != rank:
                continue
            prompt = sample["content"]

            ok = False
            response = None

            for try_idx in range(max_rounds):
                try:
                    response = mapping[tasktype](prompt, model=model)
                    ok = True
                    break
                except Exception as err:
                    print(err)
                    # print(response)
                    if try_idx == 9:
                        print(f'{idx}, {prompt} failed')
                    pass
            if ok:
                res = {
                    "id": sample["id"],
                    "content": sample["content"],
                    "label": sample["label"],
                    "response": response
                }
                with open(path, 'a') as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
  
    threads = []
    for rank in range(num_threads):
        t = threading.Thread(target=wewrite, name="thread-"+str(rank), args=(rank,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    dict_list = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dict_list.append(data)
    return dict_list

# samples = []
# stable_chat(samples, num_threads=20)