import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
from tqdm import tqdm
import json
import re
import glob
from pathlib import Path
import pandas as pd
random.seed(2023)
import sys
# 获取上上一级目录的路径
grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
# 将上上一级目录添加到 sys.path 中
sys.path.append(grandparent_dir)
from simple_thread import stable_chat

# 法考题的知识长度需要512,在2个知识下做实验
class testCFG:
    prompt_dic = {
    "query":("问题：{query}\n")}
    data_path = "/home/u12321044/share/liang_52/art-rag/baseline/contriever/test.json"
    test_num = 128
    curr_prompt = ""
    # Excel 路径与列名配置（按用户指定的文件与列名）
    input_file = "/home/u12321044/share/liang_52/LLM_API/测试题(2).xls"
    question_col = "题目"
    answer_col = "QWEN回答"

config = testCFG()

def get_label_law(text):
    pattern = r"^(.*?)(?=：)"
    match = re.search(pattern, text)
    if match:
        law_name = match.group(1)
        return law_name

def get_pred_law(text):
    pattern = r"《[^》]+》第[^条]+条|《[^》]+》"
    try:
        matches = re.findall(pattern, text)
        return matches
    except:
        return ["无"]

def test(config, top_k):
    inputs, messages, results, labels=[],[],[],[]
    data_path = config.data_path
    with open(data_path,"r",encoding="utf-8") as f:
        config.curr_prompt = config.prompt_dic["query"]
        lines = f.readlines()
        random.shuffle(lines)
        i=0
        for line in tqdm(lines[:config.test_num]):
            data = json.loads(line) 
            # if not data["fact"].endswith("认定事实如下："):
            #     messages.append({"id":i,'content': config.curr_prompt.replace("{query}", data["fact"]),"label":data["article"]})
            messages.append({"id":i,'content': config.curr_prompt.replace("{query}", data["fact"]),"label":data["article"]})
            i+=1
    result = stable_chat(tasktype="qwen",messages=messages,model="qwen-72b-chat", max_rounds=5, requests_per_minite=5)
    # from train_w_llm.utils.openai_parallel.multiple_request import multiple_request
    # result = multiple_request(tasktype="openai",messages=messages, model = "gpt-4", max_rounds=1, requests_per_minite=10)
    # 使用字典存储第二个字典的数据，方便快速查找
    dict2_lookup = {item["id"]: item for item in result}
    # 遍历第一个字典，进行合并
    docs_lsts = [[] for i in range(len(top_k))]
    eva_labels = []
    for item in messages: 
        answer = dict2_lookup.get(item["id"], {}).get("response")
        answer_lst = get_pred_law(answer)
        print(answer_lst)
        eva_label = [get_pred_law(el)[0] for el in item["label"]]
        # eva_label = item["label"]
        eva_labels.append(eva_label)
        print(eva_label)
        for i in range(len(top_k)):
            docs_lsts[i].append(answer_lst[:top_k[i]])
    score_dict, corr_index = pred_law_metric(docs_lsts, eva_labels, top_k)
    print(score_dict)
def _guess_question_col(columns):
    candidates = ["题目", "问题", "Question", "question", "内容", "content", "text"]
    for name in candidates:
        if name in columns:
            return name
    return columns[0] if len(columns) > 0 else None

def _save_dataframe(df, path_str):
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        with pd.ExcelWriter(path_str, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return str(path)
    elif suffix == ".xls":
        try:
            with pd.ExcelWriter(path_str, engine="xlwt") as writer:
                df.to_excel(writer, index=False)
            return str(path)
        except Exception:
            fallback = path.with_suffix(".xlsx")
            with pd.ExcelWriter(str(fallback), engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            return str(fallback)
    else:
        fallback = path.with_suffix(".xlsx")
        with pd.ExcelWriter(str(fallback), engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return str(fallback)

def run_excel_inference_with_gpt4(config):
    # 仅处理用户提供的固定路径
    files = [getattr(config, "input_file", "").strip()]
    files = [f for f in files if f]
    if len(files) == 0:
        print("未提供有效的 Excel 文件路径。")
        return
    messages = []
    index_to_location = {}
    next_id = 0
    for fpath in files:
        try:
            # 显式指定 .xls 使用 xlrd，引擎缺失会提示安装
            if str(fpath).lower().endswith(".xls"):
                df = pd.read_excel(fpath, sheet_name=0, engine="xlrd")
            else:
                df = pd.read_excel(fpath, sheet_name=0)  # .xlsx 由 openpyxl 处理
        except Exception as e:
            print(f"读取失败，跳过文件：{fpath}，错误：{e}")
            continue
        if df is None or df.shape[0] == 0:
            print(f"空表或无数据，跳过文件：{fpath}")
            continue
        question_col = config.question_col if getattr(config, "question_col", "") else ""
        if not question_col or question_col not in df.columns:
            question_col = _guess_question_col(list(df.columns))
        if question_col is None:
            print(f"无法确定题目列，跳过文件：{fpath}")
            continue
        config.curr_prompt = config.prompt_dic["query"]
        for row_idx, val in enumerate(df[question_col].tolist()):
            if pd.isna(val):
                continue
            content = str(val)
            messages.append({"id": next_id, "content": config.curr_prompt.replace("{query}", content), "label": None})
            index_to_location[next_id] = (fpath, row_idx)
            next_id += 1
    if len(messages) == 0:
        print("未构建到有效问题，结束。")
        return
    result = stable_chat(tasktype="qwen", messages=messages, model="qwen-turbo", max_rounds=5, requests_per_minite=5)
    id_to_resp = {item["id"]: item.get("response", "") for item in result}
    files_touched = set([loc[0] for loc in index_to_location.values()])
    for fpath in files_touched:
        try:
            if str(fpath).lower().endswith(".xls"):
                df = pd.read_excel(fpath, sheet_name=0, engine="xlrd")
            else:
                df = pd.read_excel(fpath, sheet_name=0)
        except Exception as e:
            print(f"重读失败，跳过写回：{fpath}，错误：{e}")
            continue
        answer_col = getattr(config, "answer_col", "GPT4答案") or "GPT4答案"
        if answer_col not in df.columns:
            df[answer_col] = None
        for _id, (fp, row_idx) in index_to_location.items():
            if fp != fpath:
                continue
            df.at[row_idx, answer_col] = id_to_resp.get(_id, "")
        out_path = _save_dataframe(df, fpath)
        print(f"已写回：{out_path}")

run_excel_inference_with_gpt4(config)