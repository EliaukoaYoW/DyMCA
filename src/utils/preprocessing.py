import preprocessor as p
import re
import wordninja
import torch
import json
import pandas as pd
from transformers import BartTokenizer

with open("../noslang_data.json", "r") as f:
    data1 = json.load(f)
data2 = {}

with open("../emnlp_dict.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        row = line.split('\t')
        data2[row[0]] = row[1].rstrip()
norm_dict = {**data1, **data2}

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli", normalization=True, local_files_only=True)

def load_data(filename):
    concat_text = pd.DataFrame()
    raw_text = pd.read_csv(filename, usecols=[0], encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename, usecols=[1], encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename, usecols=[2], encoding='ISO-8859-1')
    seen = pd.read_csv(filename, usecols=[3], encoding='ISO-8859-1')
    label = pd.DataFrame.replace(raw_label, ['AGAINST', 'FAVOR', 'NEUTRAL'],[0, 1, 2])  # 将raw_label中的文字标签['AGAINST','FAVOR','NONE']替换成 0 1 2
    concat_text = pd.concat([raw_text, label, raw_target, seen], axis=1)  # 所有数据合并到一个表格中
    if 'train' not in filename:  # 如果文件不是训练数据 则将seen列中值为1的行去掉
        concat_text = concat_text[concat_text['seen?'] != 1]  # remove few-shot labels

    return concat_text

def data_clean(strings, norm_dict):
    # pd.set_option('future.no_silent_downcasting', True)
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    clean_data = p.clean(strings)
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    clean_data = [[x.lower()] for x in clean_data]

    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():  #
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])  # separate hashtags
    clean_data = [j for i in clean_data for j in i]

    return clean_data

def clean_and_tokenize(data):
    tweet = data['Tweet'].values.tolist()
    target = data['Target'].values.tolist()
    stance = data['Stance'].values.tolist()

    text = [None for _ in range(len(tweet))]
    for i in range(len(tweet)):
        text[i] = data_clean(tweet[i], norm_dict)
        target[i] = data_clean(target[i], norm_dict)

    concat_sent = []
    for tar, sent in zip(target, text):
        concat_sent.append([' '.join(sent), ' '.join(tar)])

    encoded_dict = tokenizer.batch_encode_plus(concat_sent, add_special_tokens=True, max_length=int(200),padding='max_length', return_attention_mask=True, truncation=True)
    encoded_dict['stance'] = stance

    return encoded_dict

def convert_tensor(encoded_dict):
    input_ids = torch.tensor(encoded_dict['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(encoded_dict['attention_mask'], dtype=torch.long)
    stance = torch.tensor(encoded_dict['stance'], dtype=torch.long)
    return input_ids, attention_mask, stance