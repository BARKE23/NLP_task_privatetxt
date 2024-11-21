import os
import torch
import json
import re
from transformers import RobertaTokenizer

from config import MODEL_NAME, MAX_LENGTH, LABEL_LIST_PATH, TRAIN_JSON_PATH, VALID_JSON_PATH, TEST_TXT_PATH

#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 文本清洗函数
def clean_text(text):
    text = re.sub(r'<.*?>|[^\w\s]', '', text)  # 使用正则表达式去除特殊字符（如标点符号、HTML标签等），<.*?>匹配HTML标签，[^\w\s]匹配非字母数字和空格的字符
    text = re.sub(r'\s+','', text).strip()  # 将连续的多个空格替换为单个空格，并去除文本两端的空格
    return text.lower()  # 转换为小写

# 加载标签列表
def load_label_list():
    with open(LABEL_LIST_PATH, 'r', encoding='utf-8') as f:
        label_list = [line.strip() for line in f.readlines()]
    return label_list

# 处理train.json和valid.json数据
def process_json_data(json_path, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    with open(json_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            text = clean_text(data['text'])
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                # 添加特殊标记，如[CLS]和[SEP]等，这是预训练模型输入要求的一部分
                max_length=MAX_LENGTH,
                # 按照配置文件中设定的最大长度对文本进行截断或填充
                padding='max_length',
                # 使用最大长度进行填充，确保每个样本的长度一致
                truncation=True,
                # 如果文本长度超过最大长度，则进行截断
                return_attention_mask=True,
                # 返回注意力掩码，用于指示模型哪些位置是真实文本，哪些是填充的部分
                return_tensors='pt'
                # 返回的结果将转换为PyTorch张量形式，方便后续处理
            )
            input_ids_list.append(encoding['input_ids'].squeeze())
            attention_mask_list.append(encoding['attention_mask'].squeeze())
            #labels_list.append(data['label'])
            #print("刚添加进labels_list的元素格式:", type(data['label']), data['label'])
            labels_list.append(data['label'][0] if isinstance(data['label'], list) else data['label'])
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.tensor([label_list.index(l) for l in labels_list])
    return input_ids, attention_mask, labels

# 处理test.txt数据
def process_test_data(test_txt_path, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    with open(test_txt_path, 'r') as f:
        for line in f.readlines():
            text = clean_text(line)
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids_list.append(encoding['input_ids'].squeeze())
            attention_mask_list.append(encoding['attention_mask'].squeeze())
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    return input_ids, attention_mask

tokenizer = RobertaTokenizer.from_pretrained("/home/hlh/hlhsource/roberta/")
label_list = load_label_list()
train_input_ids, train_attention_mask, train_labels = process_json_data(TRAIN_JSON_PATH, tokenizer)
valid_input_ids, valid_attention_mask, valid_labels = process_json_data(VALID_JSON_PATH, tokenizer)
test_input_ids, test_attention_mask = process_test_data(TEST_TXT_PATH, tokenizer)

if __name__ == "__main__":
    print(train_input_ids.shape)
    print(valid_input_ids.shape)
    print(test_input_ids.shape)