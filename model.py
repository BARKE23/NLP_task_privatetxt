import os
import torch
import torch.nn as nn
from transformers import RobertaModel
from config import MODEL_NAME, NUM_CLASSES

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class CNN_Attention_Classifier(nn.Module):
    def __init__(self):
        super(CNN_Attention_Classifier, self).__init__()
        # 加载预训练的RoBERTa模型
        self.roberta = RobertaModel.from_pretrained("/home/hlh/hlhsource/roberta/")

        # CNN模块
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 768))  # 调整 kernel_size 高度为 1
        self.conv2 = nn.Conv2d(1, 128, kernel_size=(1, 768))
        self.conv3 = nn.Conv2d(1, 256, kernel_size=(1, 768))
        self.relu = nn.ReLU()

        # 分类器
        self.fc = nn.Linear(448, NUM_CLASSES)  # 全连接层输入调整为拼接后的维度

    def forward(self, input_ids, attention_mask):
        # RoBERTa 部分
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # 提取 [CLS] 标记的隐藏状态，形状 [batch_size, hidden_dim]
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # 调整维度以适配 CNN 输入，形状变为 [batch_size, 1, 1, hidden_dim]
        pooled_output = pooled_output.unsqueeze(1).unsqueeze(1)

        # CNN 部分
        x1 = self.relu(self.conv1(pooled_output))  # [batch_size, 64, 1, 1]
        x2 = self.relu(self.conv2(pooled_output))  # [batch_size, 128, 1, 1]
        x3 = self.relu(self.conv3(pooled_output))  # [batch_size, 256, 1, 1]
        x = torch.cat([x1, x2, x3], dim=1)  # 拼接 CNN 输出，形状为 [batch_size, 448, 1, 1]
        x = x.view(x.size(0), -1)  # 扁平化，形状为 [batch_size, 448]

        # 分类部分
        logits = self.fc(x)  # 最终分类
        return logits
