import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import CNN_Attention_Classifier
from data_preprocessing import test_input_ids, test_attention_mask, label_list
from config import RESULT_TXT_PATH, CHECKPOINT_PATH


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Attention_Classifier().to(device)

    # 加载已保存的模型权重
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()  # 设置为评估模式，在该模式下，Dropout等操作会被禁用

    test_dataset = TensorDataset(test_input_ids.to(device), test_attention_mask.to(device))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []  # 用于存储每行文本的预测标签

    with torch.no_grad():  # 不需要计算梯度
        for batch in test_loader:
            input_ids, attention_mask = batch
            logits = model(input_ids, attention_mask)
            # _, predicted = torch.max(logits.data, 1)  # 获取预测的标签
            # label_list.append(predicted.item())
            probabilities = torch.sigmoid(logits).cpu().numpy()  # 获取每个标签的概率

            # 找到最大概率标签
            max_index = probabilities.argmax()  # 最大值的索引
            max_label = label_list[max_index]  # 对应的标签名

            if max_label == "No_Mentioned":
                predicted_labels = ["No_Mentioned"]
            else:
                # 筛选出概率大于阈值的标签
                predicted_indices = (probabilities > 0.9).astype(int)  # 选择概率大于 0.9 的标签
                predicted_labels = [
                    label_list[i] for i, value in enumerate(predicted_indices[0]) if value == 1
                ]

                # 如果没有筛选出大于阈值的标签，就选取最大概率标签
                if not predicted_labels:
                    # 找到概率最大的标签
                    predicted_labels = [max_label]

                # 检查并删除 "No_Mentioned" 标签（如果存在）
                if "No_Mentioned" in predicted_labels:
                    predicted_labels.remove("No_Mentioned")

            #predicted_indices = (probabilities > 0.9).astype(int)  # 选择概率大于 0.5 的标签
            # predicted_label_indices = torch.argmax(logits, dim=1).squeeze().tolist()
            # predicted_labels = [label_list[index] for index in predicted_label_indices]
            # all_labels.append(",".join(predicted_labels))

            # 将序号转换为标签名
            # predicted_labels = [
            #     label_list[i] for i, value in enumerate(predicted_indices[0]) if value == 1
            # ]
            results.append(predicted_labels)

    # 检查结果文件所在的目录是否存在，如果不存在则创建目录
    dir_path = os.path.dirname(RESULT_TXT_PATH)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 将预测结果保存到文件
    with open(RESULT_TXT_PATH, 'w') as f:
        for labels in results:
            # 以逗号分隔写入标签
            f.write(','.join(labels) + '\n')


if __name__ == "__main__":
    test()
