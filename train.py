import os
import time
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model import CNN_Attention_Classifier
from data_preprocessing import train_input_ids, train_attention_mask, train_labels, valid_input_ids, valid_attention_mask, valid_labels
from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 保存模型和日志的函数
def save_model_and_log(model, optimizer, epoch, train_loss, valid_loss, valid_acc, remark=""):
    # 路径
    save_path = "checkpoints"
    os.makedirs(save_path, exist_ok=True)

    # 时间戳（格式为 MMDD-HHMM）
    timestamp = time.strftime("%m%d-%H%M")

    # 文件名
    model_name = f"{remark}_{timestamp}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_ep{NUM_EPOCHS}.pth"
    model_path = os.path.join(save_path, model_name)

    # 日志记录
    log_path = os.path.join(save_path, "training_log.txt")
    with open(log_path, "a") as log_file:
        log_file.write(
            f"{timestamp} - lr: {LEARNING_RATE}, batch_size: {BATCH_SIZE}, epochs: {NUM_EPOCHS}, valid_acc: {valid_acc:.4f}, file: {model_name}\n"
        )

    #print(f"模型已保存至: {model_path}")

    #保存模型
    torch.save(model.state_dict(), model_path)

    return model_path  # 返回保存的模型路径


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("即将加载模型...")
    model = CNN_Attention_Classifier().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)#定义优化器
    criterion = torch.nn.CrossEntropyLoss()


    # # 学习率调度器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 早停策略参数
    early_stopping_patience = 5
    no_improvement_epochs = 0

    train_dataset = TensorDataset(train_input_ids.to(device), train_attention_mask.to(device), train_labels.to(device))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    valid_dataset = TensorDataset(valid_input_ids.to(device), valid_attention_mask.to(device), valid_labels.to(device))
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # # 计算训练步数
    # total_steps = len(train_loader) * NUM_EPOCHS
    #
    # # 设置学习率调度器
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=int(0.1 * total_steps),  # 10% 用于 warmup
    #     num_training_steps=total_steps  # 总训练步数
    # )

    best_valid_accuracy = 0.0  # 用于追踪最佳验证准确度
    best_model_path = ""  # 保存最佳模型的路径

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        # 添加进度条显示训练进度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)
        for batch in progress_bar:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 更新进度条的显示内容
            progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})


        # 在验证集上验证
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attention_mask, labels = batch
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算验证集指标
        train_loss_avg = total_loss / len(train_loader)
        valid_loss_avg = valid_loss / len(valid_loader)
        valid_accuracy = correct / total

        # # 更新学习率调度器
        # scheduler.step(valid_loss_avg)

        # 追踪并保存最佳模型
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model = model.state_dict()  # 保存当前最好的模型状态
            no_improvement_epochs = 0  # 重置早停计数器
        else:
            no_improvement_epochs += 1

        # 打印当前训练和验证的指标
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss_avg:.4f} - Valid Loss: {valid_loss_avg:.4f} - Valid Accuracy: {valid_accuracy:.4f}")
        #print(f"当前最佳验证准确率: {best_valid_accuracy:.4f}")  # 打印当前最佳准确率

        # 检查早停条件
        if no_improvement_epochs >= early_stopping_patience:
            print(f"早停触发。最佳验证准确率为: {best_valid_accuracy:.4f}")
            break

        torch.cuda.empty_cache()

        # 更新最佳验证准确率
        # if valid_accuracy > best_valid_accuracy:
        #     best_valid_accuracy = valid_accuracy
        #     best_model_path = save_model_and_log(
        #         model=model,
        #         optimizer=optimizer,
        #         epoch=epoch + 1,
        #         train_loss=train_loss_avg,
        #         valid_loss=valid_loss_avg,
        #         valid_acc=valid_accuracy,
        #         remark="best_model"
        #     )

    # 在所有 epoch 完成后，保存最佳模型
    if best_model is not None:
        # 将最佳模型加载并保存
        model.load_state_dict(best_model)
        best_model_path = save_model_and_log(
            model=model,
            optimizer=optimizer,
            epoch=NUM_EPOCHS,
            train_loss=train_loss_avg,  # 使用最后一个 epoch 的训练损失
            valid_loss=valid_loss_avg,  # 使用最后一个 epoch 的验证损失
            valid_acc=best_valid_accuracy,  # 使用最佳验证准确率
            remark="best_model"
        )



    print(f"Training finished. Best valid accuracy is: {best_valid_accuracy:.4f}. Best model saved at {best_model_path}")


if __name__ == "__main__":
    train()
