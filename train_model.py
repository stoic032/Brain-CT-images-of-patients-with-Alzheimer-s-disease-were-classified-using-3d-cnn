import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from src.data.dataload import train_dataset, val_dataset, test_dataset  # 直接执行的脚本不能使用间接引用
from src.models.cnn3D import cnn3D  # 直接执行的脚本不能使用间接引用


# 载入GPU
def device_config():
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
        device = torch.device("cuda:0")
        print(f"Using {gpu_num} GPU(s).")
        print(f"Selected GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU.")

    return device


# 日志函数
def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


# 训练函数
def train_one_epoch(model, dataloader, loss_fn, optimizer, log_interval, device):
    """单次训练过程"""
    model.train()
    total_loss = 0
    correct = 0

    for batch_index, (data, label) in enumerate(dataloader, start=1):
        data, label = data.to(device), label.to(device)

        # 前向传播
        output = model(data)
        loss = loss_fn(output, label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印训练情况
        if batch_index % log_interval == 0:
            print(f"训练进度:[{batch_index}/{len(dataloader)}] >>>>>> Loss:{loss.item():.6f} ")

        # 分类正确个数
        correct += (output.argmax(1) == label).type(torch.float).sum().item()

    accuracy = correct / len(dataloader.dataset)
    # 返回平均损失和准确率
    return total_loss / len(dataloader), accuracy


# 测试函数
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_index, (data, label) in enumerate(dataloader, start=1):
            data, label = data.to(device), label.to(device)

            output = model(data)
            loss = loss_fn(output, label)
            total_loss += loss.item()

            # 分类正确个数
            correct += (output.argmax(1) == label).type(torch.float).sum().item()

    total_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)

    return total_loss, accuracy


# 模型性能可视化
def visualize(train_losses, val_losses, test_loss, train_acc, val_acc, test_acc, img_name=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # 绘制训练损失和测试损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs, val_losses, 'o-', label='Validation Loss')

    # 绘制测试损失的直线
    plt.axhline(test_loss, color='red', linestyle='--', label='Test Loss')

    plt.title('Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制模型准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'o-', color='blue', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'o-', color='green', label='Validation Accuracy')
    for i, (t_acc, v_acc) in enumerate(zip(train_acc, val_acc)):
        plt.text(epochs[i], t_acc, f"{t_acc:.2f}", ha='center', va='bottom')
        plt.text(epochs[i], v_acc, f"{v_acc:.2f}", ha='center', va='bottom')

    # 绘制测试准确率的直线
    plt.axhline(test_acc, color='red', linestyle='--', label='Test Accuracy')
    plt.text((epochs[0] + epochs[-1]) / 2, test_acc, f"{test_acc:.2f}", ha='center', va='bottom')

    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./results/{img_name} Model Accuracy.png')
    plt.show()


def main(n_epochs=10, model_pre_weight_path=None):
    # 设备配置
    device = device_config()

    # 加载数据
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # 加载模型
    model = cnn3D(num_classes=3)
    if model_pre_weight_path:
        assert os.path.exists(model_pre_weight_path), f"预训练模型权重：file {model_pre_weight_path} dose not exist."
        pre_weights = torch.load(model_pre_weight_path, map_location='cpu')
        model.load_state_dict(pre_weights, strict=False)
    model.to(device)

    # 训练参数配置
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 打印日志间隔
    log_interval = 10

    # 训练开始时间
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 打开日志记录器
    setup_logging(log_file=f"./logs/{current_time}")

    # 记录损失和准确率
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 训练循环
    # 训练开始
    start_time = datetime.now()
    for n_epoch in range(1, n_epochs + 1):
        epoch_start_time = datetime.now()
        print(f"Epochs:[{n_epoch}/{n_epochs}]")

        # 训练并验证
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, log_interval, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        # 当前epoch训练耗时
        epoch_end_time = datetime.now()
        epoch_total_time =  int((epoch_end_time - epoch_start_time).total_seconds())

        # 保存并保存日志
        log = f"Epoch {n_epoch}: time = {epoch_total_time}(s) | train_loss = {train_loss:.4f} | train_acc:{train_acc:.4f}" \
              f" val_loss:{val_loss:.6f} | val_acc:{val_acc:.4f}"
        logging.info(log)
        print(log)

        # 记录数据用于可视化
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    # 训练结束时间
    train_end_time = datetime.now()
    total_training_time = int((train_end_time - start_time).total_seconds())
    print(f"总训练耗时: {total_training_time}(s)")

    test_start_time = datetime.now()
    # 测试模型
    test_loss, test_acc = validate(model, test_loader, loss_fn, device)

    test_end_time = datetime.now()
    total_test_time = int((test_end_time - test_start_time).total_seconds())
    print(f"测试耗时：{total_test_time}(s)")

    end_time = datetime.now()
    total_time = int((end_time - start_time).total_seconds())
    log = f"all time:{total_time}(s) | test_loss:{test_loss:.6f} | test_acc:{test_acc:.4f}"
    logging.info(log)
    print(log)

    # 保存模型
    torch.save(model.state_dict(), f"./models/{current_time}_n_epochs_{n_epochs}.pt")

    # 结果可视化
    visualize(train_losses=train_losses, val_losses=val_losses, test_loss=test_loss,
              train_acc=train_accuracies, val_acc=val_accuracies, test_acc=test_acc,
              img_name=f"{current_time}_n_epochs_{n_epochs}")


if __name__ == "__main__":
    model_pre_path = "models/20240112-204825_n_epochs_3.pt"
    main(n_epochs=10, model_pre_weight_path=model_pre_path)
    # main(n_epochs=3)

