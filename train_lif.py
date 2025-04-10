import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.load_dataset_deap import dataset_prepare
from module.LIF import  SNNRateLayer, SurrogateGradientFunction

class SNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=200, output_dim=2):
        super(SNN, self).__init__()

        # 第一层参数
        self.layer1_params = {
            'thresh': 0.5,
            'tau': 10.0, 
            'dt': 1.0,
            'reset_mode': 'soft',
            'grad_type': 'gaussian',
            'lens': 0.5,
            'gamma': 1.0,
            'hight': 0.15,
            'learn_params': True
        }
        
        # 第二层参数
        self.layer2_params = {
            'thresh': 1.0,
            'tau': 20.0,  
            'dt': 1.0,
            'reset_mode': 'soft',
            'grad_type': 'multi_gaussian',
            'lens': 0.4,
            'gamma': 1.2,
            'hight': 0.2,
            'learn_params': True
        }

        # 创建两层网络 32 -> 200 -> 2
        self.fc1 = SNNRateLayer(input_dim, hidden_dim, **self.layer1_params)
        self.fc2 = SNNRateLayer(hidden_dim, output_dim, **self.layer2_params)

    def set_neuron_state(self, batch_size, device):
        self.fc1.set_neuron_state(batch_size, device)
        self.fc2.set_neuron_state(batch_size, device)

    def forward(self, input):
        batch_size, seq_length, input_dim = input.shape
        device = input.device
        self.set_neuron_state(batch_size, device)

        output = torch.zeros(batch_size, self.fc2.neurons.num_neurons, device=device)
        all_spikes_1, all_mems_1 = [], []
        all_spikes_2, all_mems_2 = [], []

        for t in range(seq_length):
            input_t = input[:, t, :]

            spike_1, mem_1 = self.fc1(input_t)  
            spike_2, mem_2 = self.fc2(spike_1)  

            if t > 0:
                output += mem_2

            all_spikes_1.append(spike_1)
            all_mems_1.append(mem_1)
            all_spikes_2.append(spike_2)
            all_mems_2.append(mem_2)

        output = output / seq_length
        return output, {
            'spikes_1': torch.stack(all_spikes_1, dim=1),
            'mems_1': torch.stack(all_mems_1, dim=1),
            'spikes_2': torch.stack(all_spikes_2, dim=1),
            'mems_2': torch.stack(all_mems_2, dim=1)
        }

# 训练函数
def train(model, train_loader, optimizer, criterion, device, epochs=10):
    """
    训练 SNN 模型，使用 tqdm 显示训练进度。
    
    参数:
        model: SNN 模型实例
        train_loader: 训练数据的 DataLoader
        optimizer: 优化器（如 Adam）
        criterion: 损失函数（如 CrossEntropyLoss）
        device: 运行设备（CPU/GPU）
        epochs: 训练轮数
    
    返回:
        train_losses: 每轮平均损失列表
        train_accuracies: 每轮准确率列表
    """
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []

        # 使用 tqdm 显示每轮的批次进度
        with tqdm(train_loader, desc=f"第 {epoch+1}/{epochs} 轮", unit="批次") as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                # 重置神经元状态
                model.set_neuron_state(data.size(0), device)

                # 前向传播
                output, _ = model(data)
                loss = criterion(output, target)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # 计算预测
                pred = torch.argmax(output, dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

                # 更新进度条信息
                pbar.set_postfix({"损失": f"{loss.item():.4f}"})

        # 计算每轮指标
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"第 {epoch+1}/{epochs} 轮, 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")

    return train_losses, train_accuracies

# 评估函数
def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估 SNN 模型。
    
    参数:
        model: SNN 模型实例
        test_loader: 测试数据的 DataLoader
        criterion: 损失函数
        device: 运行设备（CPU/GPU）
    
    返回:
        test_loss: 平均测试损失
        test_accuracy: 测试准确率
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.set_neuron_state(data.size(0), device)
            output, _ = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"测试损失: {avg_loss:.4f}, 测试准确率: {accuracy:.4f}")
    return avg_loss, accuracy

# 主函数
def main():
    """
    主脚本：加载数据、初始化模型、训练、评估并可视化结果。
    """
    # 超参数
    batch_size = 128
    learning_rate = 0.001
    epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据目录（调整为你的本地路径）
    data_dir = "C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN_code\\dataset"

    # 加载并准备数据
    train_loader, test_loader = dataset_prepare(
        window_length_sec=4,
        n_subjects=26,
        single_subject=False,
        load_all=True,
        only_EEG=True,
        label_type=[0, 2],
        data_dir=data_dir,
        batch_size=batch_size,
        normalize=True
    )

    # 初始化模型
    input_dim = 32
    hidden_dim = 200
    output_dim = 2
    model = SNN(input_dim, hidden_dim, output_dim).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    print("开始训练...")
    train_losses, train_accuracies = train(model, train_loader, optimizer, criterion, device, epochs)

    # 评估模型
    print("\n在测试集上评估...")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    # 最终可视化
    plt.figure(figsize=(12, 5))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label="训练损失")
    plt.axhline(y=test_loss, color='r', linestyle='--', label="测试损失")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.title("损失随时间变化")
    plt.legend()

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, 'g-', label="训练准确率")
    plt.axhline(y=test_accuracy, color='r', linestyle='--', label="测试准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.title("准确率随时间变化")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()