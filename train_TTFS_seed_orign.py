import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Union, List, Optional, Tuple
import time
import matplotlib.pyplot as plt

# 导入模型定义 (假设在 model 子目录)
from model.TTFS_ORIGN import SNNModel, SpikingDense

# --- 参数设置 ---
FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features_NoBandpass_Fixed" # 特征文件目录
INPUT_SIZE = 682 # 输入特征维度 (62通道 * 11特征/通道)
HIDDEN_UNITS_1 = 512 # 第一个隐藏层单元数
HIDDEN_UNITS_2 = 128 # 第二个隐藏层单元数
OUTPUT_SIZE = 3    # 输出类别数
T_MIN_INPUT = 0.0  # 输入编码时间下限
T_MAX_INPUT = 1.0  # 输入编码时间上限

LEARNING_RATE = 5e-4 # 学习率
BATCH_SIZE = 16      # 批处理大小
NUM_EPOCHS = 100     # 训练周期数
TEST_SPLIT_SIZE = 0.2 # 测试集比例
RANDOM_SEED = 42     # 随机种子

TRAINING_GAMMA = 10.0 # 动态时间调整参数

# --- 数据加载函数 ---
def load_features_from_mat(feature_dir):
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_no_bandpass_features.mat"))
    print(f"找到 {len(mat_files)} 个特征文件。正在加载...")
    for fpath in mat_files:
        mat_data = loadmat(fpath)
        features = mat_data['features']
        labels = mat_data['labels'].flatten()
        all_features.append(features.astype(np.float32))
        all_labels.append(labels)

    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    print(f"总共加载了 {combined_features.shape[0]} 个数据段。")

    label_mapping = {-1: 0, 0: 1, 1: 2} # 标签映射
    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels], dtype=np.int64)
    print("标签映射完成: {-1: 0, 0: 1, 1: 2}")
    print(f"唯一的映射后标签值: {np.unique(mapped_labels)}")

    return combined_features, mapped_labels

# --- TTFS 编码函数 ---
def ttfs_encode(features: np.ndarray, t_min: float = T_MIN_INPUT, t_max: float = T_MAX_INPUT) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = np.clip(features, 0.0, 1.0)
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)

# --- PyTorch 数据集类 ---
class EncodedEEGDataset(Dataset):
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray):
        self.features = encoded_features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

# --- 训练周期函数 ---
def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                gamma: float = TRAINING_GAMMA) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        outputs, min_ti_list = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad(): # 动态调整时间参数
            current_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
            current_t_min = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
            k = 0
            for i, layer in enumerate(model.layers_list):
                if isinstance(layer, SpikingDense):
                    new_t_max = torch.tensor(0.0, dtype=torch.float32, device=device)
                    if not layer.outputLayer:
                        if k < len(min_ti_list) and min_ti_list[k] is not None:
                            min_ti_current = min_ti_list[k].squeeze()
                            if min_ti_current.ndim > 0:
                                min_ti_current = min_ti_current[0]
                            base_interval = torch.tensor(1.0, dtype=torch.float32, device=device)
                            current_layer_t_max = layer.t_max.clone().detach()
                            dynamic_term = gamma * (current_layer_t_max - min_ti_current)
                            dynamic_term = torch.clamp(dynamic_term, min=0.0)
                            new_t_max = current_t_min + torch.maximum(base_interval, dynamic_term)
                        else:
                            new_t_max = current_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)
                        k += 1
                    else:
                        new_t_max = current_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)

                    layer.set_time_params(current_t_min_prev, current_t_min, new_t_max)
                    current_t_min_prev = current_t_min.clone()
                    current_t_min = new_t_max.clone()

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

# --- 评估模型函数 ---
def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, List, List]:
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs, _ = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc, all_labels, all_preds

# --- 绘制历史曲线函数 ---
def plot_history(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='训练损失')
    plt.plot(epochs, test_losses, 'ro-', label='测试损失')
    plt.title('损失曲线')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='训练准确率')
    plt.plot(epochs, test_accuracies, 'ro-', label='测试准确率')
    plt.title('准确率曲线')
    plt.xlabel('周期')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_history.png") # 保存图像
    print("训练历史曲线图已保存为 training_history.png")
    # plt.show() # 如果需要显示图像，取消此行注释

# --- 主程序 ---
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED) # 设置随机种子
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择设备
    print(f"使用的设备: {device}")

    print("--- 开始加载数据 ---") # 加载数据
    features, labels = load_features_from_mat(FEATURE_DIR)
    print(f"成功加载数据。特征形状: {features.shape}, 标签形状: {labels.shape}")
    print("--- 数据加载完成 ---")

    print("--- 开始划分数据集 ---") # 划分训练集和测试集
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"训练集大小: {len(X_train_orig)}")
    print(f"测试集大小: {len(X_test_orig)}")
    print("--- 数据集划分完成 ---")

    print("--- 开始特征归一化 ---") # 特征归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train_orig)
    X_test_norm = scaler.transform(X_test_orig)
    print("特征已归一化到 [0, 1]。")
    print("--- 特征归一化完成 ---")

    print("--- 开始 TTFS 编码 ---") # TTFS 编码
    X_train_encoded = ttfs_encode(X_train_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    X_test_encoded = ttfs_encode(X_test_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    print(f"编码后训练数据形状: {X_train_encoded.shape}")
    print("--- TTFS 编码完成 ---")

    print("--- 创建 PyTorch 数据集和加载器 ---") # 创建 Dataset 和 DataLoader
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    test_dataset = EncodedEEGDataset(X_test_encoded, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    print("--- 数据集和加载器创建完成 ---")

    print("--- 构建 SNN 模型 ---") # 构建模型
    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense_1', input_dim=INPUT_SIZE, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense_2', outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='dense_output', outputLayer=True, kernel_initializer='glorot_uniform'))
    model.to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总可训练参数量: {total_params}")

    criterion = nn.CrossEntropyLoss() # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 定义优化器

    print("--- 初始化模型时间参数 ---") # 初始化时间参数
    with torch.no_grad():
        init_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
        init_t_min = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
        for layer in model.layers_list:
            if isinstance(layer, SpikingDense):
                init_t_max = init_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)
                layer.set_time_params(init_t_min_prev, init_t_min, init_t_max)
                init_t_min_prev = init_t_min.clone()
                init_t_min = init_t_max.clone()

    print("--- 开始 SNN 训练 ---") # 开始训练
    start_time = time.time()
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    for epoch in range(NUM_EPOCHS): # 训练指定周期数
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)
        epoch_end_time = time.time()
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print(f"周期 [{epoch+1}/{NUM_EPOCHS}] | 训练损失: {train_loss:.4f}, Acc: {train_acc:.4f} | 测试损失: {test_loss:.4f}, Acc: {test_acc:.4f} | 耗时: {epoch_end_time - epoch_start_time:.2f} 秒")

    end_time = time.time()
    print(f"--- SNN 训练完成, 总耗时: {end_time - start_time:.2f} 秒 ---")

    plot_history(train_losses, test_losses, train_accuracies, test_accuracies) # 绘制并保存历史曲线

    print("--- 最终模型评估 ---") # 评估最终模型
    final_test_loss, final_test_acc, final_labels, final_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"最终测试损失: {final_test_loss:.4f}")
    print(f"最终测试准确率: {final_test_acc:.4f}")

    report_names = ['负面 (0)', '中性 (1)', '正面 (2)'] # 定义报告标签名
    report = classification_report(final_labels, final_preds, target_names=report_names, digits=4, zero_division=0) # 生成分类报告
    print("\n分类报告:")
    print(report)

    print("脚本运行结束。")