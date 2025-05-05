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

from model.TTFS_ORIGN import SNNModel, SpikingDense 

FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features"

# 网络结构与训练参数定义
INPUT_SIZE = 2728
HIDDEN_UNITS_1 = 512
HIDDEN_UNITS_2 = 128
OUTPUT_SIZE = 3
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0

LEARNING_RATE = 5e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

# 动态调整超参数
TRAINING_GAMMA = 10.0

def load_features_from_mat(feature_dir):
    # 加载所有 .mat 特征文件并合并数据与标签
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_features_minimal.mat"))

    print(f"找到 {len(mat_files)} 个特征文件。正在加载...")

    for fpath in mat_files:
        mat_data = loadmat(fpath)
        features = mat_data['features']
        labels = mat_data['labels'].flatten()
        all_features.append(features.astype(np.float32))
        all_labels.append(labels)

    # 垂直堆叠所有特征和标签
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    print(f"总共加载了 {combined_features.shape[0]} 个数据段。")

    # 将原始标签映射到 0,1,2 三类
    label_mapping = {-1: 0, 0: 1, 1: 2}
    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels], dtype=np.int64)
    print("标签映射已应用: {-1: 0 (负面), 0: 1 (中性), 1: 2 (正面)}")
    unique_mapped_labels = np.unique(mapped_labels)
    print(f"唯一的映射后标签值: {unique_mapped_labels}")

    return combined_features, mapped_labels


def ttfs_encode(features: np.ndarray, t_min: float = T_MIN_INPUT, t_max: float = T_MAX_INPUT) -> torch.Tensor:
    # 将连续特征值编码为脉冲触发时间 (TTFS 编码)
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = np.clip(features, 0.0, 1.0)
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)


class EncodedEEGDataset(Dataset):
    # 自定义 PyTorch 数据集，存储 TTFS 编码后的特征与标签
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray):
        self.features = encoded_features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                gamma: float = TRAINING_GAMMA) -> Tuple[float, float]:
    # 训练一个周期，返回损失与准确率
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

        # 动态调整每层的时间参数
        with torch.no_grad():
            current_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
            current_t_min = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
            k = 0

            for i, layer in enumerate(model.layers_list):
                if isinstance(layer, SpikingDense):
                    new_t_max = torch.tensor(0.0, dtype=torch.float32, device=device)
                    if not layer.outputLayer:
                        # 内部层根据最早脉冲时间动态更新
                        if k < len(min_ti_list) and min_ti_list[k] is not None:
                            min_ti_current = min_ti_list[k].squeeze()
                            base_interval = torch.tensor(1.0, dtype=torch.float32, device=device)
                            current_layer_t_max = layer.t_max.clone().detach()
                            dynamic_term = gamma * (current_layer_t_max - min_ti_current)
                            dynamic_term = torch.clamp(dynamic_term, min=0.0)
                            new_t_max = current_t_min + torch.maximum(base_interval, dynamic_term)
                        else:
                            new_t_max = current_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)
                        k += 1
                    else:
                        # 输出层固定间隔
                        new_t_max = current_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)

                    # 更新层的时间参数
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


def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, List, List]:
    # 在验证/测试集上评估模型性能，返回损失、准确率及预测结果
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


def plot_history(train_losses, test_losses, train_accuracies, test_accuracies):
    # 绘制训练与测试过程中的损失和准确率曲线
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, test_losses, 'ro-', label='Test Loss')
    plt.title('Training and test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy')
    plt.title('Train and test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设置随机种子以保证可复现
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # 选择运行设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 加载、预处理数据并创建数据加载器
    print("--- 开始加载数据 ---")
    features, labels = load_features_from_mat(FEATURE_DIR)
    print(f"成功加载数据。特征形状: {features.shape}, 标签形状: {labels.shape}")
    print("--- 数据加载完成 ---")

    print("--- 开始划分数据集 ---")
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"训练集大小: {len(X_train_orig)}")
    print(f"测试集大小: {len(X_test_orig)}")
    print("--- 数据集划分完成 ---")

    # 特征归一化
    print("--- 开始特征归一化 ---")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train_orig)
    X_test_norm = scaler.transform(X_test_orig)
    print("特征已归一化到 [0, 1]。")
    print("--- 特征归一化完成 ---")

    # TTFS 编码
    print("--- 开始 TTFS 编码 ---")
    X_train_encoded = ttfs_encode(X_train_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    X_test_encoded = ttfs_encode(X_test_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    print(f"编码后训练数据形状: {X_train_encoded.shape}")
    print("--- TTFS 编码完成 ---")

    # 创建数据集与加载器
    print("--- 创建 PyTorch 数据集和加载器 ---")
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    test_dataset = EncodedEEGDataset(X_test_encoded, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())
    print("--- 数据集和加载器创建完成 ---")

    # 构建 SNN 模型结构
    print("--- 构建 SNN 模型 ---")
    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense_1', input_dim=INPUT_SIZE,
                           outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense_2', outputLayer=False,
                           kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='dense_output', outputLayer=True,
                           kernel_initializer='glorot_uniform'))
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam 优化器

    # 初始化每层的时间参数
    with torch.no_grad():
        init_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
        init_t_min = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
        init_t_max = init_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)
        for layer in model.layers_list:
            if isinstance(layer, SpikingDense):
                layer.set_time_params(init_t_min_prev, init_t_min, init_t_max)
                init_t_min_prev, init_t_min = init_t_min, init_t_max
                init_t_max = init_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)

    # 训练循环
    print("--- 开始 SNN 训练 ---")
    start_time = time.time()
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | 训练损失: {train_loss:.4f}, Acc: {train_acc:.4f} | 测试损失: {test_loss:.4f}, Acc: {test_acc:.4f}")

    end_time = time.time()
    print(f"--- SNN 训练完成, 耗时: {end_time - start_time:.2f} 秒 ---")

    # 绘制训练历史
    if train_losses:
        plot_history(train_losses, test_losses, train_accuracies, test_accuracies)

    # 最终评估与分类报告
    final_test_loss, final_test_acc, final_labels, final_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"最终测试损失: {final_test_loss:.4f}, 最终测试准确率: {final_test_acc:.4f}")
    report_names = ['负面 (0)', '中性 (1)', '正面 (2)']
    print(classification_report(final_labels, final_preds, target_names=report_names, digits=4, zero_division=0))
    print("脚本运行结束。")
