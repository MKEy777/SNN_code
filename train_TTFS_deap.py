# 导入必要的库
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Union, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
from model.TTFS_ORIGN import SNNModel, SpikingDense

# 配置参数
FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN_code\deap_features"
INPUT_SIZE = 1408
HIDDEN_UNITS = [512, 128]
OUTPUT_SIZE = 2
T_MIN_INPUT, T_MAX_INPUT = 0.0, 1.0
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0
BATCH_SIZE = 16
NUM_EPOCHS = 100
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
TRAINING_GAMMA = 10.0

# 从 .mat 文件加载特征数据
def load_features_from_mat(feature_dir):
    mat_files = glob.glob(os.path.join(feature_dir, "s*_direct_features.mat"))
    if not mat_files:
        mat_files = glob.glob(os.path.join(feature_dir, "*_features.mat"))

    print(f"找到 {len(mat_files)} 个特征文件。正在加载...")
    all_f, all_l, first = [], [], True
    global INPUT_SIZE

    loaded_count = 0
    for fpath in mat_files:
        data = loadmat(fpath)
        feats = data['features']
        labs = data['labels']

        if first:
            INPUT_SIZE = feats.shape[1]
            print(f"从第一个文件推断输入大小: {INPUT_SIZE}")
            first = False

        all_f.append(feats.astype(np.float32))
        all_l.append(labs[:, 0].astype(np.int64)) # 仅使用效价标签
        loaded_count += 1

    print(f"成功加载了 {loaded_count} 个文件的有效数据。")
    return np.vstack(all_f), np.concatenate(all_l)

# TTFS 编码函数
def ttfs_encode(features: np.ndarray, t_min: float = T_MIN_INPUT, t_max: float = T_MAX_INPUT) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = np.clip(features, 0.0, 1.0)
    features = np.nan_to_num(features, nan=0.5)
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)

# PyTorch 数据集类
class EncodedEEGDataset(Dataset):
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray):
        self.features = encoded_features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

# 训练一个 Epoch 的函数
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

        with torch.no_grad():
            current_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
            current_t_min = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
            k = 0

            for i, layer in enumerate(model.layers_list):
                if isinstance(layer, SpikingDense):
                    new_t_max = torch.tensor(0.0, dtype=torch.float32, device=device)
                    min_ti_current = None

                    if not layer.outputLayer:
                        if k < len(min_ti_list) and min_ti_list[k] is not None:
                            min_ti_current = min_ti_list[k].squeeze()
                            current_layer_t_max = layer.t_max.clone().detach()
                            base_interval = torch.tensor(1.0, dtype=torch.float32, device=device)
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

# 评估模型的函数
@torch.no_grad()
def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, List, List]:
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []

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

# 绘制训练历史的函数
def plot_history(tr_losses, te_losses, tr_accs, te_accs, label_name="Valence"):
    epochs = range(1, len(tr_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, te_losses, 'ro-', label='Test Loss')
    plt.title('Loss curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, tr_accs, 'bo-', label='Training Accuracy')
    plt.plot(epochs, te_accs, 'ro-', label='Test Accuracy')
    plt.title('Accuracy curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"history_{label_name}_dynamic_TTFS.png")
    print(f"训练历史图已保存为: history_{label_name}_dynamic_TTFS.png")

# 主执行块
if __name__ == "__main__":
    # 设置随机种子和设备
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # 加载数据
    print("--- 开始加载数据 ---")
    features, labels = load_features_from_mat(FEATURE_DIR)
    print(f"成功加载数据。特征形状: {features.shape}, 标签形状: {labels.shape}")
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"标签分布: {dict(zip(unique_labels, counts))}")
    print("--- 数据加载完成 ---")

    # 划分训练集和测试集
    print("--- 开始划分数据集 ---")
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
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
    print(f"特征已编码为脉冲时间。范围: [{T_MIN_INPUT}, {T_MAX_INPUT}]")
    print(f"编码后训练数据形状: {X_train_encoded.shape}")
    print("--- TTFS 编码完成 ---")

    # 创建数据集和数据加载器
    print("--- 创建 PyTorch 数据集和加载器 ---")
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    test_dataset = EncodedEEGDataset(X_test_encoded, y_test)
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    print("--- 数据集和加载器创建完成 ---")

    # 构建 SNN 模型
    print("--- 构建 SNN 模型 ---")
    model = SNNModel()
    current_dim = INPUT_SIZE
    for i, hidden_unit in enumerate(HIDDEN_UNITS):
        model.add(SpikingDense(units=hidden_unit, name=f'dense_{i+1}', input_dim=current_dim,
                               outputLayer=False, kernel_initializer='glorot_uniform'))
        current_dim = hidden_unit
    model.add(SpikingDense(units=OUTPUT_SIZE, name='dense_output', input_dim=current_dim,
                           outputLayer=True, kernel_initializer='glorot_uniform'))

    print("SNN 模型结构:")
    print(model)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型可训练参数数量: {total_params}")
    print("--- SNN 模型构建完成 ---")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"损失函数: {criterion}")
    print(f"优化器: {optimizer}")

    # 初始化模型时间参数
    print("--- 初始化模型时间参数 ---")
    with torch.no_grad():
        init_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
        init_t_min = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
        init_t_max = init_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)

        for layer in model.layers_list:
            if isinstance(layer, SpikingDense):
                layer.set_time_params(init_t_min_prev, init_t_min, init_t_max)
                init_t_min_prev = init_t_min.clone()
                init_t_min = init_t_max.clone()
                init_t_max = init_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)
    print("--- 时间参数初始化完成 ---")

    # 开始训练
    print(f"\n--- 开始 SNN 训练 (共 {NUM_EPOCHS} 轮) ---")
    start_time = time.time()

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                            gamma=TRAINING_GAMMA)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | 训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"测试 Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    end_time = time.time()
    print(f"\n--- SNN 训练完成 ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    # 绘制训练历史
    if train_losses:
        print("\n--- 生成训练历史图 ---")
        plot_history(train_losses, test_losses, train_accuracies, test_accuracies, label_name="Valence")
    else:
        print("\n未能收集到训练历史数据。")

    # 最终评估
    print("\n--- 在测试集上进行最终评估 ---")
    final_test_loss, final_test_acc, final_labels, final_preds = evaluate_model(model, test_loader, criterion, device)

    print(f"最终测试损失: {final_test_loss:.4f}")
    print(f"最终测试准确率: {final_test_acc:.4f}")

    # 生成分类报告
    if final_labels and final_preds:
        report_target_names = ['Valence Low (0)', 'Valence High (1)']
        print("\n分类报告 (测试集 - Valence):")
        print(classification_report(final_labels, final_preds,
                                    target_names=report_target_names,
                                    digits=4,
                                    zero_division=0))
    else:
        print("无最终评估结果可用于生成分类报告。")

    print("\n脚本运行结束。")