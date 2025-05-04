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

from model.TTFS_ORIGN import SNNModel, SpikingDense  # 导入SNN模型类

# --- 配置 ---
FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN_code\deap_features"  # 特征文件路径
INPUT_SIZE = 2728  # 特征维度（动态调整）
HIDDEN_UNITS_1 = 512  # 第一隐藏层大小
HIDDEN_UNITS_2 = 256  # 第二隐藏层大小
HIDDEN_UNITS_3 = 128  # 第三隐藏层大小
OUTPUT_SIZE = 2  # 输出类别（高/低）
T_MIN_INPUT = 0.0  # TTFS编码最小时间
T_MAX_INPUT = 1.0  # TTFS编码最大时间
LEARNING_RATE = 5e-4  # 学习率
WEIGHT_DECAY = 0  # 权重衰减
BATCH_SIZE = 16  # 批次大小
NUM_EPOCHS = 100  # 训练轮数
TEST_SPLIT_SIZE = 0.2  # 测试集比例
RANDOM_SEED = 42  # 随机种子
TRAINING_GAMMA = 10.0  # SNN参数

# --- 加载数据 ---
def load_features_from_mat(feature_dir):
    """加载特征和二值化标签"""
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_features.mat"))  # 获取.mat文件

    global INPUT_SIZE
    first_file = True

    for fpath in mat_files:
        mat_data = loadmat(fpath)  # 加载文件
        features = mat_data['features']  # 获取特征
        labels = mat_data['labels']  # 获取标签 (n_samples, 2)

        if first_file:
            INPUT_SIZE = features.shape[1]  # 设置输入维度
            first_file = False

        all_features.append(features.astype(np.float32))  # 添加特征
        binarized_labels = labels.astype(np.int64)  
        all_labels.append(binarized_labels)

    combined_features = np.vstack(all_features)  # 合并特征
    combined_labels = np.vstack(all_labels)  # 合并标签 (n_total_samples, 2)
    return combined_features, combined_labels

# --- TTFS编码 ---
def ttfs_encode(features: np.ndarray, t_min: float = T_MIN_INPUT, t_max: float = T_MAX_INPUT) -> torch.Tensor:
    """特征转脉冲时间"""
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features_np = np.array(features)
    features_np = np.clip(features_np, 0.0, 1.0)  # 限制到[0,1]
    spike_times = t_max - features_np * (t_max - t_min)  # 计算脉冲时间
    return torch.tensor(spike_times, dtype=torch.float32)

# --- 数据集类 ---
class EncodedEEGDataset(Dataset):
    """编码后的EEG数据集"""
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray, label_type: int = 0):
        """
        label_type: 0 for Valence, 1 for Arousal
        """
        self.features = encoded_features.to(dtype=torch.float32)  # 存储特征
        self.labels = torch.tensor(labels[:, label_type], dtype=torch.long)  # 选择Valence或Arousal标签

    def __len__(self) -> int:
        return len(self.labels)  # 样本数

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]  # 返回特征和选定标签

# --- 训练一个epoch ---
def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                gamma: float = TRAINING_GAMMA) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()  # 训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 设置SNN层时间参数
    current_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
    current_t_min = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
    current_t_max = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
    for layer in model.layers_list:
        if isinstance(layer, SpikingDense):
            layer_t_max = current_t_max
            layer.set_time_params(current_t_min_prev, current_t_min, layer_t_max)
            current_t_min_prev = current_t_min.clone()
        elif isinstance(layer, nn.Flatten):
            pass

    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)  # 移到设备
        optimizer.zero_grad()  # 清零梯度
        outputs, _ = model(features)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        running_loss += loss.item() * features.size(0)  # 累加损失
        _, predicted = torch.max(outputs.data, 1)  # 获取预测
        correct_predictions += (predicted == labels).sum().item()  # 统计正确预测
        total_samples += labels.size(0)  # 统计样本数

    epoch_loss = running_loss / total_samples  # 平均损失
    epoch_acc = correct_predictions / total_samples  # 平均准确率
    return epoch_loss, epoch_acc

# --- 评估模型 ---
def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, List, List]:
    """评估模型"""
    model.eval()  # 评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []

    with torch.no_grad():  # 禁用梯度
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            current_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
            current_t_min = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
            current_t_max = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
            for layer in model.layers_list:
                if isinstance(layer, SpikingDense):
                    layer_t_max = current_t_max
                    layer.set_time_params(current_t_min_prev, current_t_min, layer_t_max)
                    current_t_min_prev = current_t_min.clone()
                elif isinstance(layer, nn.Flatten):
                    pass
            outputs, _ = model(features)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())  # 收集标签
            all_preds.extend(predicted.cpu().numpy())  # 收集预测

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc, all_labels, all_preds

# --- 绘制训练历史 ---
def plot_history(train_losses, test_losses, train_accuracies, test_accuracies, label_type: int):
    """绘制损失和准确率曲线"""
    label_names = ['Valence', 'Arousal']
    label_name = label_names[label_type]
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='训练损失')
    plt.plot(epochs, test_losses, 'ro-', label='测试损失')
    plt.title(f'{label_name} 训练和测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='训练准确率')
    plt.plot(epochs, test_accuracies, 'ro-', label='测试准确率')
    plt.title(f'{label_name} 训练和测试准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"training_history_deap_{label_name}.png")  # 保存图像
    print(f"训练历史图已保存到 training_history_deap_{label_name}.png")
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 选择训练 Valence (0) 或 Arousal (1)
    label_type = 0  # 例如，0 for Valence

    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
    print(f"使用设备: {device}")

    # 加载数据
    features, labels = load_features_from_mat(FEATURE_DIR)
    print(f"成功加载数据. 特征形状: {features.shape}, 标签形状: {labels.shape}")
    print(f"确认输入维度: {INPUT_SIZE}")

    # 分割数据集
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        features, labels,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels[:, label_type]  # 按选定的标签分层
    )

    # 归一化特征
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train_orig)
    X_test_norm = scaler.transform(X_test_orig)

    # 编码特征
    X_train_encoded = ttfs_encode(X_train_norm)
    X_test_encoded = ttfs_encode(X_test_norm)
    print(f"特征编码为脉冲时间. 范围: [{T_MIN_INPUT:.2f}, {T_MAX_INPUT:.2f}]")
    print(f"编码训练数据形状: {X_train_encoded.shape}")

    # 创建数据集和加载器
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train, label_type=label_type)
    test_dataset = EncodedEEGDataset(X_test_encoded, y_test, label_type=label_type)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 构建SNN模型
    model = SNNModel().to(device)
    hidden_units_list = [HIDDEN_UNITS_1, HIDDEN_UNITS_2, HIDDEN_UNITS_3]
    current_dim = INPUT_SIZE
    for i, hidden_units in enumerate(hidden_units_list):
        model.add(SpikingDense(units=hidden_units, name=f"hidden_{i+1}", input_dim=current_dim))
        current_dim = hidden_units
    model.add(SpikingDense(units=OUTPUT_SIZE, name="output", outputLayer=True, input_dim=current_dim))
    model = model.to(device)
    print("\n模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总可训练参数: {total_params:,}")

    # 设置损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 训练模型
    train_losses, test_losses, train_accs, test_accs = [], [], []
    best_val_acc = 0.0
    start_time = time.time()
    print("\n开始训练...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _ = evaluate_model(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        train_accs.append(train_acc)
        test_accs.append(val_acc)
        print(f"轮次 {epoch+1}/{NUM_EPOCHS} | "
              f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # 输出结果
    end_time = time.time()
    print(f"\n训练完成，用时 {end_time - start_time:.2f} 秒")
    print(f"最佳验证准确率: {best_val_acc:.4f}")

    # 测试集评估
    final_loss, final_acc, all_labels, all_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"最终测试损失: {final_loss:.4f}")
    print(f"最终测试准确率: {final_acc:.4f}")

    # 分类报告
    if all_labels and all_preds:
        target_names = ['低', '高']
        report = classification_report(all_labels, all_preds, target_names=target_names)
        print("\n分类报告:")
        print(report)

    # 绘制训练历史
    if train_losses:
        plot_history(train_losses, test_losses, train_accs, test_accs, label_type)