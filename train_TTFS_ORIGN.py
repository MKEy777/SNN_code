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
import warnings
import matplotlib.pyplot as plt

from model.TTFS_ORIGN import SNNModel, SpikingDense


FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features_Minimal"

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
EARLY_STOPPING_PATIENCE = 10
TRAINING_GAMMA = 10.0


def load_features_from_mat(feature_dir):
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_features_minimal.mat"))

    print(f"找到 {len(mat_files)} 个特征文件。正在加载...")

    for fpath in mat_files:
        fname = os.path.basename(fpath)
        try:
            mat_data = loadmat(fpath)
            if 'features' not in mat_data or 'labels' not in mat_data:
                 print(f"警告: 文件 {fname} 缺少 'features' 或 'labels' 键。已跳过。")
                 continue
            features = mat_data['features']
            labels = mat_data['labels'].flatten()

            global INPUT_SIZE
            if not all_features:
                INPUT_SIZE = features.shape[1]
                print(f"推断输入维度: {INPUT_SIZE}")
            elif features.shape[1] != INPUT_SIZE:
                 print(f"警告: 文件 {fname} 特征维度 ({features.shape[1]}) 与预期 ({INPUT_SIZE}) 不符。已跳过。")
                 continue

            if features.shape[0] != labels.shape[0]:
                 print(f"警告: 文件 {fname} 特征和标签数量不匹配。已跳过。")
                 continue

            all_features.append(features.astype(np.float32))
            all_labels.append(labels)
        except Exception as e:
            print(f"加载文件 {fname} 时出错: {e}。已跳过。")

    if not all_features:
        raise ValueError("未能从指定目录加载任何有效的特征文件。")

    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    print(f"总共加载了 {combined_features.shape[0]} 个数据段。")

    label_mapping = {-1: 0, 0: 1, 1: 2}
    unique_original_labels = np.unique(combined_labels)
    valid_labels = set(label_mapping.keys())
    invalid_labels = set(unique_original_labels) - valid_labels
    if invalid_labels:
        print(f"警告: 数据中包含未定义的标签值: {invalid_labels}。这些样本将被跳过。")
        valid_indices = np.isin(combined_labels, list(valid_labels))
        combined_features = combined_features[valid_indices]
        combined_labels = combined_labels[valid_indices]
        if combined_features.shape[0] == 0:
             raise ValueError("过滤无效标签后没有剩余数据。")

    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels], dtype=np.int64)
    print("标签映射已应用: {-1: 0 (负面), 0: 1 (中性), 1: 2 (正面)}")
    unique_mapped_labels = np.unique(mapped_labels)
    print(f"唯一的映射后标签值: {unique_mapped_labels}")

    if len(unique_mapped_labels) != OUTPUT_SIZE:
         print(f"警告: 映射后的唯一标签数量 ({len(unique_mapped_labels)}) 与期望的输出大小 ({OUTPUT_SIZE}) 不匹配。")

    return combined_features, mapped_labels

def ttfs_encode(features: np.ndarray, t_min: float = T_MIN_INPUT, t_max: float = T_MAX_INPUT) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = np.clip(features, 0.0, 1.0)
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)

class EncodedEEGDataset(Dataset):
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
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    processed_batches = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        try:
            outputs, min_ti_list = model(features)
        except Exception as e:
             print(f"错误发生在模型前向传播: {e}")
             continue

        if not torch.all(torch.isfinite(outputs)):
            print(f"警告: Epoch {epoch+1}, Batch {batch_idx}: 模型输出包含 NaN/Inf。跳过此批次。")
            continue

        try:
            loss = criterion(outputs, labels)
        except Exception as e:
             print(f"错误发生在损失计算: {e}")
             continue

        if not torch.isfinite(loss):
            print(f"警告: Epoch {epoch+1}, Batch {batch_idx}: 计算出的损失为 NaN/Inf。跳过此批次。")
            continue

        optimizer.zero_grad()
        try:
            loss.backward()
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            # print(f"Epoch {epoch + 1}, Batch {batch_idx}: Gradient norm = {grad_norm:.4f}")
        except Exception as e:
             print(f"错误发生在反向传播: {e}")
             continue

        try:
            optimizer.step()
        except Exception as e:
             print(f"错误发生在优化器步骤: {e}")
             continue

        with torch.no_grad():
            current_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
            current_t_min = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
            k = 0

            for i, layer in enumerate(model.layers_list):
                if isinstance(layer, SpikingDense):
                    new_t_max = torch.tensor(0.0, dtype=torch.float32, device=device)
                    min_ti_current = None

                    if not layer.outputLayer:
                        if k >= len(min_ti_list) or min_ti_list[k] is None:
                            warnings.warn(f"Layer {layer.name}: Missing min_ti for t_max update. Using fixed interval.")
                            new_t_max = current_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)
                        else:
                            min_ti_current = min_ti_list[k].squeeze()
                            if not torch.isfinite(min_ti_current):
                                 warnings.warn(f"Layer {layer.name}: min_ti is NaN/Inf. Using fixed interval for t_max.")
                                 new_t_max = current_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)
                            else:
                                base_interval = torch.tensor(1.0, dtype=torch.float32, device=device)
                                current_layer_t_max = layer.t_max.clone().detach()
                                dynamic_term = gamma * (current_layer_t_max - min_ti_current)
                                dynamic_term = torch.clamp(dynamic_term, min=0.0)
                                new_t_max = current_t_min + torch.maximum(base_interval, dynamic_term)

                        k += 1
                    else:
                        new_t_max = current_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)

                    layer.set_time_params(current_t_min_prev, current_t_min, new_t_max)

                    current_t_min_prev = current_t_min.clone()
                    current_t_min = new_t_max.clone()

                elif isinstance(layer, nn.Flatten):
                     pass


        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        processed_batches += 1

    if total_samples == 0:
        print("警告: Epoch 中没有处理任何样本。")
        return 0.0, 0.0

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc


def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, List, List]:
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []
    processed_batches = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            try:
                outputs, _ = model(features)
            except Exception as e:
                 print(f"错误发生在评估前向传播: {e}")
                 continue

            if not torch.all(torch.isfinite(outputs)):
                print(f"警告: 评估期间模型输出包含 NaN/Inf。跳过此批次。")
                continue

            try:
                loss = criterion(outputs, labels)
            except Exception as e:
                 print(f"错误发生在评估损失计算: {e}")
                 continue

            if not torch.isfinite(loss):
                 print(f"警告: 评估期间计算出的损失为 NaN/Inf。跳过此批次。")
                 continue

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            processed_batches +=1

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    if total_samples == 0:
        print("警告: 评估期间没有处理任何样本。")
        return 0.0, 0.0, [], []

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc, all_labels, all_preds

# Function to plot training history
def plot_history(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='训练损失 (Training Loss)')
    plt.plot(epochs, test_losses, 'ro-', label='测试损失 (Test Loss)')
    plt.title('训练和测试损失 (Training and Test Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='训练准确率 (Training Accuracy)')
    plt.plot(epochs, test_accuracies, 'ro-', label='测试准确率 (Test Accuracy)')
    plt.title('训练和测试准确率 (Training and Test Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('准确率 (Accuracy)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    print("--- 开始加载数据 ---")
    try:
        features, labels = load_features_from_mat(FEATURE_DIR)
        print(f"成功加载数据。特征形状: {features.shape}, 标签形状: {labels.shape}")
    except ValueError as e:
         print(f"数据加载失败: {e}")
         exit()
    print("--- 数据加载完成 ---")

    print("--- 开始划分数据集 ---")
    try:
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            features, labels, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=labels
        )
        print(f"训练集大小: {len(X_train_orig)}")
        print(f"测试集大小: {len(X_test_orig)}")
    except Exception as e:
         print(f"数据集划分失败: {e}")
         exit()
    print("--- 数据集划分完成 ---")

    print("--- 开始特征归一化 ---")
    scaler = MinMaxScaler(feature_range=(0, 1))
    try:
        X_train_norm = scaler.fit_transform(X_train_orig)
        X_test_norm = scaler.transform(X_test_orig)
    except Exception as e:
         print(f"特征归一化失败: {e}")
         exit()
    print("特征已归一化到 [0, 1]。")
    print("--- 特征归一化完成 ---")

    print("--- 开始 TTFS 编码 ---")
    X_train_encoded = ttfs_encode(X_train_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    X_test_encoded = ttfs_encode(X_test_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    print(f"特征已编码为脉冲时间。范围: [{T_MIN_INPUT}, {T_MAX_INPUT}]")
    print(f"编码后训练数据形状: {X_train_encoded.shape}")
    print("--- TTFS 编码完成 ---")

    print("--- 创建 PyTorch 数据集和加载器 ---")
    try:
        train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
        test_dataset = EncodedEEGDataset(X_test_encoded, y_test)
        num_workers = 0
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    except Exception as e:
         print(f"创建数据集/加载器失败: {e}")
         exit()
    print("--- 数据集和加载器创建完成 ---")

    print("--- 构建 SNN 模型 ---")
    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense_1', input_dim=INPUT_SIZE,
                           outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense_2',
                           outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='dense_output', outputLayer=True,
                           kernel_initializer='glorot_uniform'))

    print("SNN 模型结构:")
    print(model)
    model.to(device)
    print("--- SNN 模型构建完成 ---")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"损失函数: {criterion}")
    print(f"优化器: {optimizer}")

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

    print(f"\n--- 开始 SNN 训练 ---")
    start_time = time.time()
    best_test_acc = 0.0
    epochs_without_improvement = 0

    # Lists to store metrics for plotting
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                            gamma=TRAINING_GAMMA)

        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if not np.isfinite(train_loss) or not np.isfinite(train_acc) or \
           not np.isfinite(test_loss) or not np.isfinite(test_acc):
            print(f"错误: Epoch {epoch+1} 产生 NaN/Inf 值。停止训练。")
            break

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | 训练 损失: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"测试 损失: {test_loss:.4f}, Acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
             best_test_acc = test_acc
             print(f"    * 新的最佳测试准确率: {best_test_acc:.4f}")
             epochs_without_improvement = 0
        else:
             epochs_without_improvement += 1
             if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                 print(f"\n测试准确率连续 {EARLY_STOPPING_PATIENCE} 轮没有改进。触发早停。")
                 break

    end_time = time.time()
    print(f"\n--- SNN 训练完成 (或提前停止) ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"训练期间最佳测试准确率: {best_test_acc:.4f}")

    # --- Plotting ---
    if train_losses and test_losses and train_accuracies and test_accuracies:
        print("\n--- 生成训练历史图 ---")
        plot_history(train_losses, test_losses, train_accuracies, test_accuracies)
    else:
        print("\n未能收集到足够的训练历史数据以生成图表。")


    print("\n--- 在测试集上进行最终评估 ---")
    # Evaluate the model state at the end of training (or early stopping)
    final_test_loss, final_test_acc, final_labels, final_preds = evaluate_model(model, test_loader, criterion, device)

    if not np.isfinite(final_test_loss) or not np.isfinite(final_test_acc):
         print("错误: 最终评估产生 NaN/Inf 值。")
    else:
        print(f"最终测试损失: {final_test_loss:.4f}")
        print(f"最终测试准确率: {final_test_acc:.4f}")

        if final_labels and final_preds:
            report_target_names = ['负面 (0)', '中性 (1)', '正面 (2)']
            try:
                print("\n分类报告 (测试集):")
                unique_final_labels = np.unique(final_labels)
                unique_final_preds = np.unique(final_preds)
                print(classification_report(final_labels, final_preds,
                                            target_names=report_target_names,
                                            digits=4,
                                            zero_division=0))
            except ValueError as e:
                 print(f"生成分类报告时出错: {e}")
                 print("检查标签和预测值是否包含所有预期的类别。")
                 print(f"唯一标签: {np.unique(final_labels)}")
                 print(f"唯一预测: {np.unique(final_preds)}")
            except Exception as e:
                 print(f"生成分类报告时发生未知错误: {e}")
        else:
            print("无最终评估结果可用于生成分类报告。")


    print("\n脚本运行结束。")