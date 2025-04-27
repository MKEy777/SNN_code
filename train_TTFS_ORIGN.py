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
# Assuming TTFS_ORIGN.py is in a 'module' subdirectory relative to this script
# If not, adjust the import path accordingly
from module.TTFS_ORIGN import SNNModel, SpikingDense

# --- 配置参数 ---
FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features" # 特征文件所在目录 (示例路径)
# Removed: Directory existence check (os.path.isdir)

# Model structure params (kept as is)
INPUT_SIZE = 4216
HIDDEN_UNITS_1 = 512
HIDDEN_UNITS_2 = 128
OUTPUT_SIZE = 3
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0

# Training params (kept as is)
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 50
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 10
TRAINING_GAMMA = 10.0

# --- 数据加载与预处理 ---

def load_features_from_mat(feature_dir):
    """
    从指定目录加载所有 *_features.mat 文件中的特征和标签。
    执行标签映射 (-1->0, 0->1, 1->2)。

    Args:
        feature_dir (str): 包含 .mat 特征文件的目录。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - combined_features (np.ndarray): 合并后的所有特征数据 (N, INPUT_SIZE)。
            - mapped_labels (np.ndarray): 映射后的标签 (N,)。
    """
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_features.mat"))
    # Removed: Check if mat_files is empty

    print(f"找到 {len(mat_files)} 个特征文件。正在加载...")

    for fpath in mat_files:
        fname = os.path.basename(fpath)
        try: # Keep basic try-except for file loading robustness
            mat_data = loadmat(fpath)
            # Removed: Check if 'features' and 'labels' keys exist
            # Removed: Dimension check (features.shape[1] != INPUT_SIZE)
            # Removed: Feature/label count mismatch check
            features = mat_data['features']
            labels = mat_data['labels'].flatten()

            all_features.append(features.astype(np.float32))
            all_labels.append(labels)
        except Exception as e:
            print(f"加载文件 {fname} 时出错: {e}。已跳过。") # Keep this basic error handling

    # Removed: Check if all_features is empty

    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    print(f"总共加载了 {combined_features.shape[0]} 个数据段。")

    label_mapping = {-1: 0, 0: 1, 1: 2}
    # Removed: Check for invalid label values (np.isin)

    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels], dtype=np.int64)
    print("标签映射已应用: {-1: 0 (负面), 0: 1 (中性), 1: 2 (正面)}")
    unique_mapped_labels = np.unique(mapped_labels)
    print(f"唯一的映射后标签值: {unique_mapped_labels}")

    # Removed: Check if unique_mapped_labels count matches OUTPUT_SIZE

    return combined_features, mapped_labels


def ttfs_encode(features: np.ndarray, t_min: float = T_MIN_INPUT, t_max: float = T_MAX_INPUT) -> torch.Tensor:
    """
    使用 TTFS 编码将归一化特征 [0, 1] 转换为脉冲时间。
    编码公式: spike_time = t_max - feature * (t_max - t_min)
    这对应论文中 tau_c = t_max - t_min 的情况。

    Args:
        features (np.ndarray): 归一化特征 (N, n_features)。
        t_min (float): 编码最小脉冲时间。
        t_max (float): 编码最大脉冲时间。

    Returns:
        torch.Tensor: 编码后的脉冲时间 (N, n_features)，float32。
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = np.clip(features, 0.0, 1.0) # Keep clip as it's part of the encoding logic
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)


class EncodedEEGDataset(Dataset):
    """PyTorch 数据集类"""
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray):
        self.features = encoded_features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# --- 训练与评估 ---

def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                initial_t_min: float = 1.0, gamma: float = TRAINING_GAMMA) -> Tuple[float, float]:
    """Trains the SNN model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    processed_batches = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        # Removed: try-except block around model forward pass
        outputs, min_ti_list = model(features)

        # Removed: Check for NaN/Inf in outputs

        # Removed: try-except block around loss calculation
        loss = criterion(outputs, labels)
        # Removed: Check for NaN/Inf in loss
        # Removed: Check if loss requires grad

        optimizer.zero_grad()
        # Removed: try-except block around backward/step
        loss.backward()
        optimizer.step()


        # --- 更新层的时间边界 ---
        with torch.no_grad():
            t_min_prev_for_update = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
            t_min_for_update = torch.tensor(initial_t_min, dtype=torch.float32, device=device)
            k = 0

            for i, layer in enumerate(model.layers_list):
                if isinstance(layer, SpikingDense):
                    previous_t_max = layer.t_max.clone().detach()
                    current_t_min = layer.set_params(t_min_prev_for_update, t_min_for_update)

                    t_max_for_current_layer = None
                    if not layer.outputLayer:
                        # Removed: Check if k < len(min_ti_list)
                        min_ti_current_layer = min_ti_list[k]

                        # Removed: Check if min_ti_current_layer is valid and < previous_t_max
                        base_interval = torch.tensor(1.0, device=device, dtype=torch.float32)
                        dynamic_term = gamma * (previous_t_max - min_ti_current_layer)
                        t_max_for_current_layer = current_t_min + torch.maximum(base_interval, dynamic_term)
                        k += 1
                    else:
                        # Output layer t_max update (remains simple)
                        t_max_for_current_layer = current_t_min + torch.tensor(1.0, device=device, dtype=torch.float32)

                    # Removed: Check if t_max_for_current_layer is None
                    layer.t_max.copy_(t_max_for_current_layer)

                    t_min_prev_for_update = current_t_min
                    t_min_for_update = layer.t_max.clone().detach()

                    # Removed: Check for time window collapse (t_min_for_update <= t_min_prev_for_update)


        # --- 累积损失和准确率 ---
        # Assuming loss and outputs are valid now
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        processed_batches += 1

    # Removed: Check if processed_batches > 0
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    # Removed: Warning if no batches/samples processed

    return epoch_loss, epoch_acc


def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, List, List]:
    """Evaluates the model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []
    processed_batches = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            # Removed: try-except around model forward pass
            outputs, _ = model(features)
            # Removed: Check for NaN/Inf in outputs

            # Removed: try-except around loss calculation
            loss = criterion(outputs, labels)
            # Removed: Check for NaN/Inf in loss

            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            processed_batches +=1

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Removed: Check if processed_batches > 0
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    # Removed: Warning if no batches/samples processed

    return epoch_loss, epoch_acc, all_labels, all_preds


# --- 主执行流程 ---
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    print("--- 开始加载数据 ---")
    features, labels = load_features_from_mat(FEATURE_DIR)
    print("--- 数据加载完成 ---")

    print("--- 开始划分数据集 ---")
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"训练集大小: {len(X_train_orig)}")
    print(f"测试集大小: {len(X_test_orig)}")
    print("--- 数据集划分完成 ---")

    print("--- 开始特征归一化 ---")
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train_orig)
    X_test_norm = scaler.transform(X_test_orig)
    print("特征已归一化到 [0, 1]。")
    print("--- 特征归一化完成 ---")

    print("--- 开始 TTFS 编码 ---")
    X_train_encoded = ttfs_encode(X_train_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    X_test_encoded = ttfs_encode(X_test_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    print(f"特征已编码为脉冲时间。范围: [{T_MIN_INPUT}, {T_MAX_INPUT}]")
    print(f"编码后训练数据形状: {X_train_encoded.shape}")
    print("--- TTFS 编码完成 ---")

    print("--- 创建 PyTorch 数据集和加载器 ---")
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    test_dataset = EncodedEEGDataset(X_test_encoded, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    print("--- 数据集和加载器创建完成 ---")

    print("--- 构建 SNN 模型  ---")
    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense1', input_dim=INPUT_SIZE,
                           outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense2',
                           outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='output', outputLayer=True,
                           kernel_initializer='glorot_uniform'))
    print("SNN 模型结构:")
    print(model)
    model.to(device)
    print("--- SNN 模型构建完成 ---")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"损失函数: {criterion}")
    print(f"优化器: {optimizer}")

    print(f"\n--- 开始 SNN 训练  ---")
    start_time = time.time()
    best_test_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                            initial_t_min=T_MAX_INPUT, gamma=TRAINING_GAMMA)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        # Removed: Check for NaN values in train/test loss/acc

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | 训练 损失: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"测试 损失: {test_loss:.4f}, Acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
             best_test_acc = test_acc
             print(f"    * 新的最佳测试准确率: {best_test_acc:.4f}")
             epochs_without_improvement = 0
             # torch.save(model.state_dict(), 'best_snn_model_paper_aligned.pth') # Optional save
        else:
             epochs_without_improvement += 1
             if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                 print(f"\n测试准确率连续 {EARLY_STOPPING_PATIENCE} 轮没有改进。触发早停。")
                 break

    end_time = time.time()
    print(f"\n--- SNN 训练完成 (或提前停止) ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"训练期间最佳测试准确率: {best_test_acc:.4f}")

    print("\n--- 在测试集上进行最终评估 ---")
    final_test_loss, final_test_acc, final_labels, final_preds = evaluate_model(model, test_loader, criterion, device)

    # Removed: Check for NaN in final test loss/acc

    print(f"最终测试损失: {final_test_loss:.4f}")
    print(f"最终测试准确率: {final_test_acc:.4f}")

    report_target_names = ['负面 (0)', '中性 (1)', '正面 (2)']
    # Removed: try-except around classification_report
    print("\n分类报告 (测试集):")
    final_labels_np = np.array(final_labels)
    final_preds_np = np.array(final_preds)
    # Removed: Check if labels/preds are empty
    print(classification_report(final_labels_np, final_preds_np,
                                target_names=report_target_names, digits=4, zero_division=0))


    print("\n脚本运行结束。")