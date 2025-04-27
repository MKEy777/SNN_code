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
import time
import warnings
from module.TTFS_ORIGN import SNNModel, SpikingDense

# 特征文件所在目录
FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features"
if not os.path.isdir(FEATURE_DIR):
    print(f"错误: 特征目录未找到: {FEATURE_DIR}")
    exit()

# 网络参数配置
INPUT_SIZE = 4216
HIDDEN_UNITS_1 = 512
HIDDEN_UNITS_2 = 128
OUTPUT_SIZE = 3
X_N_HIDDEN = 1.0
X_N_OUTPUT = 1.0
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0

# 训练超参数配置
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 50
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

def load_features_from_mat(feature_dir):
    """
    从指定目录加载所有 *_features.mat 文件，提取特征和标签
    返回: 特征矩阵, 标签数组
    """
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_features.mat"))
    if not mat_files:
        print(f"错误: 在 {feature_dir} 中未找到 '*_features.mat' 文件")
        exit()

    print(f"找到 {len(mat_files)} 个特征文件。正在加载...")
    for fpath in mat_files:
        try:
            mat_data = loadmat(fpath)
            if 'features' in mat_data and 'labels' in mat_data:
                features = mat_data['features']
                labels = mat_data['labels'].flatten()

                # 检查特征维度
                if features.shape[1] != INPUT_SIZE:
                    warnings.warn(f"文件 {os.path.basename(fpath)} 特征维度 {features.shape[1]} != {INPUT_SIZE}，跳过。")
                    continue

                # 检查特征数量和标签数量是否一致
                if features.shape[0] != len(labels):
                    warnings.warn(f"文件 {os.path.basename(fpath)} 特征 ({features.shape[0]}) 和标签 ({len(labels)}) 不匹配，跳过。")
                    continue

                all_features.append(features.astype(np.float32))
                all_labels.append(labels)
        except Exception as e:
            print(f"加载 {os.path.basename(fpath)} 时出错: {e}，跳过。")

    if not all_features:
        print("错误: 未加载任何有效数据。")
        exit()

    # 合并所有数据
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    print(f"总共加载了 {combined_features.shape[0]} 个段。")

    # 标签映射: {-1, 0, 1} -> {0, 1, 2}
    label_mapping = {-1: 0, 0: 1, 1: 2}
    valid_labels_mask = np.isin(combined_labels, list(label_mapping.keys()))
    if not np.all(valid_labels_mask):
        print("错误: 发现无效标签值。")
        print(f"无效标签: {np.unique(combined_labels[~valid_labels_mask])}")
        exit()

    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels], dtype=np.int64)
    print("标签映射已应用。")
    print(f"唯一的映射后标签: {np.unique(mapped_labels)}")

    if len(np.unique(mapped_labels)) != OUTPUT_SIZE:
        warnings.warn(f"警告: 映射后的唯一标签数量 ({len(np.unique(mapped_labels))}) 与 OUTPUT_SIZE ({OUTPUT_SIZE}) 不匹配。")

    return combined_features, mapped_labels


def ttfs_encode(features, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT):
    """
    将归一化特征转为脉冲时间编码 (TTFS)
    时间早表示值大
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    features = np.clip(features, 0.0, 1.0)
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)


class EncodedEEGDataset(Dataset):
    """
    将编码后的特征和标签打包成 Dataset，供 DataLoader 使用
    """
    def __init__(self, encoded_features, labels):
        self.features = encoded_features
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    单轮训练，更新参数，并动态更新各层脉冲时间界限
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        outputs, min_ti_list = model(features)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            t_min_prev = torch.tensor(0.0, device=device)
            t_min = torch.tensor(1.0, device=device)

            k = 0  # min_ti_list 的索引
            for layer in model.layers_list:
                if isinstance(layer, SpikingDense):
                    layer_B_n = layer.B_n.to(device)

                    if not layer.outputLayer:
                        if k < len(min_ti_list):
                            min_ti_current = min_ti_list[k].to(device)
                            if torch.isfinite(min_ti_current):
                                conceptual_t_max = t_min + layer_B_n
                                dynamic_term = 10.0 * (conceptual_t_max - min_ti_current)
                                t_max = t_min + torch.maximum(layer_B_n, dynamic_term)
                            else:
                                t_max = t_min + layer_B_n
                            k += 1
                        else:
                            warnings.warn(f"Warning: min_ti_list exhausted at layer {layer.name}.")
                            t_max = t_min + layer_B_n
                    else:
                        t_max = t_min + layer_B_n

                    layer.update_time_bounds(t_min_prev.clone(), t_min.clone(), t_max.clone())
                    t_min_prev = t_min
                    t_min = t_max

                    if torch.any(t_min <= t_min_prev):
                        t_min = t_min_prev + 1e-6

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    """
    在验证集或测试集上评估模型性能
    """
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
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc, all_labels, all_preds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载并预处理特征数据
    features, labels = load_features_from_mat(FEATURE_DIR)

    # 划分训练集和测试集
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"训练集大小: {len(X_train_orig)}")
    print(f"测试集大小: {len(X_test_orig)}")

    # 特征归一化到 [0, 1]
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train_orig)
    X_test_norm = scaler.transform(X_test_orig)
    print("特征已归一化。")

    # 脉冲时间编码 (TTFS)
    X_train_encoded = ttfs_encode(X_train_norm)
    X_test_encoded = ttfs_encode(X_test_norm)
    print("归一化特征已编码为脉冲时间。")

    # 打包为 Dataset 和 DataLoader
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    test_dataset = EncodedEEGDataset(X_test_encoded, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 构建 SNN 模型
    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense1', input_dim=INPUT_SIZE, X_n=X_N_HIDDEN, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense2', X_n=X_N_HIDDEN, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='output', outputLayer=True, X_n=X_N_OUTPUT, kernel_initializer='glorot_uniform'))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- 开始 SNN 训练 ---")
    start_time = time.time()
    best_test_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | 训练损失: {train_loss:.4f}, Acc: {train_acc:.4f} | 测试损失: {test_loss:.4f}, Acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # 可在此保存最佳模型
            # torch.save(model.state_dict(), 'best_snn_model_tf_timing.pth')

    end_time = time.time()
    print("\n--- SNN 训练完成 ---")
    print(f"耗时: {end_time - start_time:.2f}s | 最佳测试准确率: {best_test_acc:.4f}")

    # 最终评估
    final_test_loss, final_test_acc, final_labels, final_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"最终测试损失: {final_test_loss:.4f}, 最终测试准确率: {final_test_acc:.4f}")

    report_target_names = ['负面 (-1)', '中性 (0)', '正面 (1)']
    try:
        print("\n分类报告:")
        print(classification_report(np.array(final_labels), np.array(final_preds), target_names=report_target_names, digits=4, zero_division=0))
    except ValueError as e:
        print(f"\n无法生成报告: {e}")
    except Exception as e:
        print(f"\n生成报告时发生意外错误: {e}")

    print("\n脚本运行结束。")
