import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR 
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Union, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
import json
from model.TTFS_ORIGN import SNNModel, SpikingDense 
import itertools
import copy

# --- 固定参数设置  ---
FEATURE_DIR = r"Individual_Features_NoBandpass_Fixed_BaselineCorrected"
OUTPUT_DIR_BASE = r"B_8_result_L1_L2_best"
INPUT_SIZE = 682
OUTPUT_SIZE = 3
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
TRAINING_GAMMA = 10.0
NUM_EPOCHS = 150 
L1_LAMBDA = 1e-7 
LR_SCHEDULER_GAMMA = 0.99
L2_LAMBDA=1e-6
# --- 早停参数  ---
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.0005

# --- 超参数搜索空间定义  ---
hyperparameter_grid = {
    'LEARNING_RATE': [5e-4],
    'BATCH_SIZE': [8],
    'HIDDEN_UNITS_1': [512],
    'HIDDEN_UNITS_2': [256],
    'L1_LAMBDA_SEARCH': [1e-7],
    'L2_LAMBDA': [0,1e-6,5e-7,1e-7]  
}
# 加载特征数据
def load_features_from_mat(feature_dir):
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_no_bandpass_features.mat"))
    for fpath in mat_files:
        mat_data = loadmat(fpath)
        features = mat_data['features'].astype(np.float32)
        labels = mat_data['labels'].flatten()
        all_features.append(features)
        all_labels.append(labels)
    if not all_features:
        raise ValueError(f"在目录 {feature_dir} 中没有找到特征文件。")
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    label_mapping = {-1: 0, 0: 1, 1: 2}
    valid_labels_indices = np.isin(combined_labels, list(label_mapping.keys()))
    combined_features_filtered = combined_features[valid_labels_indices]
    combined_labels_filtered = combined_labels[valid_labels_indices]
    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels_filtered], dtype=np.int64)
    return combined_features_filtered, mapped_labels

# TTFS编码 
def ttfs_encode(features: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = np.clip(features, 0.0, 1.0)
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)

# 数据集类 
class EncodedEEGDataset(Dataset):
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray):
        self.features = encoded_features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

def custom_weight_init(m: nn.Module):
    """
    应用自定义权重初始化: W ~ N(0, 1/N_in)
    其中 N_in 是该层的输入特征数量。
    """
    if isinstance(m, SpikingDense):
        if hasattr(m, 'kernel') and m.kernel is not None:
            input_dim_for_layer = m.kernel.shape[0]
            if input_dim_for_layer > 0:
                stddev = 1.0 / np.sqrt(input_dim_for_layer)
                with torch.no_grad():
                    m.kernel.data.normal_(mean=0.0, std=stddev)

# 训练一个周期 
def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                gamma_ttfs: float, current_t_min_input: float, current_t_max_input: float,
                l1_lambda_val: float) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        outputs, min_ti_list = model(features)

        primary_loss = criterion(outputs, labels)

        # --- L1 正则化 (仅应用于第一层 SpikingDense) ---
        l1_penalty = torch.tensor(0.0, device=device)
        if l1_lambda_val > 0:
            if len(model.layers_list) > 0 and \
               isinstance(model.layers_list[0], SpikingDense) and \
               hasattr(model.layers_list[0], 'kernel') and \
               model.layers_list[0].kernel is not None:
                l1_penalty += torch.sum(torch.abs(model.layers_list[0].kernel))

        loss = primary_loss + l1_lambda_val * l1_penalty
        # --- L1 正则化结束 ---

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TTFS 时间参数更新逻辑 
        with torch.no_grad():
            current_t_min_prev_loop = torch.tensor(current_t_min_input, dtype=torch.float32, device=device)
            current_t_min_layer = torch.tensor(current_t_max_input, dtype=torch.float32, device=device)
            t_min_prev_layer = torch.tensor(current_t_min_input, dtype=torch.float32, device=device)
            k = 0
            for layer_idx, layer in enumerate(model.layers_list):
                if isinstance(layer, SpikingDense):
                    new_t_max_layer = torch.tensor(0.0, dtype=torch.float32, device=device)
                    if not layer.outputLayer:
                        min_ti_for_layer = None
                        if k < len(min_ti_list) and min_ti_list[k] is not None:
                            positive_spike_times = min_ti_list[k][min_ti_list[k] > 1e-6]
                            if positive_spike_times.numel() > 0:
                                min_ti_for_layer = torch.min(positive_spike_times)
                        base_interval = torch.tensor(1.0, dtype=torch.float32, device=device)
                        if min_ti_for_layer is not None:
                             current_layer_t_max_for_gamma = layer.t_max.clone().detach()
                             dynamic_term = gamma_ttfs * (current_layer_t_max_for_gamma - min_ti_for_layer)
                             dynamic_term = torch.clamp(dynamic_term, min=0.0)
                             new_t_max_layer = current_t_min_layer + torch.maximum(base_interval, dynamic_term)
                        else:
                             new_t_max_layer = current_t_min_layer + base_interval
                        k += 1
                    else: # 输出层
                        new_t_max_layer = current_t_min_layer + torch.tensor(1.0, dtype=torch.float32, device=device)
                    layer.set_time_params(t_min_prev_layer, current_t_min_layer, new_t_max_layer)
                    t_min_prev_layer = current_t_min_layer.clone()
                    current_t_min_layer = new_t_max_layer.clone()

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

# 评估模型 
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
            primary_loss = criterion(outputs, labels)
            loss = primary_loss
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc, all_labels, all_preds

# 构建文件名前缀 
def build_filename_prefix(params: Dict[str, Union[int, float]]) -> str:
    lr_val = params.get('LEARNING_RATE', fixed_parameters.get('LEARNING_RATE'))
    lr_str = f"{lr_val:.0e}".replace('-', 'm').replace('+', '')
    gamma_ttfs_val = params.get('TRAINING_GAMMA', fixed_parameters.get('TRAINING_GAMMA'))
    gamma_str = str(gamma_ttfs_val).replace('.', 'p')
    lr_decay_gamma_val = params.get('LR_SCHEDULER_GAMMA', fixed_parameters.get('LR_SCHEDULER_GAMMA', 'NA'))
    lr_decay_gamma_str = f"_lrdecay{str(lr_decay_gamma_val).replace('.', 'p')}" if lr_decay_gamma_val != 'NA' else ""
    l2_lambda_val = params.get('L2_LAMBDA', 0.0)
    l2_str = f"_l2_{l2_lambda_val:.0e}".replace('-', 'm').replace('+', '') if l2_lambda_val > 0 else ""
    prefix = (f"lr{lr_str}_bs{params['BATCH_SIZE']}_epochsMax{params['NUM_EPOCHS']}"
              f"_h1_{params['HIDDEN_UNITS_1']}_h2_{params['HIDDEN_UNITS_2']}"
              f"_gammaTTFS{gamma_str}{lr_decay_gamma_str}_seed{params['RANDOM_SEED']}{l2_str}")
    return prefix

# 绘制训练历史曲线 
def plot_history(train_losses, val_losses, train_accuracies, val_accuracies, train_lrs, filename_prefix: str, save_dir: str, stopped_epoch: Optional[int] = None):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))
    title_suffix = f" (Stopped at epoch {stopped_epoch})" if stopped_epoch else ""
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'Loss curve{title_suffix}')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title(f'Accuracy curve{title_suffix}')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_lrs, 'go-', label='Learning Rate')
    plt.title(f'Learning Rate curve{title_suffix}')
    plt.xlabel('Epochs'); plt.ylabel('Learning Rate'); plt.legend(); plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"history_and_lr_{filename_prefix}_{timestamp}.png")
    plt.savefig(filename); plt.close()
    print(f"训练历史和学习率曲线图已保存为 {filename}")

# 保存模型 (保持不变)
def save_model_torch(model: SNNModel, filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_{filename_prefix}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型已成功保存到: {save_path}")

# 保存参数 
def save_params(params: Dict[str, Union[int, float, str]], filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"params_{filename_prefix}_{timestamp}.json")
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, np.integer): serializable_params[k] = int(v)
        elif isinstance(v, np.floating): serializable_params[k] = float(v)
        elif isinstance(v, np.ndarray): serializable_params[k] = v.tolist()
        else: serializable_params[k] = v
    if 'LR_SCHEDULER_GAMMA' not in serializable_params and 'LR_SCHEDULER_GAMMA' in params:
         serializable_params['LR_SCHEDULER_GAMMA'] = params['LR_SCHEDULER_GAMMA']
    if 'L1_LAMBDA_EFFECTIVE' not in serializable_params and 'L1_LAMBDA_EFFECTIVE' in params:
         serializable_params['L1_LAMBDA_EFFECTIVE'] = params['L1_LAMBDA_EFFECTIVE']
    if 'L2_LAMBDA' not in serializable_params and 'L2_LAMBDA' in params:
         serializable_params['L2_LAMBDA'] = params['L2_LAMBDA']
    with open(save_path, 'w') as f: json.dump(serializable_params, f, indent=4)
    print(f"训练参数已保存到: {save_path}")

# --- 封装的训练和评估函数 ---
def run_training_session(current_hyperparams: Dict, fixed_params_dict: Dict, run_id: int):
    all_params = {**fixed_params_dict, **current_hyperparams}

    LEARNING_RATE_INITIAL = all_params['LEARNING_RATE']
    BATCH_SIZE = all_params['BATCH_SIZE']
    _MAX_NUM_EPOCHS = all_params['NUM_EPOCHS']
    _TRAINING_GAMMA_TTFS = all_params['TRAINING_GAMMA']
    HIDDEN_UNITS_1 = all_params['HIDDEN_UNITS_1']
    HIDDEN_UNITS_2 = all_params['HIDDEN_UNITS_2']
    _L1_LAMBDA_CURRENT = current_hyperparams.get('L1_LAMBDA_SEARCH', all_params['L1_LAMBDA'])
    _L2_LAMBDA_CURRENT = current_hyperparams.get('L2_LAMBDA', 0.0)  # Get L2 regularization strength
    _LR_SCHEDULER_GAMMA = all_params['LR_SCHEDULER_GAMMA']
    _EARLY_STOPPING_PATIENCE = all_params['EARLY_STOPPING_PATIENCE']
    _EARLY_STOPPING_MIN_DELTA = all_params['EARLY_STOPPING_MIN_DELTA']

    l1_tag = f"_l1_{_L1_LAMBDA_CURRENT:.0e}".replace('-', 'm').replace('+', '') if _L1_LAMBDA_CURRENT > 0 else "_l1_0"
    l2_tag = f"_l2_{_L2_LAMBDA_CURRENT:.0e}".replace('-', 'm').replace('+', '') if _L2_LAMBDA_CURRENT > 0 else ""

    all_params_for_naming_and_saving = all_params.copy()
    all_params_for_naming_and_saving['L1_LAMBDA_EFFECTIVE'] = _L1_LAMBDA_CURRENT
    all_params_for_naming_and_saving['L2_LAMBDA'] = _L2_LAMBDA_CURRENT
    all_params_for_naming_and_saving['LR_SCHEDULER_GAMMA'] = _LR_SCHEDULER_GAMMA
    all_params_for_naming_and_saving['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))'

    base_prefix = build_filename_prefix(all_params_for_naming_and_saving)
    run_specific_output_dir_name = f"{base_prefix}{l1_tag}{l2_tag}_val_loss_stop_customInit"
    run_specific_output_dir = os.path.join(OUTPUT_DIR_BASE, run_specific_output_dir_name)

    if not os.path.exists(run_specific_output_dir):
        os.makedirs(run_specific_output_dir, exist_ok=True)

    filename_prefix = f"final_best_val_loss_early_stop{l1_tag}{l2_tag}_customInit"

    print(f"\n--- 开始运行 ID: {run_id} ---")
    print(f"完整参数 (用于此运行): {all_params_for_naming_and_saving}")
    print(f"L1 正则化强度: {_L1_LAMBDA_CURRENT}")
    print(f"L2 正则化强度: {_L2_LAMBDA_CURRENT}")
    print(f"初始学习率: {LEARNING_RATE_INITIAL}, 学习率衰减 Gamma: {_LR_SCHEDULER_GAMMA}")
    print(f"权重初始化: N(0, 1/sqrt(N_in))")
    print(f"早停设置 (基于验证损失): Patience={_EARLY_STOPPING_PATIENCE}, Min Delta={_EARLY_STOPPING_MIN_DELTA}")
    print(f"结果将保存在: {run_specific_output_dir}")

    save_params(all_params_for_naming_and_saving, filename_prefix, run_specific_output_dir)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    features_data, labels_data = load_features_from_mat(FEATURE_DIR)
    print(f"成功加载数据。特征形状: {features_data.shape}, 标签形状: {labels_data.shape}")

    unique_labels_data, counts_data = np.unique(labels_data, return_counts=True)
    stratify_option = labels_data if all(count >= 2 for count in counts_data[counts_data > 0]) and len(unique_labels_data[counts_data > 0]) >= 2 else None
    if stratify_option is None: print("警告: 数据集不满足分层抽样的条件。使用普通随机抽样。")
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(
        features_data, labels_data, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=stratify_option
    )
    print(f"训练集大小: {len(X_train_orig)}")
    print(f"验证集大小: {len(X_val_orig)}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train_orig)
    X_val_norm = scaler.transform(X_val_orig)
    X_train_encoded = ttfs_encode(X_train_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    X_val_encoded = ttfs_encode(X_val_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)

    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    val_dataset = EncodedEEGDataset(X_val_encoded, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense_1', input_dim=INPUT_SIZE, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense_2', input_dim=HIDDEN_UNITS_1, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='dense_output', input_dim=HIDDEN_UNITS_2, outputLayer=True, kernel_initializer='glorot_uniform'))

    print("应用自定义权重初始化 N(0, 1/sqrt(N_in))...")
    model.apply(custom_weight_init)
    print("自定义权重初始化完成。")

    model.to(device)
    print(f"模型总可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL, weight_decay=_L2_LAMBDA_CURRENT)  # Added L2 regularization
    scheduler = ExponentialLR(optimizer, gamma=_LR_SCHEDULER_GAMMA)

    with torch.no_grad():
        init_t_min_prev = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
        init_t_min = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
        for layer in model.layers_list:
            if isinstance(layer, SpikingDense):
                init_t_max = init_t_min + torch.tensor(1.0, dtype=torch.float32, device=device)
                layer.set_time_params(init_t_min_prev, init_t_min, init_t_max)
                init_t_min_prev = init_t_min.clone()
                init_t_min = init_t_max.clone()

    start_time_run = time.time()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    learning_rates_over_epochs = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state_dict = None
    stopped_epoch = None

    print(f"开始训练，最多 {_MAX_NUM_EPOCHS} 个周期...")
    for epoch in range(_MAX_NUM_EPOCHS):
        epoch_start_time = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates_over_epochs.append(current_lr)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                            gamma_ttfs=_TRAINING_GAMMA_TTFS,
                                            current_t_min_input=T_MIN_INPUT,
                                            current_t_max_input=T_MAX_INPUT,
                                            l1_lambda_val=_L1_LAMBDA_CURRENT)

        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        epoch_end_time = time.time()

        scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"周期 [{epoch+1}/{_MAX_NUM_EPOCHS}] | LR: {current_lr:.2e} | 训练损失: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f}, Acc: {val_acc:.4f} | 耗时: {epoch_end_time - epoch_start_time:.2f} 秒")

        if val_loss < best_val_loss - _EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  验证损失改善至 {best_val_loss:.4f}。重置耐心计数器。")
        else:
            patience_counter += 1
            print(f"  验证损失未显著改善。耐心计数器: {patience_counter}/{_EARLY_STOPPING_PATIENCE}")

        if patience_counter >= _EARLY_STOPPING_PATIENCE:
            print(f"早停触发于周期 {epoch+1}。最大耐心 {_EARLY_STOPPING_PATIENCE} 已达到 (基于验证损失)。")
            stopped_epoch = epoch + 1
            break

    if stopped_epoch is None:
        stopped_epoch = _MAX_NUM_EPOCHS
        print(f"训练完成 {_MAX_NUM_EPOCHS} 个周期。")
        if best_model_state_dict is None or (val_losses and val_losses[-1] < best_val_loss - _EARLY_STOPPING_MIN_DELTA):
             if val_losses:
                best_val_loss = val_losses[-1]
                best_model_state_dict = copy.deepcopy(model.state_dict())
                print(f"  训练结束，记录最后周期的模型状态，验证损失: {best_val_loss:.4f}")

    if best_model_state_dict is not None:
        print("加载在验证集上表现最佳的模型权重 (基于损失)。")
        model.load_state_dict(best_model_state_dict)
    else:
        print("警告: 未找到最佳模型状态 (基于验证损失)。将使用最后一个周期的模型。")

    end_time_run = time.time()
    print(f"--- SNN 训练完成 (运行 ID: {run_id}), 总耗时: {end_time_run - start_time_run:.2f} 秒, 实际训练周期: {stopped_epoch} ---")

    save_model_torch(model, filename_prefix, run_specific_output_dir)
    plot_history(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates_over_epochs,
                 filename_prefix, run_specific_output_dir, stopped_epoch=stopped_epoch)

    final_val_loss, final_val_acc, final_labels, final_preds = evaluate_model(model, val_loader, criterion, device)
    print(f"最终验证损失 (来自最佳模型基于损失): {final_val_loss:.4f}")
    print(f"最终验证准确率 (来自最佳模型基于损失): {final_val_acc:.4f}")

    report_names = ['负面 (0)', '中性 (1)', '正面 (2)']
    report = classification_report(final_labels, final_preds, target_names=report_names, digits=4, zero_division=0)
    print("\n分类报告 (基于验证集和最佳模型基于损失):")
    print(report)

    report_filename = os.path.join(run_specific_output_dir, f"classification_report_{filename_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_filename, 'w') as f:
        params_to_save_report = {**all_params_for_naming_and_saving, 'stopped_epoch': stopped_epoch, 'best_validation_accuracy_achieved': final_val_acc, 'weight_initialization_note': 'N(0, 1/sqrt(N_in)) applied after __init__'}
        f.write(f"Hyperparameters: {json.dumps(params_to_save_report, indent=4)}\n\n")
        f.write(f"Final Validation Loss (best model on val_loss): {final_val_loss:.4f}\n")
        f.write(f"Final Validation Accuracy (best model on val_loss): {final_val_acc:.4f}\n\n")
        f.write("Classification Report (on validation set with best model based on val_loss):\n")
        f.write(report)
    print(f"分类报告已保存到: {report_filename}")
    print(f"--- 运行 ID: {run_id} 完成 ---")
    return final_val_acc

# 主程序 
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

    fixed_parameters = {
        'FEATURE_DIR': FEATURE_DIR,
        'INPUT_SIZE': INPUT_SIZE,
        'OUTPUT_SIZE': OUTPUT_SIZE,
        'T_MIN_INPUT': T_MIN_INPUT,
        'T_MAX_INPUT': T_MAX_INPUT,
        'TEST_SPLIT_SIZE': TEST_SPLIT_SIZE,
        'RANDOM_SEED': RANDOM_SEED,
        'TRAINING_GAMMA': TRAINING_GAMMA,
        'NUM_EPOCHS': NUM_EPOCHS,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'EARLY_STOPPING_MIN_DELTA': EARLY_STOPPING_MIN_DELTA,
        'L1_LAMBDA': L1_LAMBDA,
        'LR_SCHEDULER_GAMMA': LR_SCHEDULER_GAMMA
    }

    if 'L1_LAMBDA_SEARCH' not in hyperparameter_grid:
         print(f"L1_LAMBDA_SEARCH 未在 hyperparameter_grid 中找到, 将使用固定的 L1_LAMBDA: {fixed_parameters['L1_LAMBDA']}")
    else:
        print(f"L1_LAMBDA_SEARCH 将从 grid 中选择: {hyperparameter_grid['L1_LAMBDA_SEARCH']}")

    keys, values = zip(*hyperparameter_grid.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    num_combinations = len(hyperparam_combinations)
    print(f"将要进行 {num_combinations} 组超参数的训练。")
    print(f"固定参数 (部分): TRAINING_GAMMA (TTFS) = {TRAINING_GAMMA}, MAX_NUM_EPOCHS = {NUM_EPOCHS}, LR_SCHEDULER_GAMMA = {LR_SCHEDULER_GAMMA}")
    if 'L1_LAMBDA_SEARCH' not in hyperparameter_grid:
        print(f"固定 L1 正则化强度 (L1_LAMBDA): {L1_LAMBDA}")
    print(f"权重初始化: N(0, 1/sqrt(N_in))")
    print(f"早停参数 (基于验证损失): Patience = {EARLY_STOPPING_PATIENCE}, Min Delta = {EARLY_STOPPING_MIN_DELTA}")

    best_accuracy_overall = 0.0
    best_hyperparams_combo_overall = None
    all_results = []

    for i, params_combo_iter in enumerate(hyperparam_combinations):
        validation_accuracy_for_run = run_training_session(current_hyperparams=params_combo_iter,
                                                           fixed_params_dict=fixed_parameters,
                                                           run_id=i+1)

        full_params_for_log = {**fixed_parameters, **params_combo_iter}
        l1_eff = params_combo_iter.get('L1_LAMBDA_SEARCH', fixed_parameters['L1_LAMBDA'])
        full_params_for_log['L1_LAMBDA_EFFECTIVE'] = l1_eff
        full_params_for_log['L2_LAMBDA'] = params_combo_iter.get('L2_LAMBDA', 0.0)
        full_params_for_log['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))'

        all_results.append({'id': i+1, 'params': full_params_for_log, 'best_validation_accuracy_for_this_run (from best_loss_model)': validation_accuracy_for_run})

        if validation_accuracy_for_run > best_accuracy_overall:
            best_accuracy_overall = validation_accuracy_for_run
            best_hyperparams_combo_overall = full_params_for_log

    print("\n--- 所有超参数遍历完成 ---")
    if best_hyperparams_combo_overall:
        print(f"所有运行中最佳验证准确率 (来自基于损失早停选择的模型): {best_accuracy_overall:.4f}")
        print(f"对应的最佳超参数组合: {json.dumps(best_hyperparams_combo_overall, indent=2)}")

        l1_val_for_summary = best_hyperparams_combo_overall.get('L1_LAMBDA_EFFECTIVE', fixed_parameters['L1_LAMBDA'])
        l2_val_for_summary = best_hyperparams_combo_overall.get('L2_LAMBDA', 0.0)
        summary_file_suffix = f"grid_search_summary_val_loss_early_stop_L1_{l1_val_for_summary:.0e}_L2_{l2_val_for_summary:.0e}_customInit.json"
        summary_file_name = summary_file_suffix.replace(':', '_').replace('-', 'm').replace('+', '')
        summary_file = os.path.join(OUTPUT_DIR_BASE, summary_file_name)

        summary_data = {
            "best_overall_validation_accuracy (from_best_loss_model)": best_accuracy_overall,
            "best_overall_hyperparameters": best_hyperparams_combo_overall,
            "all_run_results": all_results,
            "early_stopping_criterion": "validation_loss",
            "weight_initialization": "N(0, 1/sqrt(N_in))",
            "learning_rate_policy": f"Exponential Decay with gamma={LR_SCHEDULER_GAMMA}"
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"所有运行的结果摘要已保存到: {summary_file}")
    else:
        print("没有成功的训练运行或所有运行的准确率均为0。")

    print("脚本运行结束。")