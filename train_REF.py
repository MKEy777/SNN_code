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
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
from joblib import parallel_backend
from typing import Dict, Union, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
import json
from model.TTFS_ORIGN import SNNModel, SpikingDense 
import itertools
import copy

FEATURE_DIR = r"Individual_Features_BaselineCorrected_Variable_MP"
OUTPUT_DIR_BASE = r"REF_result"
INPUT_SIZE = 682  
OUTPUT_SIZE = 3
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
TRAINING_GAMMA = 10.0
NUM_EPOCHS = 150 
LR_SCHEDULER_GAMMA = 0.99

EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.0005

hyperparameter_grid = {
    'LEARNING_RATE': [5e-4],
    'BATCH_SIZE': [8],
    'HIDDEN_UNITS_1': [256],
    'HIDDEN_UNITS_2': [128],
    'K_FEATURES': [350,400,500]  
}

def load_features_from_mat(feature_dir):
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_features.mat"))
    for fpath in mat_files:
        mat_data = loadmat(fpath)
        features = mat_data['features'].astype(np.float32)
        labels = mat_data['labels'].flatten()
        all_features.append(features)
        all_labels.append(labels)
    if not all_features:
        raise ValueError(f"目录 {feature_dir} 中未找到特征文件。")
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    label_mapping = {-1: 0, 0: 1, 1: 2}
    valid_labels_indices = np.isin(combined_labels, list(label_mapping.keys()))
    combined_features_filtered = combined_features[valid_labels_indices]
    combined_labels_filtered = combined_labels[valid_labels_indices]
    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels_filtered], dtype=np.int64)
    return combined_features_filtered, mapped_labels

def ttfs_encode(features: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
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

def custom_weight_init(m: nn.Module):
    if isinstance(m, SpikingDense):
        if hasattr(m, 'kernel') and m.kernel is not None:
            input_dim_for_layer = m.kernel.shape[0]
            if input_dim_for_layer > 0:
                stddev = 1.0 / np.sqrt(input_dim_for_layer)
                with torch.no_grad():
                    m.kernel.data.normal_(mean=0.0, std=stddev)

def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                gamma_ttfs: float, current_t_min_input: float, current_t_max_input: float, l1_reg: float) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        outputs, min_ti_list = model(features)
        primary_loss = criterion(outputs, labels)
        loss = primary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
                    else:
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

def build_filename_prefix(params: Dict[str, Union[int, float]]) -> str:
    lr_val = params.get('LEARNING_RATE', fixed_parameters.get('LEARNING_RATE'))
    lr_str = f"{lr_val:.0e}".replace('-', 'm').replace('+', '')
    gamma_ttfs_val = params.get('TRAINING_GAMMA', fixed_parameters.get('TRAINING_GAMMA'))
    gamma_str = str(gamma_ttfs_val).replace('.', 'p')
    lr_decay_gamma_val = params.get('LR_SCHEDULER_GAMMA', fixed_parameters.get('LR_SCHEDULER_GAMMA', 'NA'))
    lr_decay_gamma_str = f"_lrdecay{str(lr_decay_gamma_val).replace('.', 'p')}" if lr_decay_gamma_val != 'NA' else ""
    k_features_str = f"_kfeat{params['K_FEATURES']}" if 'K_FEATURES' in params else ""
    prefix = (f"lr{lr_str}_bs{params['BATCH_SIZE']}_epochsMax{params['NUM_EPOCHS']}"
              f"_h1_{params['HIDDEN_UNITS_1']}_h2_{params['HIDDEN_UNITS_2']}"
              f"_gammaTTFS{gamma_str}{lr_decay_gamma_str}_seed{params['RANDOM_SEED']}{k_features_str}")
    return prefix

def plot_history(train_losses, val_losses, train_accuracies, val_accuracies, train_lrs, filename_prefix: str, save_dir: str, stopped_epoch: Optional[int] = None):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))
    title_suffix = f" (停止于第 {stopped_epoch} 轮)" if stopped_epoch else ""
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='训练损失')
    plt.plot(epochs_range, val_losses, 'ro-', label='验证损失')
    plt.title(f'损失曲线{title_suffix}')
    plt.xlabel('轮次'); plt.ylabel('损失'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='训练准确率')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='验证准确率')
    plt.title(f'准确率曲线{title_suffix}')
    plt.xlabel('轮次'); plt.ylabel('准确率'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_lrs, 'go-', label='学习率')
    plt.title(f'学习率曲线{title_suffix}')
    plt.xlabel('轮次'); plt.ylabel('学习率'); plt.legend(); plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"历史记录和学习率_{filename_prefix}_{timestamp}.png")
    plt.savefig(filename); plt.close()
    print(f"训练历史和学习率图表已保存为 {filename}")

def save_model_torch(model: SNNModel, filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"模型_{filename_prefix}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型已成功保存至: {save_path}")

def save_params(params: Dict[str, Union[int, float, str]], filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"参数_{filename_prefix}_{timestamp}.json")
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, np.integer): serializable_params[k] = int(v)
        elif isinstance(v, np.floating): serializable_params[k] = float(v)
        elif isinstance(v, np.ndarray): serializable_params[k] = v.tolist()
        else: serializable_params[k] = v
    if 'LR_SCHEDULER_GAMMA' not in serializable_params and 'LR_SCHEDULER_GAMMA' in params:
         serializable_params['LR_SCHEDULER_GAMMA'] = params['LR_SCHEDULER_GAMMA']
    with open(save_path, 'w') as f: json.dump(serializable_params, f, indent=4)
    print(f"训练参数已保存至: {save_path}")

def run_training_session(current_hyperparams: Dict, fixed_params_dict: Dict, run_id: int):
    global INPUT_SIZE
    all_params = {**fixed_params_dict, **current_hyperparams}
    LEARNING_RATE_INITIAL = all_params['LEARNING_RATE']
    BATCH_SIZE = all_params['BATCH_SIZE']
    _MAX_NUM_EPOCHS = all_params['NUM_EPOCHS']
    _TRAINING_GAMMA_TTFS = all_params['TRAINING_GAMMA']
    HIDDEN_UNITS_1 = all_params['HIDDEN_UNITS_1']
    HIDDEN_UNITS_2 = all_params['HIDDEN_UNITS_2']
    _LR_SCHEDULER_GAMMA = all_params['LR_SCHEDULER_GAMMA']
    _EARLY_STOPPING_PATIENCE = all_params['EARLY_STOPPING_PATIENCE']
    _EARLY_STOPPING_MIN_DELTA = all_params['EARLY_STOPPING_MIN_DELTA']
    L1_REG = all_params['L1_REG']
    all_params_for_naming_and_saving = all_params.copy()
    all_params_for_naming_and_saving['INPUT_SIZE_INITIAL'] = fixed_params_dict['INPUT_SIZE']
    all_params_for_naming_and_saving['LR_SCHEDULER_GAMMA'] = _LR_SCHEDULER_GAMMA
    all_params_for_naming_and_saving['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))'
    base_prefix = build_filename_prefix(all_params_for_naming_and_saving)
    run_specific_output_dir_name = f"{base_prefix}_val_acc_stop_customInit_LinearSVC_RFE"
    run_specific_output_dir = os.path.join(OUTPUT_DIR_BASE, run_specific_output_dir_name)
    if not os.path.exists(run_specific_output_dir):
        os.makedirs(run_specific_output_dir, exist_ok=True)
    filename_prefix = f"最佳验证准确率_早停_自定义初始化_LinearSVC_RFE"
    print(f"\n--- 开始运行 ID: {run_id} ---")
    print(f"本轮完整参数 (INPUT_SIZE 将在RFE后更新): {all_params_for_naming_and_saving}")
    print(f"初始学习率: {LEARNING_RATE_INITIAL}, 学习率衰减Gamma: {_LR_SCHEDULER_GAMMA}")
    print(f"权重初始化: N(0, 1/sqrt(N_in))")
    print(f"早停设置（基于验证准确率）: 耐心={_EARLY_STOPPING_PATIENCE}, 最小增量={_EARLY_STOPPING_MIN_DELTA}")
    print(f"结果将保存至: {run_specific_output_dir}")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    features_data, labels_data = load_features_from_mat(FEATURE_DIR)
    print(f"数据加载成功。特征形状: {features_data.shape}, 标签形状: {labels_data.shape}")
    num_original_features = features_data.shape[1]
    # 先分割数据为训练集和验证集
    unique_labels, counts = np.unique(labels_data, return_counts=True)
    stratify_option = labels_data if all(count >= 2 for count in counts) and len(unique_labels) >= 2 else None
    if stratify_option is None:
        print("警告: 数据集不满足分层采样条件（可能由于RFE前的标签分布或标签本身），将不使用分层进行数据分割。")
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(
        features_data, labels_data,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_option
    )
    print(f"训练集大小: {len(X_train_orig)} (特征维度: {X_train_orig.shape[1]}), 验证集大小: {len(X_val_orig)} (特征维度: {X_val_orig.shape[1]})")
    # 在训练集上应用 RFE 进行特征选择
    print("正在应用递归特征消除（RFE）进行特征选择...")
    estimator = LinearSVC(dual="auto", C=0.1, max_iter=2000)
    rfe_step = 20
    print(f"RFE 使用的评估器: LinearSVC(dual=\"auto\", C=0.1, max_iter=2000)")
    print(f"RFE 使用的 step 参数: {rfe_step}")
    k_features_to_select = min(all_params['K_FEATURES'], num_original_features)
    if k_features_to_select < all_params['K_FEATURES']:
        print(f"警告: K_FEATURES ({all_params['K_FEATURES']}) 大于原始特征数 ({num_original_features})。将选择 {k_features_to_select} 个特征。")
    if k_features_to_select == num_original_features:
        print("K_FEATURES 等于原始特征数，RFE 将选择所有特征，可能不会执行实际的消除步骤。")
        selected_features_mask = np.ones(num_original_features, dtype=bool)
        X_train_selected = X_train_orig
        X_val_selected = X_val_orig
    elif k_features_to_select <= 0:
        raise ValueError(f"K_FEATURES ({k_features_to_select}) 必须为正整数。")
    else:
        selector = RFE(estimator, n_features_to_select=k_features_to_select, step=rfe_step)
        with parallel_backend('loky', n_jobs=-1):  # 启用并行化，使用所有CPU核心
            selector = selector.fit(X_train_orig, y_train)
        selected_features_mask = selector.support_
        X_train_selected = X_train_orig[:, selected_features_mask]
        X_val_selected = X_val_orig[:, selected_features_mask]
    num_selected_k_features = selected_features_mask.sum()
    print(f"从 {num_original_features} 个特征中选择了 {num_selected_k_features} 个特征")
    INPUT_SIZE = X_train_selected.shape[1]
    print(f"已将全局 INPUT_SIZE 更新为 {INPUT_SIZE}")
    all_params_for_naming_and_saving['INPUT_SIZE_FINAL_AFTER_RFE'] = INPUT_SIZE
    all_params_for_naming_and_saving['K_FEATURES_USED'] = num_selected_k_features
    save_params(all_params_for_naming_and_saving, filename_prefix, run_specific_output_dir)
    # 特征缩放
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train_selected)
    X_val_norm = scaler.transform(X_val_selected)
    # TTFS 编码
    X_train_encoded = ttfs_encode(X_train_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    X_val_encoded = ttfs_encode(X_val_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    # 创建数据集和 DataLoader
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    val_dataset = EncodedEEGDataset(X_val_encoded, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    # 定义模型
    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense_1', input_dim=INPUT_SIZE, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense_2', input_dim=HIDDEN_UNITS_1, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='dense_output', input_dim=HIDDEN_UNITS_2, outputLayer=True, kernel_initializer='glorot_uniform'))
    print("正在应用自定义权重初始化 N(0, 1/sqrt(N_in))...")
    model.apply(custom_weight_init)
    print("自定义权重初始化完成。")
    model.to(device)
    print(f"模型中总共可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL)
    scheduler = ExponentialLR(optimizer, gamma=_LR_SCHEDULER_GAMMA)
    with torch.no_grad():
        init_t_minFlt = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
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
    best_val_acc = 0.0
    best_val_loss_at_best_acc = float('inf')
    patience_counter = 0
    best_model_state_dict = None
    stopped_epoch = None
    print(f"开始训练，最多进行 {_MAX_NUM_EPOCHS} 轮...")
    for epoch in range(_MAX_NUM_EPOCHS):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates_over_epochs.append(current_lr)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                            gamma_ttfs=_TRAINING_GAMMA_TTFS,
                                            current_t_min_input=T_MIN_INPUT,
                                            current_t_max_input=T_MAX_INPUT,
                                            l1_reg=L1_REG)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        epoch_end_time = time.time()
        scheduler.step()
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"轮次 [{epoch+1}/{_MAX_NUM_EPOCHS}] | 学习率: {current_lr:.2e} | 训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f} | 时间: {epoch_end_time - epoch_start_time:.2f} 秒")
        if val_acc > best_val_acc + _EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            best_val_loss_at_best_acc = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  验证准确率提升至 {best_val_acc:.4f}。对应损失: {best_val_loss_at_best_acc:.4f}。保存模型，重置耐心计数器。")
        else:
            patience_counter += 1
            print(f"  验证准确率未显著提升。耐心计数器: {patience_counter}/{_EARLY_STOPPING_PATIENCE}")
        if patience_counter >= _EARLY_STOPPING_PATIENCE:
            print(f"在第 {epoch+1} 轮触发早停。达到最大耐心 {_EARLY_STOPPING_PATIENCE}（基于验证准确率）。")
            stopped_epoch = epoch + 1
            break
    if stopped_epoch is None:
        stopped_epoch = _MAX_NUM_EPOCHS
        print(f"训练完成，共 {_MAX_NUM_EPOCHS} 轮。")
        if best_model_state_dict is None or (val_accuracies and val_accuracies[-1] > best_val_acc):
             if val_accuracies:
                best_val_acc = val_accuracies[-1]
                best_val_loss_at_best_acc = val_losses[-1]
                best_model_state_dict = copy.deepcopy(model.state_dict())
                print(f"  训练结束，记录最后一轮的模型状态。验证准确率: {best_val_acc:.4f}, 验证损失: {best_val_loss_at_best_acc:.4f}")
    if best_model_state_dict is not None:
        print("加载在验证集上表现最佳的模型权重（基于准确率早停标准）。")
        model.load_state_dict(best_model_state_dict)
    else:
        print("警告: 未找到最佳模型状态。使用最后一轮的模型（如果存在）。")
    end_time_run = time.time()
    print(f"--- SNN训练完成（运行ID: {run_id}），总时间: {end_time_run - start_time_run:.2f} 秒，实际轮次: {stopped_epoch} ---")
    save_model_torch(model, filename_prefix, run_specific_output_dir)
    plot_history(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates_over_epochs,
                 filename_prefix, run_specific_output_dir, stopped_epoch=stopped_epoch)
    final_val_loss, final_val_acc, final_labels, final_preds = evaluate_model(model, val_loader, criterion, device)
    print(f"最终验证损失（来自最佳模型，基于准确率早停）: {final_val_loss:.4f}")
    print(f"最终验证准确率（来自最佳模型，基于准确率早停）: {final_val_acc:.4f}")
    report_names = ['负向 (0)', '中性 (1)', '正向 (2)']
    report = classification_report(final_labels, final_preds, target_names=report_names, digits=4, zero_division=0)
    print("\n分类报告（基于验证集和最佳模型，基于准确率早停）:")
    print(report)
    report_filename = os.path.join(run_specific_output_dir, f"分类报告_{filename_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_filename, 'w') as f:
        params_to_save_report = all_params_for_naming_and_saving.copy()
        params_to_save_report.update({'stopped_epoch': stopped_epoch, 'best_validation_accuracy_achieved': final_val_acc, 'best_validation_loss_at_best_accuracy': best_val_loss_at_best_acc, 'weight_initialization_note': 'N(0, 1/sqrt(N_in)) 在初始化后应用'})
        f.write(f"超参数: {json.dumps(params_to_save_report, indent=4)}\n\n")
        f.write(f"最终验证损失（最佳模型基于准确率早停）: {final_val_loss:.4f}\n")
        f.write(f"最终验证准确率（最佳模型基于准确率早停）: {final_val_acc:.4f}\n\n")
        f.write("分类报告（基于验证集和最佳模型，基于准确率早停）:\n")
        f.write(report)
    print(f"分类报告已保存至: {report_filename}")
    print(f"--- 运行 ID: {run_id} 完成 ---")
    return final_val_acc

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    initial_input_size_placeholder = INPUT_SIZE
    fixed_parameters = {
        'FEATURE_DIR': FEATURE_DIR,
        'INPUT_SIZE': initial_input_size_placeholder,
        'OUTPUT_SIZE': OUTPUT_SIZE,
        'T_MIN_INPUT': T_MIN_INPUT,
        'T_MAX_INPUT': T_MAX_INPUT,
        'TEST_SPLIT_SIZE': TEST_SPLIT_SIZE,
        'RANDOM_SEED': RANDOM_SEED,
        'TRAINING_GAMMA': TRAINING_GAMMA,
        'NUM_EPOCHS': NUM_EPOCHS,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'EARLY_STOPPING_MIN_DELTA': EARLY_STOPPING_MIN_DELTA,
        'LR_SCHEDULER_GAMMA': LR_SCHEDULER_GAMMA,
        'L1_REG': 0.001
    }
    keys, values = zip(*hyperparameter_grid.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_combinations = len(hyperparam_combinations)
    print(f"将为 {num_combinations} 个超参数组合执行训练。")
    print(f"固定参数（部分）: TRAINING_GAMMA (TTFS) = {TRAINING_GAMMA}, 最大轮次 = {NUM_EPOCHS}, LR_SCHEDULER_GAMMA = {LR_SCHEDULER_GAMMA}")
    print(f"权重初始化: N(0, 1/sqrt(N_in))")
    print(f"早停参数（基于验证准确率）: 耐心 = {EARLY_STOPPING_PATIENCE}, 最小增量 = {EARLY_STOPPING_MIN_DELTA}")
    best_accuracy_overall = 0.0
    best_hyperparams_combo_overall = None
    all_results = []
    for i, params_combo_iter in enumerate(hyperparam_combinations):
        validation_accuracy_for_run = run_training_session(current_hyperparams=params_combo_iter,
                                                           fixed_params_dict=fixed_parameters.copy(),
                                                           run_id=i+1)
        logged_params_for_summary = {**fixed_parameters, **params_combo_iter}
        logged_params_for_summary['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))'
        all_results.append({'id': i+1,
                            'params_set': logged_params_for_summary,
                            'best_validation_accuracy_achieved': validation_accuracy_for_run})
        if validation_accuracy_for_run > best_accuracy_overall:
            best_accuracy_overall = validation_accuracy_for_run
            best_hyperparams_combo_overall = logged_params_for_summary
    print("\n--- 所有超参数试验完成 ---")
    if best_hyperparams_combo_overall:
        print(f"所有运行中的最佳验证准确率（来自基于准确率早停选择的模型）: {best_accuracy_overall:.4f}")
        print(f"取得最佳准确率的参数设置（K_FEATURES 和 RFE后INPUT_SIZE 请查阅对应运行文件夹的参数文件）: {json.dumps(best_hyperparams_combo_overall, indent=2)}")
        summary_file_suffix = f"网格搜索总结_验证准确率_早停_L1正则化_自定义初始化_LinearSVC_RFE.json"
        summary_file_name = summary_file_suffix.replace(':', '_').replace('-', 'm').replace('+', '')
        summary_file = os.path.join(OUTPUT_DIR_BASE, summary_file_name)
        summary_data = {
            "最佳整体验证准确率（来自最佳准确率模型）": best_accuracy_overall,
            "取得最佳准确率的参数设置（详细RFE后参数请查阅对应运行文件夹）": best_hyperparams_combo_overall,
            "所有运行结果概述": all_results,
            "早停标准": "验证准确率",
            "RFE评估器": "LinearSVC",
            "权重初始化": "N(0, 1/sqrt(N_in))",
            "学习率策略": f"指数衰减，gamma={LR_SCHEDULER_GAMMA}",
            "正则化": "L1"
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"所有运行的总结已保存至: {summary_file}")
    else:
        print("没有成功的训练运行，或所有运行的准确率为零。")
    print("脚本执行完成。")