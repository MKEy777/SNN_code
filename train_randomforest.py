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
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Union, List, Optional, Tuple, Any
import time
import matplotlib.pyplot as plt
import json
from model.TTFS_ORIGN import SNNModel, SpikingDense
import itertools
import copy

# 设置固定参数
FEATURE_DIR = r"Individual_Features_BaselineCorrected_Variable_MP"  
OUTPUT_DIR_BASE = r"RF_结果"  
OUTPUT_SIZE = 3 
T_MIN_INPUT = 0.0 
T_MAX_INPUT = 1.0  
TEST_SPLIT_SIZE = 0.2 
RANDOM_SEED = 42  
TRAINING_GAMMA = 10.0 
NUM_EPOCHS = 200  
LR_SCHEDULER_GAMMA = 0.99  
EARLY_STOPPING_PATIENCE = 20  
EARLY_STOPPING_MIN_DELTA = 0.0001  

# 定义超参数搜索空间
hyperparameter_grid = {
    'LEARNING_RATE': [5e-4],
    'BATCH_SIZE': [8],
    'HIDDEN_UNITS_1': [256],
    'HIDDEN_UNITS_2': [128],
    'LAMBDA_L1': [0],
    'RF_N_ESTIMATORS': [150],       
    'RF_MAX_DEPTH': [10],     
    'RF_MIN_SAMPLES_SPLIT': [5],      
    'RF_MIN_SAMPLES_LEAF': [5],     
    'RF_MAX_FEATURES': [0.7],
    'K_FEATURES': [300]       
}

fixed_parameters_for_naming: Dict[str, Union[str, int, float]] = {
    'FEATURE_DIR': FEATURE_DIR,
    'OUTPUT_DIR_BASE': OUTPUT_DIR_BASE,
    'OUTPUT_SIZE': OUTPUT_SIZE,
    'T_MIN_INPUT': T_MIN_INPUT,
    'T_MAX_INPUT': T_MAX_INPUT,
    'TEST_SPLIT_SIZE': TEST_SPLIT_SIZE,
    'RANDOM_SEED': RANDOM_SEED,
    'TRAINING_GAMMA': TRAINING_GAMMA,
    'NUM_EPOCHS': NUM_EPOCHS,
    'LR_SCHEDULER_GAMMA': LR_SCHEDULER_GAMMA,
    'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
    'EARLY_STOPPING_MIN_DELTA': EARLY_STOPPING_MIN_DELTA
}

# 加载特征数据
def load_features_from_mat(feature_dir: str) -> Tuple[np.ndarray, np.ndarray]:
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

# TTFS编码
def ttfs_encode(features: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = np.clip(features, 0.0, 1.0)
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)

# 定义数据集类
class EncodedEEGDataset(Dataset):
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray):
        self.features = encoded_features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

# 自定义权重初始化
def custom_weight_init(m: nn.Module):
    if isinstance(m, SpikingDense):
        if hasattr(m, 'kernel') and m.kernel is not None:
            if hasattr(m.kernel, 'data'):
                kernel_data = m.kernel.data
            else: 
                kernel_data = m.kernel
            input_dim_for_layer = kernel_data.shape[0] 
            if hasattr(m, 'input_dim') and m.input_dim is not None:
                 input_dim_for_layer = m.input_dim
            if input_dim_for_layer > 0:
                stddev = 1.0 / np.sqrt(input_dim_for_layer)
                with torch.no_grad():
                    kernel_data.normal_(mean=0.0, std=stddev)

# 训练一个epoch
def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                gamma_ttfs: float, current_t_min_input: float, current_t_max_input: float,
                lambda_l1: float) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        outputs, min_ti_list = model(features)
        primary_loss = criterion(outputs, labels)
        l1_reg = 0.0
        for layer in model.layers_list:
            if isinstance(layer, SpikingDense):
                l1_reg += torch.sum(torch.abs(layer.kernel))
        loss = primary_loss + lambda_l1 * l1_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
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
                            if current_layer_t_max_for_gamma > min_ti_for_layer:
                                dynamic_term = gamma_ttfs * (current_layer_t_max_for_gamma - min_ti_for_layer)
                                dynamic_term = torch.clamp(dynamic_term, min=0.0)
                                new_t_max_layer = current_t_min_layer + torch.maximum(base_interval, dynamic_term)
                            else:
                                new_t_max_layer = current_t_min_layer + base_interval
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
def build_filename_prefix(params: Dict[str, Any], fixed_params_dict: Optional[Dict[str, Any]] = None) -> str:
    rf_n_estimators = params.get('RF_N_ESTIMATORS_USED', 'RFeNA')
    rf_max_depth_val = params.get('RF_MAX_DEPTH_USED', 'RFdNA')
    rf_min_samples_split = params.get('RF_MIN_SAMPLES_SPLIT_USED', 'RFmssNA')
    rf_min_samples_leaf = params.get('RF_MIN_SAMPLES_LEAF_USED', 'RFmslNA')
    rf_max_features_val = params.get('RF_MAX_FEATURES_USED', 'RFmfNA')
    rf_max_depth_str = str(rf_max_depth_val)
    if isinstance(rf_max_features_val, float):
        rf_max_features_str_formatted = str(rf_max_features_val).replace('.', 'p')
    else:
        rf_max_features_str_formatted = str(rf_max_features_val)
    rf_part = (f"RFe{rf_n_estimators}_RFd{rf_max_depth_str}_RFmss{rf_min_samples_split}"
               f"_RFmsl{rf_min_samples_leaf}_RFmf{rf_max_features_str_formatted}")
    k_features_val = params.get('K_FEATURES', 'KNA')
    k_features_part = f"_kfeat{k_features_val}"
    h1_units = params.get('HIDDEN_UNITS_1', 'h1NA')
    h2_units = params.get('HIDDEN_UNITS_2', 'h2NA')
    random_seed_val = params.get('RANDOM_SEED', 'RSNA')
    snn_part = f"_h1_{h1_units}_h2_{h2_units}_seed{random_seed_val}"
    final_prefix = rf_part + k_features_part + snn_part
    return final_prefix

# 绘制训练历史
def plot_history(train_losses, val_losses, train_accuracies, val_accuracies, train_lrs, filename_prefix: str, save_dir: str, stopped_epoch: Optional[int] = None):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))
    title_suffix = f" (Early stopped at epoch {stopped_epoch})" if stopped_epoch else ""
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'Loss Curve{title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title(f'Accuracy Curve{title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_lrs, 'go-', label='Learning Rate')
    plt.title(f'Learning Rate Curve{title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"training_history_and_lr_{filename_prefix}_{timestamp}.png")
    plt.savefig(filename)
    plt.close()
    print(f"训练历史和学习率图已保存为: {filename}")

# 保存模型
def save_model_torch(model: SNNModel, filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"模型_{filename_prefix}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型成功保存至: {save_path}")

# 保存参数
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
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_params, f, indent=4, ensure_ascii=False)
    print(f"训练参数已保存至: {save_path}")

# 运行训练会话
def run_training_session(current_hyperparams: Dict, fixed_params_dict: Dict, run_id: int) -> float:
    FEATURE_DIR = fixed_params_dict['FEATURE_DIR']
    OUTPUT_DIR_BASE = fixed_params_dict['OUTPUT_DIR_BASE']
    OUTPUT_SIZE = fixed_params_dict['OUTPUT_SIZE']
    T_MIN_INPUT = fixed_params_dict['T_MIN_INPUT']
    T_MAX_INPUT = fixed_params_dict['T_MAX_INPUT']
    TEST_SPLIT_SIZE = fixed_params_dict['TEST_SPLIT_SIZE']
    RANDOM_SEED = fixed_params_dict['RANDOM_SEED']
    all_params = {**fixed_params_dict, **current_hyperparams}
    lambda_l1 = all_params['LAMBDA_L1']
    LEARNING_RATE_INITIAL = all_params['LEARNING_RATE']
    BATCH_SIZE = all_params['BATCH_SIZE']
    _MAX_NUM_EPOCHS = all_params['NUM_EPOCHS']
    _TRAINING_GAMMA_TTFS = all_params['TRAINING_GAMMA']
    HIDDEN_UNITS_1 = all_params['HIDDEN_UNITS_1']
    HIDDEN_UNITS_2 = all_params['HIDDEN_UNITS_2']
    _LR_SCHEDULER_GAMMA = all_params['LR_SCHEDULER_GAMMA']
    _EARLY_STOPPING_PATIENCE = all_params['EARLY_STOPPING_PATIENCE']
    _EARLY_STOPPING_MIN_DELTA = all_params['EARLY_STOPPING_MIN_DELTA']
    k_features_to_select = all_params['K_FEATURES']
    rf_n_estimators = all_params.get('RF_N_ESTIMATORS', 100)
    rf_max_depth = all_params.get('RF_MAX_DEPTH', 10)
    if isinstance(rf_max_depth, str) and rf_max_depth.lower() == 'none':
        rf_max_depth = None
    rf_min_samples_split = all_params.get('RF_MIN_SAMPLES_SPLIT', 5)
    rf_min_samples_leaf = all_params.get('RF_MIN_SAMPLES_LEAF', 2)
    rf_max_features = all_params.get('RF_MAX_FEATURES', 'sqrt')
    all_params_for_naming_and_saving = all_params.copy()
    all_params_for_naming_and_saving['LR_SCHEDULER_GAMMA'] = _LR_SCHEDULER_GAMMA
    all_params_for_naming_and_saving['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))'
    all_params_for_naming_and_saving['RF_N_ESTIMATORS_USED'] = rf_n_estimators
    all_params_for_naming_and_saving['RF_MAX_DEPTH_USED'] = rf_max_depth if rf_max_depth is not None else "None"
    all_params_for_naming_and_saving['RF_MIN_SAMPLES_SPLIT_USED'] = rf_min_samples_split
    all_params_for_naming_and_saving['RF_MIN_SAMPLES_LEAF_USED'] = rf_min_samples_leaf
    all_params_for_naming_and_saving['RF_MAX_FEATURES_USED'] = rf_max_features
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_prefix_for_dir = build_filename_prefix(all_params_for_naming_and_saving, fixed_params_dict)
    run_specific_output_dir_name = f"{base_prefix_for_dir}_RF_验证准确率早停_自定义初始化_{run_timestamp}"
    run_specific_output_dir = os.path.join(OUTPUT_DIR_BASE, run_specific_output_dir_name)
    if not os.path.exists(run_specific_output_dir):
        os.makedirs(run_specific_output_dir, exist_ok=True)
    files_internal_prefix = f"最终最佳验证准确率_RF_早停_自定义初始化"
    print(f"\n--- 开始运行 ID: {run_id} ---")
    tunable_keys_to_print = list(current_hyperparams.keys())
    if 'K_FEATURES' not in tunable_keys_to_print and 'K_FEATURES' in all_params:
         tunable_keys_to_print.append('K_FEATURES')
    rf_param_keys_in_grid = [k for k in ['RF_N_ESTIMATORS', 'RF_MAX_DEPTH', 'RF_MIN_SAMPLES_SPLIT', 'RF_MIN_SAMPLES_LEAF', 'RF_MAX_FEATURES'] if k in current_hyperparams]
    tunable_keys_to_print.extend(rf_param_keys_in_grid)
    tunable_keys_to_print = sorted(list(set(tunable_keys_to_print)))
    print(f"当前运行参数 (可调部分): {json.dumps({k: all_params_for_naming_and_saving.get(k, all_params.get(k)) for k in tunable_keys_to_print}, indent=2, ensure_ascii=False)}")
    print(f"用于特征选择的随机森林参数: n_estimators={rf_n_estimators}, max_depth={rf_max_depth}, min_samples_split={rf_min_samples_split}, min_samples_leaf={rf_min_samples_leaf}, max_features='{rf_max_features}'")
    print(f"使用随机森林选择的特征数量 (K_FEATURES): {k_features_to_select}")
    print(f"初始学习率: {LEARNING_RATE_INITIAL}, 学习率衰减 Gamma: {_LR_SCHEDULER_GAMMA}")
    print(f"权重初始化方法: {all_params_for_naming_and_saving['WEIGHT_INITIALIZATION']}")
    print(f"早停设置 (基于验证准确率): 耐心轮数={_EARLY_STOPPING_PATIENCE}, 最小提升阈值={_EARLY_STOPPING_MIN_DELTA}")
    print(f"结果将保存于: {run_specific_output_dir}")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    features_data, labels_data = load_features_from_mat(FEATURE_DIR)
    print(f"数据加载成功。原始特征形状: {features_data.shape}, 标签形状: {labels_data.shape}")
    unique_labels_data, counts_data = np.unique(labels_data, return_counts=True)
    stratify_option = labels_data if all(count >= 2 for count in counts_data[counts_data > 0]) and len(unique_labels_data[counts_data > 0]) >= 2 else None
    if stratify_option is None:
        print("警告：数据集不满足所有类别的分层抽样条件。可能使用随机抽样或部分分层。")
    X_train_full, X_val_full, y_train, y_val = train_test_split(
        features_data, labels_data, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=stratify_option
    )
    print(f"训练集大小: {len(X_train_full)}")
    print(f"验证集大小: {len(X_val_full)}")
    print(f"应用随机森林特征选择 (n_jobs=-1)，选择 k={k_features_to_select} 个特征...")
    if k_features_to_select > X_train_full.shape[1]:
        print(f"警告: 请求的 K_FEATURES ({k_features_to_select}) 大于训练集特征数 ({X_train_full.shape[1]})。将使用所有训练集特征。")
        k_features_to_select = X_train_full.shape[1]
    if k_features_to_select <= 0:
        raise ValueError(f"K_FEATURES ({k_features_to_select}) 必须为正数。")
    rf_fit_start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        random_state=RANDOM_SEED,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        max_features=rf_max_features,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train_full, y_train)
    rf_fit_end_time = time.time()
    print(f"随机森林拟合完成，耗时: {rf_fit_end_time - rf_fit_start_time:.2f} 秒")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:k_features_to_select]
    X_train_selected = X_train_full[:, indices]
    X_val_selected = X_val_full[:, indices]
    print(f"特征选择完成。训练集选择后特征形状: {X_train_selected.shape}, 验证集选择后特征形状: {X_val_selected.shape}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train_selected)
    X_val_norm = scaler.transform(X_val_selected)
    X_train_encoded = ttfs_encode(X_train_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    X_val_encoded = ttfs_encode(X_val_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    val_dataset = EncodedEEGDataset(X_val_encoded, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    INPUT_SIZE_SELECTED = X_train_encoded.shape[1]
    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense_1', input_dim=INPUT_SIZE_SELECTED, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense_2', input_dim=HIDDEN_UNITS_1, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='dense_output', input_dim=HIDDEN_UNITS_2, outputLayer=True, kernel_initializer='glorot_uniform'))
    print("应用自定义权重初始化 N(0, 1/sqrt(N_in))...")
    model.apply(custom_weight_init)
    print("自定义权重初始化完成。")
    model.to(device)
    print(f"模型中可训练参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL)
    scheduler = ExponentialLR(optimizer, gamma=_LR_SCHEDULER_GAMMA)
    with torch.no_grad():
        _t_min_for_afferent_spikes = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
        _t_min_for_efferent_spikes = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
        for layer in model.layers_list:
            if isinstance(layer, SpikingDense):
                _t_max_for_efferent_spikes = _t_min_for_efferent_spikes + torch.tensor(1.0, dtype=torch.float32, device=device)
                layer.set_time_params(_t_min_for_afferent_spikes, _t_min_for_efferent_spikes,  _t_max_for_efferent_spikes)
                _t_min_for_afferent_spikes = _t_min_for_efferent_spikes.clone()
                _t_min_for_efferent_spikes = _t_max_for_efferent_spikes.clone()
    start_time_run = time.time()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    learning_rates_over_epochs = []
    best_val_acc = 0.0
    best_val_loss_at_best_acc = float('inf')
    patience_counter = 0
    best_model_state_dict = None
    stopped_epoch = None
    print(f"开始训练，最多 {_MAX_NUM_EPOCHS} 轮...")
    for epoch in range(_MAX_NUM_EPOCHS):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates_over_epochs.append(current_lr)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                            gamma_ttfs=_TRAINING_GAMMA_TTFS,
                                            current_t_min_input=T_MIN_INPUT,
                                            current_t_max_input=T_MAX_INPUT,
                                            lambda_l1=lambda_l1)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        epoch_end_time = time.time()
        scheduler.step()
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"轮数 [{epoch+1}/{_MAX_NUM_EPOCHS}] | 学习率: {current_lr:.2e} | "
              f"训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f} | "
              f"耗时: {epoch_end_time - epoch_start_time:.2f} 秒")
        if val_acc > best_val_acc + _EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            best_val_loss_at_best_acc = val_loss
            patience_counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  验证准确率提升至 {best_val_acc:.4f}。对应损失: {best_val_loss_at_best_acc:.4f}。重置耐心计数器。")
        else:
            patience_counter += 1
            print(f"  验证准确率未显著提升。耐心计数器: {patience_counter}/{_EARLY_STOPPING_PATIENCE}")
        if patience_counter >= _EARLY_STOPPING_PATIENCE:
            print(f"在第 {epoch+1} 轮触发早停。已达到最大耐心轮数 {_EARLY_STOPPING_PATIENCE} (基于验证准确率)。")
            stopped_epoch = epoch + 1
            break
    if stopped_epoch is None:
        stopped_epoch = _MAX_NUM_EPOCHS
        print(f"已完成 {_MAX_NUM_EPOCHS} 轮训练。")
        if best_model_state_dict is None:
            if val_accuracies:
                print(f"  由于准确率在训练全程未显著提升（或仅完成少量轮次），使用最后一轮的模型状态。")
                best_val_acc = val_accuracies[-1]
                best_val_loss_at_best_acc = val_losses[-1] if val_losses else float('inf')
                best_model_state_dict = copy.deepcopy(model.state_dict())
                print(f"  记录最后一轮模型。验证准确率: {best_val_acc:.4f}, 验证损失: {best_val_loss_at_best_acc:.4f}")
            else:
                 print("  警告：训练结束时没有可用的验证准确率/损失记录，且未通过早停标准保存模型。将使用模型当前状态。")
    if best_model_state_dict is not None:
        print("加载在验证集上表现最佳的模型权重 (基于最高验证准确率)。")
        model.load_state_dict(best_model_state_dict)
    else:
        print("警告：未找到可加载的最佳模型状态字典 (基于验证准确率)。将使用模型当前 (可能是最后一轮) 的状态。")
        if not val_accuracies:
            best_val_acc = float('nan')
            best_val_loss_at_best_acc = float('nan')
            print("  由于没有验证记录，最佳验证准确率和损失设为 NaN。")
        elif best_val_acc == 0.0 and best_val_loss_at_best_acc == float('inf'):
            best_val_acc = val_accuracies[-1] if val_accuracies else float('nan')
            best_val_loss_at_best_acc = val_losses[-1] if val_losses else float('nan')
            print(f"  使用最后一轮的模型状态。验证准确率: {best_val_acc:.4f}, 验证损失: {best_val_loss_at_best_acc:.4f}")
    end_time_run = time.time()
    print(f"--- SNN 训练完成 (运行 ID: {run_id}), 总耗时: {(end_time_run - start_time_run) + (rf_fit_end_time - rf_fit_start_time):.2f} 秒 (含RF拟合), SNN训练耗时: {end_time_run - start_time_run:.2f} 秒, 实际轮数: {stopped_epoch} ---")
    save_model_torch(model, files_internal_prefix, run_specific_output_dir)
    plot_history(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates_over_epochs,
                 files_internal_prefix, run_specific_output_dir, stopped_epoch=stopped_epoch)
    final_val_loss, final_val_acc, final_labels, final_preds = evaluate_model(model, val_loader, criterion, device)
    print(f"最终验证损失 (来自验证准确率最佳的模型): {final_val_loss:.4f}")
    print(f"最终验证准确率 (来自验证准确率最佳的模型): {final_val_acc:.4f}")
    report_names = ['负向 (0)', '中性 (1)', '正向 (2)']
    report = classification_report(final_labels, final_preds, target_names=report_names, digits=4, zero_division=0)
    print("\n分类报告 (基于验证集，使用验证准确率最佳的模型):")
    print(report)
    report_filename = os.path.join(run_specific_output_dir, f"分类报告_{files_internal_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_filename, 'w', encoding='utf-8') as f:
        params_to_save_report = {
            **all_params_for_naming_and_saving,
            '停止轮数': stopped_epoch,
            '达成的最佳验证准确率': best_val_acc,
            '最佳准确率时的验证损失': best_val_loss_at_best_acc,
            '最终评估的最佳模型验证准确率': final_val_acc,
            '最终评估的最佳模型验证损失': final_val_loss,
            '权重初始化说明': all_params_for_naming_and_saving.get('WEIGHT_INITIALIZATION', 'N(0, 1/sqrt(N_in))'),
            '选择的特征数量(K_FEATURES_USED)': k_features_to_select,
            '特征选择方法': f"随机森林 (RandomForestClassifier, n_jobs=-1, class_weight='balanced')",
            '随机森林拟合耗时(秒)': round(rf_fit_end_time - rf_fit_start_time, 2)
        }
        keys_to_remove_from_report_params = ['FEATURE_DIR', 'OUTPUT_DIR_BASE', 'RANDOM_SEED', 'TEST_SPLIT_SIZE', 'OUTPUT_SIZE']
        for k_rem in keys_to_remove_from_report_params:
            params_to_save_report.pop(k_rem, None)
        f.write(f"超参数及运行信息: {json.dumps(params_to_save_report, indent=4, ensure_ascii=False)}\n\n")
        f.write(f"最终验证损失 (来自验证准确率最佳的模型，重新评估得到): {final_val_loss:.4f}\n")
        f.write(f"最终验证准确率 (来自验证准确率最佳的模型，重新评估得到): {final_val_acc:.4f}\n\n")
        f.write("分类报告 (基于验证集，使用验证准确率最佳的模型):\n")
        f.write(report)
    print(f"分类报告已保存至: {report_filename}")
    print(f"--- 运行 ID: {run_id} 完成 ---")
    return best_val_acc

# 主程序入口
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    fixed_parameters_for_naming.update({
        'FEATURE_DIR': FEATURE_DIR,
        'OUTPUT_SIZE': OUTPUT_SIZE,
        'T_MIN_INPUT': T_MIN_INPUT,
        'T_MAX_INPUT': T_MAX_INPUT,
        'TEST_SPLIT_SIZE': TEST_SPLIT_SIZE,
        'RANDOM_SEED': RANDOM_SEED,
        'TRAINING_GAMMA': TRAINING_GAMMA,
        'NUM_EPOCHS': NUM_EPOCHS,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'EARLY_STOPPING_MIN_DELTA': EARLY_STOPPING_MIN_DELTA,
        'LR_SCHEDULER_GAMMA': LR_SCHEDULER_GAMMA
    })
    keys, values = zip(*hyperparameter_grid.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_combinations = len(hyperparam_combinations)
    print(f"将为 {num_combinations} 种超参数组合执行训练。")
    print(f"固定参数 (部分): TTFS训练伽马值 = {TRAINING_GAMMA}, 最大轮数 = {NUM_EPOCHS}, 学习率衰减伽马值 = {LR_SCHEDULER_GAMMA}")
    print(f"将使用的权重初始化方法: N(0, 1/sqrt(N_in))")
    print(f"特征选择方法: 随机森林 (RandomForestClassifier)")
    print(f"早停参数 (基于验证准确率): 耐心轮数 = {EARLY_STOPPING_PATIENCE}, 最小提升阈值 = {EARLY_STOPPING_MIN_DELTA}")
    best_accuracy_overall = 0.0
    best_hyperparams_combo_overall = None
    all_results_summary = []
    for i, params_combo_iter in enumerate(hyperparam_combinations):
        current_fixed_params = fixed_parameters_for_naming.copy()
        validation_accuracy_for_run = run_training_session(
            current_hyperparams=params_combo_iter,
            fixed_params_dict=current_fixed_params,
            run_id=i+1
        )
        full_params_for_log = {**current_fixed_params, **params_combo_iter}
        full_params_for_log['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))'
        all_results_summary.append({
            '运行ID': i+1,
            '参数': full_params_for_log,
            '此运行的最佳验证准确率': validation_accuracy_for_run
        })
        if validation_accuracy_for_run > best_accuracy_overall:
            best_accuracy_overall = validation_accuracy_for_run
            best_hyperparams_combo_overall = full_params_for_log
    print("\n--- 所有超参数试验完成 ---")
    if best_hyperparams_combo_overall:
        print(f"所有运行中的最佳验证准确率 (来自基于验证准确率早停选择的模型): {best_accuracy_overall:.4f}")
        serializable_best_hyperparams = {}
        for k, v in best_hyperparams_combo_overall.items():
            if isinstance(v, np.integer): serializable_best_hyperparams[k] = int(v)
            elif isinstance(v, np.floating): serializable_best_hyperparams[k] = float(v)
            elif isinstance(v, np.ndarray): serializable_best_hyperparams[k] = v.tolist()
            else: serializable_best_hyperparams[k] = v
        print(f"对应的最佳超参数组合: {json.dumps(serializable_best_hyperparams, indent=2, ensure_ascii=False)}")
        summary_file_name = f"网格搜索摘要_RF_验证准确率早停_自定义初始化_{time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_file_path = os.path.join(OUTPUT_DIR_BASE, summary_file_name)
        serializable_all_results = []
        for res in all_results_summary:
            s_params = {}
            for k_p, v_p in res['参数'].items():
                if isinstance(v_p, np.integer): s_params[k_p] = int(v_p)
                elif isinstance(v_p, np.floating): s_params[k_p] = float(v_p)
                elif isinstance(v_p, np.ndarray): s_params[k_p] = v_p.tolist()
                else: s_params[k_p] = v_p
            serializable_all_results.append({'运行ID': res['运行ID'], '参数': s_params, '此运行的最佳验证准确率': res['此运行的最佳验证准确率']})
        summary_data = {
            "最高总体验证准确率": best_accuracy_overall,
            "最佳总体超参数": serializable_best_hyperparams,
            "所有运行结果详情": serializable_all_results,
            "早停标准": "验证准确率",
            "早停耐心轮数": EARLY_STOPPING_PATIENCE,
            "早停最小提升阈值": EARLY_STOPPING_MIN_DELTA,
            "权重初始化方法": "N(0, 1/sqrt(N_in))",
            "特征选择方法": "随机森林 (RandomForestClassifier)",
            "学习率策略": f"Adam优化器, 然后指数衰减 (gamma={LR_SCHEDULER_GAMMA})",
            "TTFS训练伽马值": TRAINING_GAMMA
        }
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"所有运行的摘要已保存至: {summary_file_path}")
    else:
        print("没有成功的训练运行，或者所有运行的准确率为零或NaN。")
    print("脚本执行完成。")