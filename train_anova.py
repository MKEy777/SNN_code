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
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Dict, Union, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
import json

from model.TTFS_ORIGN import SNNModel, SpikingDense
import itertools
import copy

# --- 固定参数设置 ---
FEATURE_DIR = r"Individual_Features_BaselineCorrected_Variable_MP" 
OUTPUT_DIR_BASE = r"anova_result_中文_准确率早停" 
OUTPUT_SIZE = 3
T_MIN_INPUT = 0.0 
T_MAX_INPUT = 1.0 
TEST_SPLIT_SIZE = 0.2 
RANDOM_SEED = 42 
TRAINING_GAMMA = 10.0 
NUM_EPOCHS = 150 
LR_SCHEDULER_GAMMA = 0.99 
# --- 早停参数设置 ---
EARLY_STOPPING_PATIENCE = 15 
EARLY_STOPPING_MIN_DELTA = 0.0005 
# --- 特征选择参数 ---
K_FEATURES_DEFAULT = 300 

# --- 超参数搜索空间定义 ---
hyperparameter_grid = {
    'LEARNING_RATE': [5e-4],
    'BATCH_SIZE': [8],
    'HIDDEN_UNITS_1': [392],
    'HIDDEN_UNITS_2': [128],
    'K_FEATURES': [600], 
}

# 加载特征数据并应用 ANOVA F-test 特征选择
def load_features_from_mat(feature_dir, k_features=100):
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
        raise ValueError(f"在目录 {feature_dir} 中未找到特征文件。")
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    label_mapping = {-1: 0, 0: 1, 1: 2} # 标签映射：-1(负向)->0, 0(中性)->1, 1(正向)->2
    valid_labels_indices = np.isin(combined_labels, list(label_mapping.keys()))
    combined_features_filtered = combined_features[valid_labels_indices]
    combined_labels_filtered = combined_labels[valid_labels_indices]
    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels_filtered], dtype=np.int64)

    # 应用 ANOVA F-test 特征选择
    num_available_features = combined_features_filtered.shape[1]
    if k_features > num_available_features:
        print(f"警告：请求的 k_features ({k_features}) 大于可用特征数 ({num_available_features})。将使用 {num_available_features} 个特征。")
        k_features = num_available_features
            
    if k_features <= 0:
        raise ValueError(f"k_features 必须为正数。得到 {k_features}")

    selector = SelectKBest(f_classif, k=k_features)
    selected_features = selector.fit_transform(combined_features_filtered, mapped_labels)
    return selected_features, mapped_labels

# TTFS时间编码
def ttfs_encode(features: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    features = np.clip(features, 0.0, 1.0) # 将特征值裁剪到 [0, 1] 区间
    spike_times = t_max - features * (t_max - t_min) # TTFS编码：时间与强度成反比
    return torch.tensor(spike_times, dtype=torch.float32)

# 编码后的EEG数据集类
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
    自定义权重初始化: 权重服从N(0, 1/N_in)分布 (标准差为 1/sqrt(N_in))
    N_in 表示层的输入特征数量 (或输入神经元数量)
    """
    if isinstance(m, SpikingDense):
        if hasattr(m, 'kernel') and m.kernel is not None:
            if hasattr(m.kernel, 'data'):
                kernel_data = m.kernel.data
            else:
                kernel_data = m.kernel 

            input_dim_for_layer = kernel_data.shape[0] # 假设权重矩阵形状为 [N_in, N_out]
            if hasattr(m, 'input_dim') and m.input_dim is not None: # 使用层定义的input_dim（如果存在且更准确）
                 input_dim_for_layer = m.input_dim

            if input_dim_for_layer > 0:
                stddev = 1.0 / np.sqrt(input_dim_for_layer)
                with torch.no_grad(): # 不追踪梯度历史
                    kernel_data.normal_(mean=0.0, std=stddev)

# 训练单个epoch
def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                gamma_ttfs: float, current_t_min_input: float, current_t_max_input: float) -> Tuple[float, float]:
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        outputs, min_ti_list = model(features) # 模型前向传播

        primary_loss = criterion(outputs, labels) # 计算主要损失 (交叉熵)
        loss = primary_loss

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        # TTFS 时间参数更新逻辑 (这部分逻辑基于原代码，确保其符合TTFS原理)
        with torch.no_grad():
            current_t_min_layer = torch.tensor(current_t_max_input, dtype=torch.float32, device=device) 
            t_min_prev_layer = torch.tensor(current_t_min_input, dtype=torch.float32, device=device)
            
            k = 0 
            for layer_idx, layer in enumerate(model.layers_list):
                if isinstance(layer, SpikingDense):
                    new_t_max_layer = torch.tensor(0.0, dtype=torch.float32, device=device)
                    if not layer.outputLayer: # 非输出层
                        min_ti_for_layer = None
                        if k < len(min_ti_list) and min_ti_list[k] is not None:
                            positive_spike_times = min_ti_list[k][min_ti_list[k] > 1e-6] 
                            if positive_spike_times.numel() > 0:
                                min_ti_for_layer = torch.min(positive_spike_times)
                        
                        base_interval = torch.tensor(1.0, dtype=torch.float32, device=device) 
                        
                        if min_ti_for_layer is not None:
                            current_layer_t_max_for_gamma = layer.t_max.clone().detach()
                            if current_layer_t_max_for_gamma > min_ti_for_layer: # 确保 t_max > min_ti
                                dynamic_term = gamma_ttfs * (current_layer_t_max_for_gamma - min_ti_for_layer)
                                dynamic_term = torch.clamp(dynamic_term, min=0.0) # 确保动态项非负
                                new_t_max_layer = current_t_min_layer + torch.maximum(base_interval, dynamic_term)
                            else: 
                                new_t_max_layer = current_t_min_layer + base_interval # 如果 min_ti 过大，使用基础间隔
                        else: # 如果当前批次没有观察到有效脉冲
                            new_t_max_layer = current_t_min_layer + base_interval
                        k += 1
                    else: # 输出层
                        new_t_max_layer = current_t_min_layer + torch.tensor(1.0, dtype=torch.float32, device=device) # 输出层固定时间窗口 T
                    
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

# 模型评估
def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float, List, List]:
    model.eval() # 设置模型为评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []
    with torch.no_grad(): # 评估时不需要计算梯度
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs, _ = model(features) # 评估时不需要 min_ti_list
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

# 构建文件名前缀 (用于目录名)
def build_filename_prefix(params: Dict[str, Union[int, float]], fixed_params_ref: Dict) -> str:
    lr_val = params.get('LEARNING_RATE', fixed_params_ref.get('LEARNING_RATE'))
    lr_str = f"{lr_val:.0e}".replace('-', 'm').replace('+', '') # 学习率字符串表示
    
    gamma_ttfs_val = params.get('TRAINING_GAMMA', fixed_params_ref.get('TRAINING_GAMMA'))
    gamma_str = str(gamma_ttfs_val).replace('.', 'p') # TTFS Gamma 字符串表示
    
    lr_decay_gamma_val = params.get('LR_SCHEDULER_GAMMA', fixed_params_ref.get('LR_SCHEDULER_GAMMA', 'NA'))
    lr_decay_gamma_str = f"_lrdecay{str(lr_decay_gamma_val).replace('.', 'p')}" if lr_decay_gamma_val != 'NA' else ""
    
    k_features_val = params.get('K_FEATURES', K_FEATURES_DEFAULT) # K_FEATURES会从params中获取
    k_features_str = f"_kfeat{k_features_val}" # 特征数量字符串表示

    prefix = (f"lr{lr_str}_bs{params['BATCH_SIZE']}_epochsMax{params['NUM_EPOCHS']}"
              f"_h1_{params['HIDDEN_UNITS_1']}_h2_{params['HIDDEN_UNITS_2']}"
              f"_gammaTTFS{gamma_str}{lr_decay_gamma_str}_seed{params['RANDOM_SEED']}{k_features_str}")
    return prefix

# 绘制训练过程曲线
def plot_history(train_losses, val_losses, train_accuracies, val_accuracies, train_lrs, filename_prefix: str, save_dir: str, stopped_epoch: Optional[int] = None):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))
    title_suffix = f" (在第 {stopped_epoch} 轮早停)" if stopped_epoch else ""
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='训练损失')
    plt.plot(epochs_range, val_losses, 'ro-', label='验证损失')
    plt.title(f'损失曲线{title_suffix}')
    plt.xlabel('训练轮数'); plt.ylabel('损失'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='训练准确率')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='验证准确率')
    plt.title(f'准确率曲线{title_suffix}')
    plt.xlabel('训练轮数'); plt.ylabel('准确率'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_lrs, 'go-', label='学习率')
    plt.title(f'学习率曲线{title_suffix}')
    plt.xlabel('训练轮数'); plt.ylabel('学习率'); plt.legend(); plt.grid(True)
    plt.yscale('log') # y轴使用对数刻度
    
    plt.tight_layout() # 调整子图布局
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # filename_prefix 此处是如 "最终最佳验证准确率_早停_自定义初始化"
    filename = os.path.join(save_dir, f"训练历史与学习率_{filename_prefix}_{timestamp}.png")
    plt.savefig(filename); plt.close()
    print(f"训练历史和学习率曲线图已保存为: {filename}")

# 保存模型参数
def save_model_torch(model: SNNModel, filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"模型_{filename_prefix}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型成功保存至: {save_path}")

# 保存训练参数
def save_params(params: Dict[str, Union[int, float, str]], filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"参数_{filename_prefix}_{timestamp}.json")
    serializable_params = {} # 转换为JSON兼容的类型
    for k, v in params.items():
        if isinstance(v, np.integer): serializable_params[k] = int(v)
        elif isinstance(v, np.floating): serializable_params[k] = float(v)
        elif isinstance(v, np.ndarray): serializable_params[k] = v.tolist()
        else: serializable_params[k] = v
    
    if 'LR_SCHEDULER_GAMMA' not in serializable_params and 'LR_SCHEDULER_GAMMA' in params:
         serializable_params['LR_SCHEDULER_GAMMA'] = params['LR_SCHEDULER_GAMMA']

    with open(save_path, 'w', encoding='utf-8') as f: # 使用utf-8编码保存中文
        json.dump(serializable_params, f, indent=4, ensure_ascii=False)
    print(f"训练参数已保存至: {save_path}")

# --- 封装的训练和评估函数 ---
def run_training_session(current_hyperparams: Dict, fixed_params_dict: Dict, run_id: int, fixed_params_ref_for_naming: Dict):
    all_params = {**fixed_params_dict, **current_hyperparams} # 合并固定参数和当前超参数

    LEARNING_RATE_INITIAL = all_params['LEARNING_RATE']
    BATCH_SIZE = all_params['BATCH_SIZE']
    _MAX_NUM_EPOCHS = all_params['NUM_EPOCHS']
    _TRAINING_GAMMA_TTFS = all_params['TRAINING_GAMMA'] 
    HIDDEN_UNITS_1 = all_params['HIDDEN_UNITS_1']
    HIDDEN_UNITS_2 = all_params['HIDDEN_UNITS_2']
    _LR_SCHEDULER_GAMMA = all_params['LR_SCHEDULER_GAMMA']
    _EARLY_STOPPING_PATIENCE = all_params['EARLY_STOPPING_PATIENCE']
    _EARLY_STOPPING_MIN_DELTA = all_params['EARLY_STOPPING_MIN_DELTA']
    
    k_features = all_params['K_FEATURES'] # K_FEATURES 从超参数网格中获取

    all_params_for_naming_and_saving = all_params.copy()
    all_params_for_naming_and_saving['LR_SCHEDULER_GAMMA'] = _LR_SCHEDULER_GAMMA 
    all_params_for_naming_and_saving['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))' # 权重初始化方法描述
    all_params_for_naming_and_saving['K_FEATURES'] = k_features

    base_prefix = build_filename_prefix(all_params_for_naming_and_saving, fixed_params_ref_for_naming)
    # 目录名包含具体超参数和方法描述 (中文)
    run_specific_output_dir_name = f"{base_prefix}_验证准确率早停_自定义初始化"
    run_specific_output_dir = os.path.join(OUTPUT_DIR_BASE, run_specific_output_dir_name)

    if not os.path.exists(run_specific_output_dir):
        os.makedirs(run_specific_output_dir, exist_ok=True)

    # 内部文件的前缀 (中文)
    files_internal_prefix = f"最终最佳验证准确率_早停_自定义初始化"

    print(f"\n--- 开始运行 ID: {run_id} ---")
    print(f"当前运行参数: {json.dumps(all_params_for_naming_and_saving, indent=2, ensure_ascii=False)}")
    print(f"当前运行选择的 K_FEATURES: {k_features}")
    print(f"初始学习率: {LEARNING_RATE_INITIAL}, 学习率衰减 Gamma: {_LR_SCHEDULER_GAMMA}")
    print(f"权重初始化方法: N(0, 1/sqrt(N_in))")
    print(f"早停设置 (基于验证准确率): 耐心轮数={_EARLY_STOPPING_PATIENCE}, 最小提升阈值={_EARLY_STOPPING_MIN_DELTA}")
    print(f"结果将保存于: {run_specific_output_dir}")

    save_params(all_params_for_naming_and_saving, files_internal_prefix, run_specific_output_dir)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    features_data, labels_data = load_features_from_mat(FEATURE_DIR, k_features=k_features)
    print(f"数据加载成功。特征选择后形状: {features_data.shape}, 标签形状: {labels_data.shape}")

    unique_labels_data, counts_data = np.unique(labels_data, return_counts=True)
    stratify_option = labels_data if all(count >= 2 for count in counts_data[counts_data > 0]) and len(unique_labels_data[counts_data > 0]) >= 2 else None
    if stratify_option is None: print("警告：数据集不满足所有类别的分层抽样条件。可能使用随机抽样或部分分层。")
    
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(
        features_data, labels_data, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=stratify_option
    )
    print(f"训练集大小: {len(X_train_orig)}")
    print(f"验证集大小: {len(X_val_orig)}")

    scaler = MinMaxScaler(feature_range=(0, 1)) # 特征归一化到 [0,1]
    X_train_norm = scaler.fit_transform(X_train_orig)
    X_val_norm = scaler.transform(X_val_orig)
    X_train_encoded = ttfs_encode(X_train_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT) # TTFS编码
    X_val_encoded = ttfs_encode(X_val_norm, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT)

    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    val_dataset = EncodedEEGDataset(X_val_encoded, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    INPUT_SIZE = X_train_encoded.shape[1] # 输入层大小根据选择的特征动态确定

    model = SNNModel()
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense_1', input_dim=INPUT_SIZE, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense_2', input_dim=HIDDEN_UNITS_1, outputLayer=False, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='dense_output', input_dim=HIDDEN_UNITS_2, outputLayer=True, kernel_initializer='glorot_uniform'))

    print("应用自定义权重初始化 N(0, 1/sqrt(N_in))...")
    model.apply(custom_weight_init) 
    print("自定义权重初始化完成。")

    model.to(device)
    print(f"模型中可训练参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL) # 优化器
    scheduler = ExponentialLR(optimizer, gamma=_LR_SCHEDULER_GAMMA) # 学习率调度器

    # 初始化SNN层的时间参数
    with torch.no_grad():
        _t_min_for_afferent_spikes = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
        _t_min_for_efferent_spikes = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device) 

        for layer in model.layers_list:
            if isinstance(layer, SpikingDense):
                _t_max_for_efferent_spikes = _t_min_for_efferent_spikes + torch.tensor(1.0, dtype=torch.float32, device=device) 
                layer.set_time_params(_t_min_for_afferent_spikes, _t_min_for_efferent_spikes,  _t_max_for_efferent_spikes)
                _t_min_for_afferent_spikes = _t_min_for_efferent_spikes.clone() 
                _t_min_for_efferent_spikes = _t_max_for_efferent_spikes.clone() 

    start_time_run = time.time() # 记录运行开始时间
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    learning_rates_over_epochs = []

    best_val_acc = 0.0 # 早停基于验证准确率
    best_val_loss_at_best_acc = float('inf') # 记录最佳准确率时的损失
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
                                            current_t_max_input=T_MAX_INPUT)

        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        epoch_end_time = time.time()

        scheduler.step() # 更新学习率

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"轮数 [{epoch+1}/{_MAX_NUM_EPOCHS}] | 学习率: {current_lr:.2e} | "
              f"训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f} | "
              f"耗时: {epoch_end_time - epoch_start_time:.2f} 秒")

        # 基于验证准确率的早停逻辑
        if val_acc > best_val_acc + _EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc
            best_val_loss_at_best_acc = val_loss # 记录此时的损失
            patience_counter = 0 # 重置耐心计数器
            best_model_state_dict = copy.deepcopy(model.state_dict()) # 保存最佳模型状态
            print(f"  验证准确率提升至 {best_val_acc:.4f}。对应损失: {best_val_loss_at_best_acc:.4f}。重置耐心计数器。")
        else:
            patience_counter += 1
            print(f"  验证准确率未显著提升。耐心计数器: {patience_counter}/{_EARLY_STOPPING_PATIENCE}")

        if patience_counter >= _EARLY_STOPPING_PATIENCE:
            print(f"在第 {epoch+1} 轮触发早停。已达到最大耐心轮数 {_EARLY_STOPPING_PATIENCE} (基于验证准确率)。")
            stopped_epoch = epoch + 1
            break # 结束训练循环
    
    if stopped_epoch is None: 
        stopped_epoch = _MAX_NUM_EPOCHS
        print(f"已完成 {_MAX_NUM_EPOCHS} 轮训练。")
        if best_model_state_dict is None and val_accuracies: # 如果从未保存过模型 (例如只有一轮或准确率一直未提升)
            best_val_acc = val_accuracies[-1]
            best_val_loss_at_best_acc = val_losses[-1] if val_losses else float('inf')
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  训练结束时未通过早停标准记录最佳模型，使用最后一轮的模型。验证准确率: {best_val_acc:.4f}, 损失: {best_val_loss_at_best_acc:.4f}")

    if best_model_state_dict is not None:
        print("加载在验证集上表现最佳的模型权重 (基于最高验证准确率)。")
        model.load_state_dict(best_model_state_dict)
    else:
        print("警告：未找到最佳模型状态字典。将使用模型当前 (最后一轮) 的状态。")
        if val_accuracies and val_losses: # 确保有记录
            best_val_acc = val_accuracies[-1] if val_accuracies else float('nan')
            best_val_loss_at_best_acc = val_losses[-1] if val_losses else float('nan')
            print(f"  使用最后一轮的模型状态。最终验证准确率: {best_val_acc:.4f}, 最终验证损失: {best_val_loss_at_best_acc:.4f}")


    end_time_run = time.time()
    print(f"--- SNN 训练完成 (运行 ID: {run_id}), 总耗时: {end_time_run - start_time_run:.2f} 秒, 实际轮数: {stopped_epoch} ---")

    save_model_torch(model, files_internal_prefix, run_specific_output_dir)
    plot_history(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates_over_epochs,
                 files_internal_prefix, run_specific_output_dir, stopped_epoch=stopped_epoch)

    # 使用加载的最佳模型进行最终评估
    final_val_loss, final_val_acc, final_labels, final_preds = evaluate_model(model, val_loader, criterion, device)
    print(f"最终验证损失 (来自验证准确率最佳的模型): {final_val_loss:.4f}") 
    print(f"最终验证准确率 (来自验证准确率最佳的模型): {final_val_acc:.4f}")

    report_names = ['负向 (0)', '中性 (1)', '正向 (2)'] # 分类报告的类别名称
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
            '权重初始化说明': '在SNN模型层定义后应用 N(0, 1/sqrt(N_in))'
        }
        f.write(f"超参数及运行信息: {json.dumps(params_to_save_report, indent=4, ensure_ascii=False)}\n\n")
        f.write(f"最终验证损失 (来自验证准确率最佳的模型): {final_val_loss:.4f}\n")
        f.write(f"最终验证准确率 (来自验证准确率最佳的模型): {final_val_acc:.4f}\n\n")
        f.write("分类报告 (基于验证集，使用验证准确率最佳的模型):\n")
        f.write(report)
    print(f"分类报告已保存至: {report_filename}")
    print(f"--- 运行 ID: {run_id} 完成 ---")
    return best_val_acc # 返回此运行中获得的最佳验证准确率

# 主程序
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

    fixed_parameters_runtime = {
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
    }
    
    fixed_params_for_naming_ref = fixed_parameters_runtime.copy()
    fixed_params_for_naming_ref['K_FEATURES'] = K_FEATURES_DEFAULT 

    keys, values = zip(*hyperparameter_grid.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    num_combinations = len(hyperparam_combinations)
    print(f"将为 {num_combinations} 种超参数组合执行训练。")
    print(f"固定参数 (部分): TTFS训练伽马值 = {TRAINING_GAMMA}, 最大轮数 = {NUM_EPOCHS}, 学习率衰减伽马值 = {LR_SCHEDULER_GAMMA}")
    print(f"将使用的权重初始化方法: N(0, 1/sqrt(N_in))")
    print(f"早停参数 (基于验证准确率): 耐心轮数 = {EARLY_STOPPING_PATIENCE}, 最小提升阈值 = {EARLY_STOPPING_MIN_DELTA}")

    best_accuracy_overall = 0.0 # 所有运行中的最佳准确率
    best_hyperparams_combo_overall = None # 对应的最佳超参数组合
    all_results_summary = [] # 存储所有运行结果的列表

    for i, params_combo_iter in enumerate(hyperparam_combinations):
        validation_accuracy_for_run = run_training_session(
            current_hyperparams=params_combo_iter,
            fixed_params_dict=fixed_parameters_runtime,
            run_id=i+1,
            fixed_params_ref_for_naming=fixed_params_for_naming_ref
        )

        full_params_for_log = {**fixed_parameters_runtime, **params_combo_iter}
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

        summary_file_name = f"网格搜索摘要_验证准确率早停_自定义初始化_{time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_file_path = os.path.join(OUTPUT_DIR_BASE, summary_file_name)

        serializable_all_results = []
        for res in all_results_summary:
            s_params = {}
            for k_p, v_p in res['参数'].items(): # '参数' key was used above
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
            "学习率策略": f"Adam, 然后指数衰减 (gamma={LR_SCHEDULER_GAMMA})",
            "TTFS训练伽马值": TRAINING_GAMMA
        }
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"所有运行的摘要已保存至: {summary_file_path}")
    else:
        print("没有成功的训练运行，或者所有运行的准确率为零或NaN。")

    print("脚本执行完成。")