# 文件名: CSNN_depthwise_random_split.py

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Dict, Union, List, Optional, Tuple, Any
import time
import matplotlib.pyplot as plt
import json
import itertools
import copy

# 从模型文件中导入我们需要的模块
from model.TTFS import SNNModel, SpikingDense, DivisionFreeAnnToSnnEncoder

# --- 【修改1】添加深度可分离卷积模块的定义 ---
class DepthwiseSeparableConv(nn.Module):
    """
    支持步长的深度可分离卷积模块。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# --- 核心配置 ---
FEATURE_DIR = r"Feature_PowerSpectrumEntropy_LDS_Smoothed_4x8x9_AllData" 
TEST_SPLIT_SIZE = 0.2
OUTPUT_DIR_BASE = f"1SNN_Depthwise_RandomSplit" 
OUTPUT_SIZE = 3
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0
RANDOM_SEED = 42
TRAINING_GAMMA = 10.0
NUM_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_MIN_DELTA = 0.0001

# --- 超参数网格 ---
hyperparameter_grid = {
    'LEARNING_RATE': [5e-4],
    'CONV_CHANNELS': [[8, 16]], 
    'LAMBDA_L2': [0],
    'DROPOUT_RATE': [0],
    'BATCH_SIZE': [8],
    'CONV_KERNEL_SIZE': [3],
    'HIDDEN_UNITS_1': [64],
    'HIDDEN_UNITS_2': [32],
}

# --- 固定参数 ---
fixed_parameters_for_naming: Dict[str, Union[str, int, float]] = {
    'FEATURE_DIR': FEATURE_DIR, 'OUTPUT_DIR_BASE': OUTPUT_DIR_BASE,
    'OUTPUT_SIZE': OUTPUT_SIZE, 'T_MIN_INPUT': T_MIN_INPUT, 'T_MAX_INPUT': T_MAX_INPUT,
    'RANDOM_SEED': RANDOM_SEED, 'TRAINING_GAMMA': TRAINING_GAMMA, 'NUM_EPOCHS': NUM_EPOCHS,
    'TEST_SPLIT_SIZE': TEST_SPLIT_SIZE,
    'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
    'EARLY_STOPPING_MIN_DELTA': EARLY_STOPPING_MIN_DELTA
}

# --- 数据加载与模型函数 (保持不变) ---
def load_features_from_mat(feature_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    fpath = os.path.join(feature_dir, "all_features_lds_smoothed.mat") # 确保文件名正确
    print(f"Loading data from: {fpath}")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")
    mat_data = loadmat(fpath)
    combined_features = mat_data['features'].astype(np.float32)
    combined_labels = mat_data['labels'].flatten()
    label_mapping = {-1: 0, 0: 1, 1: 2}
    valid_labels_indices = np.isin(combined_labels, list(label_mapping.keys()))
    features_filtered = combined_features[valid_labels_indices]
    labels_mapped = np.array([label_mapping[lbl] for lbl in combined_labels[valid_labels_indices]], dtype=np.int64)
    return features_filtered, labels_mapped

class NumericalEEGDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: np.ndarray):
        self.features = features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self) -> int: return len(self.labels)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

# ... custom_weight_init, train_epoch, evaluate_model, build_filename_prefix, plot_history, save_model_torch ...
# ... 这些函数都保持不变 ...
def custom_weight_init(m: nn.Module):
    if isinstance(m, SpikingDense) and m.kernel is not None:
        input_dim = m.kernel.shape[0];
        if input_dim > 0: stddev = 1.0 / np.sqrt(input_dim); m.kernel.data.normal_(mean=0.0, std=stddev)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)

def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, epoch: int, gamma_ttfs: float, t_min_input: float, t_max_input: float) -> Tuple[float, float]:
    model.train(); running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        outputs, min_ti_list = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
        with torch.no_grad():
            snn_input_t_max = t_max_input
            for layer in model.layers_list:
                if isinstance(layer, DivisionFreeAnnToSnnEncoder): snn_input_t_max = layer.t_max; break
            current_t_min_layer = torch.tensor(snn_input_t_max, dtype=torch.float32, device=device)
            t_min_prev_layer = torch.tensor(t_min_input, dtype=torch.float32, device=device)
            k = 0
            for layer in model.layers_list:
                if isinstance(layer, SpikingDense):
                    if not layer.outputLayer:
                        min_ti_for_layer = min_ti_list[k] if k < len(min_ti_list) else None
                        base_interval = torch.tensor(1.0, dtype=torch.float32, device=device)
                        new_t_max_layer = current_t_min_layer + base_interval
                        if min_ti_for_layer is not None:
                            positive_spike_times = min_ti_for_layer[min_ti_for_layer < layer.t_max]
                            if positive_spike_times.numel() > 0:
                                earliest_spike = torch.min(positive_spike_times)
                                if layer.t_max > earliest_spike:
                                    dynamic_term = gamma_ttfs * (layer.t_max - earliest_spike)
                                    new_t_max_layer = current_t_min_layer + torch.maximum(base_interval, dynamic_term)
                                    new_t_max_layer = torch.clamp(new_t_max_layer, max=current_t_min_layer + 100.0)
                        k += 1
                    else: new_t_max_layer = current_t_min_layer + 1.0
                    layer.set_time_params(t_min_prev_layer, current_t_min_layer, new_t_max_layer)
                    t_min_prev_layer = current_t_min_layer.clone(); current_t_min_layer = new_t_max_layer.clone()
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item(); total_samples += labels.size(0)
    return running_loss / total_samples, correct_predictions / total_samples

def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, List, List]:
    model.eval(); running_loss, correct_predictions, total_samples = 0.0, 0, 0; all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs, _ = model(features)
            loss = criterion(outputs, labels); running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1); correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0); all_labels.extend(labels.cpu().numpy()); all_preds.extend(predicted.cpu().numpy())
    return running_loss / total_samples, correct_predictions / total_samples, all_labels, all_preds

def build_filename_prefix(params: Dict[str, Any]) -> str:
    h1, h2 = params.get('HIDDEN_UNITS_1', 'h1NA'), params.get('HIDDEN_UNITS_2', 'h2NA')
    lr, bs = params.get('LEARNING_RATE', 'lrNA'), params.get('BATCH_SIZE', 'bsNA')
    conv_channels_list = params.get('CONV_CHANNELS', []); 
    if conv_channels_list and isinstance(conv_channels_list[0], list): channels_to_join = conv_channels_list[0]
    else: channels_to_join = conv_channels_list
    conv_ch_str = "-".join(map(str, channels_to_join)); conv_k = params.get('CONV_KERNEL_SIZE', 'kNA')
    lr_str = f"{lr:.0e}".replace('e-0', 'e-'); dp_rate = params.get('DROPOUT_RATE', 'dpNA')
    l2_lambda = params.get('LAMBDA_L2', 'l2NA'); return f"SNN_dsc_conv{conv_ch_str}_h{h1}-{h2}_lr{lr_str}_bs{bs}_dp{dp_rate}_l2_{l2_lambda}"

def plot_history(*args, **kwargs):
    train_losses, val_losses, train_accuracies, val_accuracies, train_lrs, filename_prefix, save_dir, stopped_epoch = args; epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5)); title_suffix = f" (Early stopped at epoch {stopped_epoch})" if stopped_epoch else ""
    plt.subplot(1, 3, 1); plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss'); plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss'); plt.title(f'Loss Curve{title_suffix}'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy'); plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy'); plt.title(f'Accuracy Curve{title_suffix}'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(epochs_range, train_lrs, 'go-', label='Learning Rate'); plt.title(f'Learning Rate Curve{title_suffix}'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.legend(); plt.grid(True); plt.yscale('log')
    plt.tight_layout(); timestamp = time.strftime("%Y%m%d_%H%M%S"); filename = os.path.join(save_dir, f"training_history_and_lr_{filename_prefix}_{timestamp}.png")
    plt.savefig(filename); plt.close(); print(f"训练历史和学习率图已保存为: {filename}")

def save_model_torch(*args, **kwargs):
    model, filename_prefix, save_dir = args
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S"); save_path = os.path.join(save_dir, f"模型_{filename_prefix}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path); print(f"模型成功保存至: {save_path}")

# --- 主训练流程 ---
def run_training_session(current_hyperparams: Dict, fixed_params_dict: Dict, run_id: int) -> float:
    # --- 1. 参数解包 ---
    all_params = {**fixed_params_dict, **current_hyperparams}
    # ...
    LEARNING_RATE_INITIAL, BATCH_SIZE, CONV_CHANNELS_CONFIG, CONV_KERNEL_SIZE_PARAM, HIDDEN_UNITS_1, HIDDEN_UNITS_2, DROPOUT_RATE, lambda_l2, _MAX_NUM_EPOCHS, _TRAINING_GAMMA_TTFS, _EARLY_STOPPING_PATIENCE, _EARLY_STOPPING_MIN_DELTA, T_MIN_INPUT, T_MAX_INPUT = (
    all_params['LEARNING_RATE'], all_params['BATCH_SIZE'], all_params['CONV_CHANNELS'], all_params['CONV_KERNEL_SIZE'], all_params['HIDDEN_UNITS_1'], all_params['HIDDEN_UNITS_2'], all_params['DROPOUT_RATE'], all_params.get('LAMBDA_L2', 0), all_params['NUM_EPOCHS'], all_params['TRAINING_GAMMA'], all_params['EARLY_STOPPING_PATIENCE'], all_params['EARLY_STOPPING_MIN_DELTA'], all_params['T_MIN_INPUT'], all_params['T_MAX_INPUT'])

    # --- 2. 路径与日志 ---
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_prefix_for_dir = build_filename_prefix(all_params)
    run_specific_output_dir_name = f"{base_prefix_for_dir}_{run_timestamp}"
    run_specific_output_dir = os.path.join(all_params['OUTPUT_DIR_BASE'], run_specific_output_dir_name)
    if not os.path.exists(run_specific_output_dir): os.makedirs(run_specific_output_dir, exist_ok=True)
    print(f"\n--- Starting Run ID: {run_id} ---")
    print(f"Current Hyperparameters: {json.dumps(current_hyperparams, indent=2)}")
    print(f"Output will be saved to: {run_specific_output_dir}")
    
    # --- 3. 环境与设备 ---
    torch.manual_seed(all_params['RANDOM_SEED']); np.random.seed(all_params['RANDOM_SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda': torch.cuda.manual_seed_all(all_params['RANDOM_SEED'])
    print(f"Using device: {device}")
    
    # --- 4. 加载全部数据后随机划分 ---
    try:
        features_data, labels_data = load_features_from_mat(all_params['FEATURE_DIR'])
    except FileNotFoundError as e:
        print(e); return 0.0
    
    print(f"Data loaded. Total samples: {len(features_data)}. Raw feature shape: {features_data.shape[1:]}") 
    
    X_train_full, X_val_full, y_train, y_val = train_test_split(
        features_data, labels_data, test_size=all_params['TEST_SPLIT_SIZE'],
        random_state=all_params['RANDOM_SEED'], stratify=labels_data
    )

    X_train_data = torch.tensor(X_train_full, dtype=torch.float32)
    X_val_data = torch.tensor(X_val_full, dtype=torch.float32)
    train_dataset = NumericalEEGDataset(X_train_data, y_train)
    val_dataset = NumericalEEGDataset(X_val_data, y_val)
    train_loader = DataLoader(train_dataset, batch_size=all_params['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=all_params['BATCH_SIZE'], shuffle=False, num_workers=0)

    # --- 5. 模型构建 ---
    model = SNNModel()
    in_channels = 4
    ann_layers = []
    strides = [1, 2] 
    
    # 【修改2】在模型构建中用 DepthwiseSeparableConv 替换 nn.Conv2d
    for i, out_channels in enumerate(CONV_CHANNELS_CONFIG):
        ann_layers.extend([
            DepthwiseSeparableConv(
                in_channels, 
                out_channels, 
                kernel_size=CONV_KERNEL_SIZE_PARAM, 
                stride=strides[i]
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        in_channels = out_channels
    
    model.add(nn.Sequential(*ann_layers))
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 4, 8, 9)
        cnn_part = nn.Sequential(*ann_layers)
        dummy_output = cnn_part(dummy_input)
        flattened_dim = dummy_output.numel()
    
    print(f"Hybrid model (Depthwise Separable Conv) created. Flattened dimension before SNN: {flattened_dim}")

    model.add(DivisionFreeAnnToSnnEncoder(t_min=T_MIN_INPUT, t_max=T_MAX_INPUT))
    model.add(nn.Flatten())
    if DROPOUT_RATE > 0:
        model.add(nn.Dropout(p=DROPOUT_RATE))
    
    model.add(SpikingDense(HIDDEN_UNITS_1, 'dense_1', input_dim=flattened_dim))
    model.add(SpikingDense(HIDDEN_UNITS_2, 'dense_2', input_dim=HIDDEN_UNITS_1))
    model.add(SpikingDense(all_params['OUTPUT_SIZE'], 'dense_output', input_dim=HIDDEN_UNITS_2, outputLayer=True))
    
    model.apply(custom_weight_init)
    model.to(device)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # --- 6. 优化器、损失函数和训练循环 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL, weight_decay=lambda_l2)
    scheduler = CosineAnnealingLR(optimizer, T_max=_MAX_NUM_EPOCHS, eta_min=1e-6)
    
    # ... (训练循环和后续代码保持不变) ...
    start_time_run = time.time()
    train_losses, val_losses, train_accuracies, val_accuracies, learning_rates = [], [], [], [], []
    best_val_acc, patience_counter, best_model_state_dict, stopped_epoch = 0.0, 0, None, _MAX_NUM_EPOCHS
    print(f"Starting training for up to {_MAX_NUM_EPOCHS} epochs...")
    for epoch in range(_MAX_NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, _TRAINING_GAMMA_TTFS, T_MIN_INPUT, T_MAX_INPUT)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        train_losses.append(train_loss); train_accuracies.append(train_acc)
        val_losses.append(val_loss); val_accuracies.append(val_acc)
        print(f"Epoch [{epoch+1}/{_MAX_NUM_EPOCHS}] | LR: {learning_rates[-1]:.2e} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Time: {time.time() - epoch_start_time:.2f}s")
        if val_acc > best_val_acc + _EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc; patience_counter = 0; best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  > Validation accuracy improved to {best_val_acc:.4f}. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= _EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}."); stopped_epoch = epoch + 1; break
    if best_model_state_dict: model.load_state_dict(best_model_state_dict)
    print(f"--- SNN Training Finished (Run ID: {run_id}). Total time: {time.time() - start_time_run:.2f}s ---")
    files_internal_prefix = build_filename_prefix(all_params); save_model_torch(model, files_internal_prefix, run_specific_output_dir)
    plot_history(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, files_internal_prefix, run_specific_output_dir, stopped_epoch)
    final_val_loss, final_val_acc, final_labels, final_preds = evaluate_model(model, val_loader, criterion, device)
    print(f"Final validation accuracy (from best model): {final_val_acc:.4f}")
    report = classification_report(final_labels, final_preds, target_names=['Negative (0)', 'Neutral (1)', 'Positive (2)'], digits=4, zero_division=0)
    print("\nClassification Report (on validation set, using best model):"); print(report)
    report_filename = os.path.join(run_specific_output_dir, f"classification_report_{files_internal_prefix}.txt")
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"Hyperparameters: {json.dumps(current_hyperparams, indent=4)}\n\n"); f.write(f"Best validation accuracy: {best_val_acc:.4f}\n\n")
        f.write("Classification Report:\n"); f.write(report)
    print(f"Classification report saved to: {report_filename}")
    return best_val_acc
    
# --- 主程序入口 ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    
    # 动态构建超参数组合
    keys, values = zip(*hyperparameter_grid.items())
    
    # 处理特殊配对的超参数，例如 HIDDEN_UNITS
    h1_options = hyperparameter_grid.get('HIDDEN_UNITS_1', [])
    h2_options = hyperparameter_grid.get('HIDDEN_UNITS_2', [])
    
    # 确保h1和h2的列表长度一致，以便配对
    max_len = max(len(h1_options), len(h2_options))
    h1_options_padded = h1_options + [h1_options[-1]] * (max_len - len(h1_options))
    h2_options_padded = h2_options + [h2_options[-1]] * (max_len - len(h2_options))
    snn_layer_combos = list(zip(h1_options_padded, h2_options_padded))

    # 创建不包含这些特殊参数的基础组合
    base_grid = {k: v for k, v in hyperparameter_grid.items() if k not in ['HIDDEN_UNITS_1', 'HIDDEN_UNITS_2']}
    base_keys, base_values = zip(*base_grid.items())
    base_combinations = [dict(zip(base_keys, v)) for v in itertools.product(*base_values)]
    
    # 将基础组合与特殊配对组合进行合并
    hyperparam_combinations = []
    for base_combo in base_combinations:
        for h1, h2 in snn_layer_combos:
            new_combo = base_combo.copy()
            new_combo['HIDDEN_UNITS_1'] = h1
            new_combo['HIDDEN_UNITS_2'] = h2
            hyperparam_combinations.append(new_combo)

    num_combinations = len(hyperparam_combinations)
    
    print(f"Starting grid search for {num_combinations} hyperparameter combination(s).")
    
    best_accuracy_overall = 0.0
    best_hyperparams_combo_overall = None
    all_results_summary = []
    
    for i, params_combo_iter in enumerate(hyperparam_combinations):
        validation_accuracy_for_run = run_training_session(
            current_hyperparams=params_combo_iter,
            fixed_params_dict=fixed_parameters_for_naming,
            run_id=i+1
        )
        run_summary = { 'run_id': i+1, 'hyperparameters': params_combo_iter, 'best_validation_accuracy': validation_accuracy_for_run }
        all_results_summary.append(run_summary)
        
        if validation_accuracy_for_run > best_accuracy_overall:
            best_accuracy_overall = validation_accuracy_for_run
            best_hyperparams_combo_overall = params_combo_iter
    
    print("\n--- Grid Search Finished ---")
    if best_hyperparams_combo_overall:
        print(f"Best overall validation accuracy: {best_accuracy_overall:.4f}")
        print(f"Best hyperparameter combination: {json.dumps(best_hyperparams_combo_overall, indent=2)}")
        
        summary_file_path = os.path.join(OUTPUT_DIR_BASE, f"grid_search_summary_{time.strftime('%Y%m%d_%H%M%S')}.json")
        summary_data = {
            "best_overall_validation_accuracy": best_accuracy_overall,
            "best_hyperparameters": best_hyperparams_combo_overall,
            "all_run_results": all_results_summary
        }
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Grid search summary saved to: {summary_file_path}")
    else:
        print("No successful training runs were completed.")
    print("Script execution finished.")