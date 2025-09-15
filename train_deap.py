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
from sklearn.utils.class_weight import compute_class_weight


from model.TTFS import SNNModel, SpikingDense, DivisionFreeAnnToSnnEncoder

# --- 深度可分离卷积模块的定义 ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# --- 核心配置 ---
# <<< 已确认: 输入目录为上一步生成的DEAP特征目录
FEATURE_DIR = r"DEAP_Feature_PowerSpectrumEntropy_LDS_Smoothed_6x7x5"
TEST_SPLIT_SIZE = 0.2
OUTPUT_DIR_BASE = "SNN_SubjectIndependent_DEAP_6x7x5"
OUTPUT_SIZE = 4  # DEAP的四分类情感
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0
RANDOM_SEED = 42
TRAINING_GAMMA = 10.0
NUM_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 0.0001

# --- 超参数网格 ---
hyperparameter_grid = {
    'LEARNING_RATE': [5e-4],
    'CONV_CHANNELS': [[12, 24]],
    'LAMBDA_L2': [0],
    'DROPOUT_RATE': [0],
    'BATCH_SIZE': [8],
    'CONV_KERNEL_SIZE': [3],
    'HIDDEN_UNITS_1': [64],
    'HIDDEN_UNITS_2': [32],
}

# --- 固定参数 ---
fixed_parameters_for_naming: Dict[str, Any] = {
    'FEATURE_DIR': FEATURE_DIR, 'OUTPUT_DIR_BASE': OUTPUT_DIR_BASE,
    'OUTPUT_SIZE': OUTPUT_SIZE, 'T_MIN_INPUT': T_MIN_INPUT, 'T_MAX_INPUT': T_MAX_INPUT,
    'RANDOM_SEED': RANDOM_SEED, 'TRAINING_GAMMA': TRAINING_GAMMA, 'NUM_EPOCHS': NUM_EPOCHS,
    'TEST_SPLIT_SIZE': TEST_SPLIT_SIZE,
    'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
    'EARLY_STOPPING_MIN_DELTA': EARLY_STOPPING_MIN_DELTA
}

# --- 数据加载与模型函数 ---
def load_features_from_mat(feature_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    # <<< 【修正1】: 更新为正确的DEAP特征文件名 (6x7x5)
    fpath = os.path.join(feature_dir, "all_features_deap_lds_smoothed_6x7x5.mat")
    print(f"Loading data from: {fpath}")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data file not found: {fpath}")

    mat_data = loadmat(fpath)
    # 输入特征形状已经是 (N, 6, 7, 5)，符合Conv2d要求
    features = mat_data['features'].astype(np.float32)
    labels = mat_data['labels'].flatten().astype(np.int64)
    
    print(f"Features loaded, shape: {features.shape}")
    print(f"Labels loaded, shape: {labels.shape}")
    print(f"Unique labels found: {np.unique(labels)}")

    return features, labels

def build_filename_prefix(params: Dict[str, Any]) -> str:
    """Creates a standardized filename prefix from hyperparameters."""
    h1 = params.get('HIDDEN_UNITS_1', 'h1NA')
    h2 = params.get('HIDDEN_UNITS_2', 'h2NA')
    lr = params.get('LEARNING_RATE', 'lrNA')
    bs = params.get('BATCH_SIZE', 'bsNA')
    conv_channels_list = params.get('CONV_CHANNELS', [])
    
    channels_to_join = conv_channels_list[0] if conv_channels_list and isinstance(conv_channels_list[0], list) else conv_channels_list
        
    conv_ch_str = "-".join(map(str, channels_to_join))
    lr_str = f"{lr:.0e}".replace('e-0', 'e-')
    dp_rate = params.get('DROPOUT_RATE', 'dpNA')
    l2_lambda = params.get('LAMBDA_L2', 'l2NA')
    
    return (f"SNN_dsc_conv{conv_ch_str}_h{h1}-{h2}_lr{lr_str}_"
            f"bs{bs}_dp{dp_rate}_l2_{l2_lambda}")

def plot_history(train_losses, val_losses, train_accuracies, val_accuracies, 
                 train_lrs, filename_prefix, save_dir, stopped_epoch):
    """Saves plots of the training and validation history."""
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))
    title_suffix = f" (Early stopped at epoch {stopped_epoch})" if stopped_epoch < NUM_EPOCHS else ""

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'Loss Curve{title_suffix}'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title(f'Accuracy Curve{title_suffix}'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_lrs, 'go-', label='Learning Rate')
    plt.title(f'Learning Rate Curve{title_suffix}'); plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
    plt.legend(); plt.grid(True); plt.yscale('log')

    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"history_{filename_prefix}_{timestamp}.png")
    plt.savefig(filename)
    plt.close()
    print(f"Training history plot saved to: {filename}")

def save_model_torch(model, filename_prefix, save_dir):
    """Saves the trained PyTorch model."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_{filename_prefix}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to: {save_path}")

class NumericalEEGDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
    def __len__(self) -> int: return len(self.labels)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

def custom_weight_init(m: nn.Module):
    if isinstance(m, SpikingDense) and m.kernel is not None:
        if m.kernel.shape[0] > 0: stddev = 1.0 / np.sqrt(m.kernel.shape[0]); m.kernel.data.normal_(mean=0.0, std=stddev)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)

# The train_epoch and evaluate_model functions remain the same as they are general
def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, epoch: int, gamma_ttfs: float, t_min_input: float, t_max_input: float) -> Tuple[float, float]:
    model.train(); running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        outputs, min_ti_list = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
        with torch.no_grad():
            snn_input_t_max = next((layer.t_max for layer in model.layers_list if isinstance(layer, DivisionFreeAnnToSnnEncoder)), t_max_input)
            current_t_min_layer, t_min_prev_layer = torch.tensor(snn_input_t_max, device=device), torch.tensor(t_min_input, device=device)
            k=0
            for layer in model.layers_list:
                if isinstance(layer, SpikingDense):
                    if not layer.outputLayer:
                        min_ti_for_layer = min_ti_list[k] if k < len(min_ti_list) else None; base_interval = torch.tensor(1.0, device=device); new_t_max_layer = current_t_min_layer + base_interval
                        if min_ti_for_layer is not None:
                            positive_spike_times = min_ti_for_layer[min_ti_for_layer < layer.t_max]
                            if positive_spike_times.numel() > 0:
                                earliest_spike = torch.min(positive_spike_times)
                                if layer.t_max > earliest_spike: new_t_max_layer = current_t_min_layer + torch.maximum(base_interval, gamma_ttfs * (layer.t_max - earliest_spike))
                        k+=1
                    else: new_t_max_layer = current_t_min_layer + 1.0
                    layer.set_time_params(t_min_prev_layer, current_t_min_layer, new_t_max_layer); t_min_prev_layer = current_t_min_layer.clone(); current_t_min_layer = new_t_max_layer.clone()
        running_loss += loss.item() * features.size(0); _, predicted = torch.max(outputs.data, 1); correct_predictions += (predicted == labels).sum().item(); total_samples += labels.size(0)
    return running_loss / total_samples, correct_predictions / total_samples

def evaluate_model(model: SNNModel, dataloader: DataLoader, criterion: nn.Module, device: torch.device, gamma_ttfs: float, t_min_input: float, t_max_input: float) -> Tuple[float, float, List, List]:
    model.eval(); running_loss, correct_predictions, total_samples = 0.0, 0, 0; all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs, min_ti_list = model(features)
            snn_input_t_max = next((layer.t_max for layer in model.layers_list if isinstance(layer, DivisionFreeAnnToSnnEncoder)), t_max_input)
            current_t_min_layer, t_min_prev_layer = torch.tensor(snn_input_t_max, device=device), torch.tensor(t_min_input, device=device)
            k=0
            for layer in model.layers_list:
                if isinstance(layer, SpikingDense):
                    if not layer.outputLayer:
                        min_ti_for_layer = min_ti_list[k] if k < len(min_ti_list) else None; base_interval = torch.tensor(1.0, device=device); new_t_max_layer = current_t_min_layer + base_interval
                        if min_ti_for_layer is not None:
                            positive_spike_times = min_ti_for_layer[min_ti_for_layer < layer.t_max]
                            if positive_spike_times.numel() > 0:
                                earliest_spike = torch.min(positive_spike_times)
                                if layer.t_max > earliest_spike: new_t_max_layer = current_t_min_layer + torch.maximum(base_interval, gamma_ttfs * (layer.t_max - earliest_spike))
                        k+=1
                    else: new_t_max_layer = current_t_min_layer + 1.0
                    layer.set_time_params(t_min_prev_layer, current_t_min_layer, new_t_max_layer); t_min_prev_layer = current_t_min_layer.clone(); current_t_min_layer = new_t_max_layer.clone()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0); _, predicted = torch.max(outputs.data, 1); correct_predictions += (predicted == labels).sum().item(); total_samples += labels.size(0); all_labels.extend(labels.cpu().numpy()); all_preds.extend(predicted.cpu().numpy())
    return running_loss / total_samples, correct_predictions / total_samples, all_labels, all_preds

# --- 主训练流程 ---
def run_training_session(current_hyperparams: Dict, fixed_params_dict: Dict, run_id: int) -> float:
    # --- 1. Unpack Parameters ---
    all_params = {**fixed_params_dict, **current_hyperparams}
    lr, batch_size, conv_channels_config, kernel_size, h1, h2, dropout, l2, epochs, gamma, patience, min_delta, t_min, t_max = (
        all_params['LEARNING_RATE'], all_params['BATCH_SIZE'], all_params['CONV_CHANNELS'], 
        all_params['CONV_KERNEL_SIZE'], all_params['HIDDEN_UNITS_1'], all_params['HIDDEN_UNITS_2'], 
        all_params['DROPOUT_RATE'], all_params.get('LAMBDA_L2', 0), all_params['NUM_EPOCHS'], 
        all_params['TRAINING_GAMMA'], all_params['EARLY_STOPPING_PATIENCE'], 
        all_params['EARLY_STOPPING_MIN_DELTA'], all_params['T_MIN_INPUT'], all_params['T_MAX_INPUT'])

    # --- 2. Setup Paths & Logging ---
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_prefix_for_dir = build_filename_prefix(all_params)
    run_specific_output_dir = os.path.join(all_params['OUTPUT_DIR_BASE'], f"{base_prefix_for_dir}_{run_timestamp}")
    os.makedirs(run_specific_output_dir, exist_ok=True)
    print(f"\n--- Starting Run ID: {run_id} ---")
    print(f"Hyperparameters: {json.dumps(current_hyperparams, indent=2)}")
    print(f"Output will be saved to: {run_specific_output_dir}")

    # --- 3. Setup Device ---
    torch.manual_seed(all_params['RANDOM_SEED'])
    np.random.seed(all_params['RANDOM_SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 4. Load, Preprocess, and Split Data ---
    try:
        features_data, labels_data = load_features_from_mat(all_params['FEATURE_DIR'])
    except FileNotFoundError as e:
        print(e)
        return 0.0

    X_train_full, X_val_full, y_train, y_val = train_test_split(
        features_data, labels_data, test_size=all_params['TEST_SPLIT_SIZE'],
        random_state=all_params['RANDOM_SEED'], stratify=labels_data
    )
    
    # Z-score normalization
    mu = np.mean(X_train_full, axis=0, keepdims=True)
    std = np.std(X_train_full, axis=0, keepdims=True)
    std[std == 0] = 1e-8
    X_train_normalized = (X_train_full - mu) / std
    X_val_normalized = (X_val_full - mu) / std

    # <<< ADDED: Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Calculated class weights for loss function: {class_weights}")

    train_dataset = NumericalEEGDataset(torch.tensor(X_train_normalized, dtype=torch.float32), y_train)
    val_dataset = NumericalEEGDataset(torch.tensor(X_val_normalized, dtype=torch.float32), y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 5. Build Model ---
    model = SNNModel()
    in_channels = features_data.shape[1] # Should be 4 for DEAP features
    ann_layers = []
    strides = [1, 2] 、

    for i, out_channels in enumerate(conv_channels_config):
        ann_layers.extend([
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel_size, stride=strides[i]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        in_channels = out_channels
    
    cnn_part = nn.Sequential(*ann_layers)
    model.add(cnn_part)

    with torch.no_grad():
        # Use the correct feature shape for the dummy input
        dummy_input = torch.randn(1, features_data.shape[1], features_data.shape[2], features_data.shape[3])
        dummy_output = cnn_part(dummy_input)
        flattened_dim = dummy_output.numel()

    print(f"Hybrid model created. Flattened dimension before SNN: {flattened_dim}")

    model.add(DivisionFreeAnnToSnnEncoder(t_min=t_min, t_max=t_max))
    model.add(nn.Flatten())
    if dropout > 0:
        model.add(nn.Dropout(p=dropout))
    model.add(SpikingDense(h1, 'dense_1', input_dim=flattened_dim))
    model.add(SpikingDense(h2, 'dense_2', input_dim=h1))
    model.add(SpikingDense(all_params['OUTPUT_SIZE'], 'dense_output', input_dim=h2, outputLayer=True))

    model.apply(custom_weight_init)
    model.to(device)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- 6. Training Loop ---
    # <<< MODIFIED: Apply class weights to the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

    start_time_run = time.time()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc, patience_counter, best_model_state_dict, stopped_epoch = 0.0, 0, None, epochs
    
    print(f"Starting training for up to {epochs} epochs...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, gamma, t_min, t_max)
        
        # <<< CORRECTED: Pass SNN time parameters to the evaluation function
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device, gamma, t_min, t_max)
        
        history['lr'].append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{epochs}] - Time: {epoch_duration:.2f}s - LR: {history['lr'][-1]:.2e} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  > Validation accuracy improved to {best_val_acc:.4f}. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                stopped_epoch = epoch + 1
                break
                
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
        
    print(f"--- SNN Training Finished (Run ID: {run_id}). Total time: {time.time() - start_time_run:.2f}s ---")
    
    files_internal_prefix = build_filename_prefix(all_params)
    save_model_torch(model, files_internal_prefix, run_specific_output_dir)
    plot_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'], history['lr'], files_internal_prefix, run_specific_output_dir, stopped_epoch)
    
    # <<< CORRECTED: Pass SNN time parameters for final evaluation
    final_val_loss, final_val_acc, final_labels, final_preds = evaluate_model(model, val_loader, criterion, device, gamma, t_min, t_max)
    print(f"Final validation accuracy (from best model): {final_val_acc:.4f}")

    # <<< MODIFIED: Update report labels for DEAP
    report_target_names = ['LVLA (0)', 'HVLA (1)', 'LVHA (2)', 'HVHA (3)']
    report = classification_report(final_labels, final_preds, target_names=report_target_names, digits=4, zero_division=0)

    print("\nClassification Report (on validation set, using best model):")
    print(report)
    
    report_filename = os.path.join(run_specific_output_dir, f"classification_report_{files_internal_prefix}.txt")
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"Hyperparameters: {json.dumps(current_hyperparams, indent=4)}\n\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Classification report saved to: {report_filename}")
    
    return best_val_acc

# --- 主程序入口 ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    keys, values = zip(*hyperparameter_grid.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting grid search for {len(hyperparam_combinations)} hyperparameter combination(s).")
    
    best_accuracy_overall = 0.0; best_hyperparams_combo_overall = None
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