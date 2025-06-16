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

from typing import Dict, Union, List, Optional, Tuple, Any
import time
import matplotlib.pyplot as plt
import json
import itertools
from model.TTFS_ORIGN import SNNModel, SpikingDense  # Ensure this model definition is accessible

# Fixed parameters
FEATURE_DIR = r"tmp_zip_extract/IV_Individual_Features_Variable_BaselineCorrected_MP"
OUTPUT_DIR_BASE = r"0530_grid"  
OUTPUT_SIZE = 4  
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
TRAINING_GAMMA = 10.0
NUM_EPOCHS = 200
LR_SCHEDULER_GAMMA = 0.99
EARLY_STOPPING_PATIENCE = 20  
EARLY_STOPPING_MIN_DELTA = 0.0001 

# Hyperparameter grid
hyperparameter_grid = {
    'LEARNING_RATE': [1e-3,5e-4],
    'BATCH_SIZE': [8],
    'HIDDEN_UNITS_1': [600,512],
    'HIDDEN_UNITS_2': [384,256],
    'LAMBDA_L1': [0,1e-4],
}

fixed_params_dict = {
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

# Load feature data
def load_features_from_mat(feature_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_features.mat"))
    for fpath in mat_files:
        mat_data = loadmat(fpath)
        features = mat_data['features'].astype(np.float32)
        labels = mat_data['labels'].flatten().astype(np.int64)  # SEED-IV labels are 0,1,2,3
        all_features.append(features)
        all_labels.append(labels)
    if not all_features:
        raise ValueError(f"No feature files found in directory {feature_dir}.")
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    return combined_features, combined_labels

# TTFS encoding
def ttfs_encode(features: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
    features = np.clip(features, 0.0, 1.0)
    spike_times = t_max - features * (t_max - t_min)
    return torch.tensor(spike_times, dtype=torch.float32)

# Dataset class
class EncodedEEGDataset(Dataset):
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray):
        self.features = encoded_features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

# Custom weight initialization
def custom_weight_init(m: nn.Module):
    if isinstance(m, SpikingDense):
        if hasattr(m, 'kernel') and m.kernel is not None:
            kernel_data = m.kernel.data if hasattr(m.kernel, 'data') else m.kernel
            # PyTorch nn.Linear weight shape: (out_features, in_features)
            # fan_in (input features) is kernel_data.shape[1]
            input_dim_for_init = kernel_data.shape[1] if kernel_data.dim() > 1 else kernel_data.shape[0]
            
            if kernel_data.numel() == 0:  # Check if empty tensor
                return
            if input_dim_for_init == 0:
                stddev = 0.01  # If fan_in is 0, use a small default value
            else:
                stddev = 1.0 / np.sqrt(input_dim_for_init)

            with torch.no_grad():
                kernel_data.normal_(mean=0.0, std=stddev)

# Train one epoch (integrated with dynamic time parameter adjustment)
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
        
        # Forward pass
        outputs, min_ti_list = model(features)  # Assume model returns outputs and min_ti_list
        
        # Compute loss
        primary_loss = criterion(outputs, labels)
        l1_reg = torch.tensor(0.0, device=device)
        if lambda_l1 > 0:
            # Ensure model.layers_list exists and SpikingDense has kernel attribute
            if hasattr(model, 'layers_list'):
                l1_reg = sum(torch.sum(torch.abs(layer.kernel)) for layer in model.layers_list if isinstance(layer, SpikingDense) and hasattr(layer, 'kernel') and layer.kernel is not None)
        loss = primary_loss + lambda_l1 * l1_reg
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss and accuracy metrics
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # --- Start dynamic time parameter adjustment ---
        with torch.no_grad():
            effective_current_t_min_input = torch.tensor(current_t_min_input, dtype=torch.float32, device=device)
            t_min_input_to_current_layer = effective_current_t_min_input
            t_min_output_from_current_layer = torch.tensor(current_t_max_input, dtype=torch.float32, device=device)

            k = 0  # Index for min_ti_list
            if hasattr(model, 'layers_list'):
                for layer_idx, layer in enumerate(model.layers_list):
                    if isinstance(layer, SpikingDense):
                        t_max_output_from_current_layer = t_min_output_from_current_layer + torch.tensor(1.0, dtype=torch.float32, device=device)

                        if not layer.outputLayer:
                            min_ti_for_this_layer_output = None
                            if k < len(min_ti_list) and min_ti_list[k] is not None:
                                positive_spike_times = min_ti_list[k][min_ti_list[k] > 1e-6]
                                if positive_spike_times.numel() > 0:
                                    min_ti_for_this_layer_output = torch.min(positive_spike_times)
                            
                            base_interval = torch.tensor(1.0, dtype=torch.float32, device=device)

                            if min_ti_for_this_layer_output is not None:
                                current_layer_previously_set_t_max_output = layer.t_max.clone().detach()

                                if current_layer_previously_set_t_max_output > min_ti_for_this_layer_output:
                                    dynamic_term = gamma_ttfs * (current_layer_previously_set_t_max_output - min_ti_for_this_layer_output)
                                    dynamic_term = torch.clamp(dynamic_term, min=0.0)
                                    t_max_output_from_current_layer = t_min_output_from_current_layer + torch.maximum(base_interval, dynamic_term)
                                else:
                                    t_max_output_from_current_layer = t_min_output_from_current_layer + base_interval
                            else:
                                t_max_output_from_current_layer = t_min_output_from_current_layer + base_interval
                            k += 1
                        
                        if hasattr(layer, 'set_time_params'):
                            layer.set_time_params(t_min_input_to_current_layer,
                                                  t_min_output_from_current_layer,
                                                  t_max_output_from_current_layer)
                        else:
                            if epoch == 0 and batch_idx == 0:
                                print(f"Warning: Layer {layer_idx} ({getattr(layer, 'name', 'unnamed')}) does not have set_time_params method. Skipping dynamic time adjustment.")

                        t_min_input_to_current_layer = t_min_output_from_current_layer.clone()
                        t_min_output_from_current_layer = t_max_output_from_current_layer.clone()
            else:
                if epoch == 0 and batch_idx == 0:
                    print("Warning: SNNModel does not have 'layers_list' attribute. Skipping dynamic time parameter adjustment.")
        # --- End dynamic time parameter adjustment ---
            
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

# Evaluate model
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
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc, all_labels, all_preds

# Build filename prefix
def build_filename_prefix(params: Dict[str, Any]) -> str:
    lr = params.get('LEARNING_RATE', 'NA')
    bs = params.get('BATCH_SIZE', 'NA')
    h1 = params.get('HIDDEN_UNITS_1', 'NA')
    h2 = params.get('HIDDEN_UNITS_2', 'NA')
    l1 = params.get('LAMBDA_L1', 'NA')

    lr_str = f"{lr:.1e}" if isinstance(lr, float) else str(lr)
    l1_str = f"{l1:.1e}" if isinstance(l1, float) and l1 != 0 else str(l1)

    return (f"SEEDIV_lr{lr_str}_bs{bs}"
            f"_h1{h1}_h2{h2}_L1_{l1_str}"
            f"_{time.strftime('%Y%m%d_%H%M%S')}")

# Plot history
def plot_history(train_losses, val_losses, train_accuracies, val_accuracies, train_lrs,
                 filename_prefix: str, save_dir: str, stopped_epoch: Optional[int] = None, num_total_epochs: int = NUM_EPOCHS):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(20, 6))
    
    title_suffix = ""
    if stopped_epoch is not None and stopped_epoch < num_total_epochs:
        title_suffix = f" (Early stopped at epoch {stopped_epoch})"
    elif len(train_losses) == num_total_epochs:
        title_suffix = f" (Completed all {num_total_epochs} epochs)"
    else:
        title_suffix = f" (Completed {len(train_losses)}/{num_total_epochs} epochs)"

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'Loss{title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title(f'Accuracy{title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_lrs, 'go-', label='Learning Rate')
    plt.title(f'Learning Rate{title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    if any(lr > 0 for lr in train_lrs):
        plt.yscale('log')
    
    plt.tight_layout(pad=2.0)
    plot_filename = os.path.join(save_dir, f"training_history_{filename_prefix}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Training history plot saved to: {plot_filename}")

# Save model
def save_model_torch(model: SNNModel, filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"model_{filename_prefix}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

# Save parameters
def save_params(params_to_save: Dict[str, Union[int, float, str]], filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"params_{filename_prefix}.json")
    serializable_params = {}
    for k, v in params_to_save.items():
        if isinstance(v, np.integer):
            serializable_params[k] = int(v)
        elif isinstance(v, np.floating):
            serializable_params[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable_params[k] = v.tolist()
        else:
            serializable_params[k] = v
            
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_params, f, indent=4, ensure_ascii=False)
    print(f"Parameters saved to: {save_path}")

def run_training_session(current_hyperparams: Dict, fixed_params_dict_input: Dict, run_id: int) -> float:
    all_params = {**fixed_params_dict_input, **current_hyperparams}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Run ID: {run_id} ---")
    print(f"Current hyperparameters: {json.dumps(current_hyperparams, indent=2, ensure_ascii=False)}")
    print(f"Using device: {device}")

    _FEATURE_DIR = all_params['FEATURE_DIR']
    _OUTPUT_SIZE = all_params['OUTPUT_SIZE']
    _T_MIN_INPUT = all_params['T_MIN_INPUT']
    _T_MAX_INPUT = all_params['T_MAX_INPUT']
    _TEST_SPLIT_SIZE = all_params['TEST_SPLIT_SIZE']
    _RANDOM_SEED = all_params['RANDOM_SEED']
    _TRAINING_GAMMA = all_params['TRAINING_GAMMA']
    _NUM_EPOCHS = all_params['NUM_EPOCHS']
    _LR_SCHEDULER_GAMMA = all_params['LR_SCHEDULER_GAMMA']
    _EARLY_STOPPING_PATIENCE = all_params['EARLY_STOPPING_PATIENCE']
    _EARLY_STOPPING_MIN_DELTA = all_params['EARLY_STOPPING_MIN_DELTA']
    
    _LEARNING_RATE = current_hyperparams['LEARNING_RATE']
    _BATCH_SIZE = current_hyperparams['BATCH_SIZE']
    _HIDDEN_UNITS_1 = current_hyperparams['HIDDEN_UNITS_1']
    _HIDDEN_UNITS_2 = current_hyperparams['HIDDEN_UNITS_2']
    _LAMBDA_L1 = current_hyperparams['LAMBDA_L1']

    torch.manual_seed(_RANDOM_SEED)
    np.random.seed(_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_RANDOM_SEED)

    features_data, labels_data = load_features_from_mat(_FEATURE_DIR)
    print(f"Data loaded successfully. Feature shape: {features_data.shape}, Label shape: {labels_data.shape}")

    X_train_full, X_val_full, y_train, y_val = train_test_split(
        features_data, labels_data, test_size=_TEST_SPLIT_SIZE, random_state=_RANDOM_SEED, stratify=labels_data
    )
    print(f"Training set size: {X_train_full.shape[0]}, Validation set size: {X_val_full.shape[0]}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train_full)
    X_val_norm = scaler.transform(X_val_full)

    X_train_encoded = ttfs_encode(X_train_norm, _T_MIN_INPUT, _T_MAX_INPUT)
    X_val_encoded = ttfs_encode(X_val_norm, _T_MIN_INPUT, _T_MAX_INPUT)
    print(f"Encoded training feature shape: {X_train_encoded.shape}, Encoded validation feature shape: {X_val_encoded.shape}")

    num_workers = 2 if device.type == 'cuda' else 0
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    val_dataset = EncodedEEGDataset(X_val_encoded, y_val)
    train_loader = DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Build SNN model
    model = SNNModel()
    model.add(SpikingDense(units=_HIDDEN_UNITS_1,
                           input_dim=X_train_encoded.shape[1],
                           outputLayer=False,
                           name='hidden_layer_1'))
    model.add(SpikingDense(units=_HIDDEN_UNITS_2,
                           input_dim=_HIDDEN_UNITS_1,
                           outputLayer=False,
                           name='hidden_layer_2'))
    model.add(SpikingDense(units=_OUTPUT_SIZE,
                           input_dim=_HIDDEN_UNITS_2,
                           outputLayer=True,
                           name='output_layer'))
    model.apply(custom_weight_init)
    model.to(device)
    print(f"Total trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if hasattr(model, 'layers_list'):
        print("Setting initial static time parameters for SNN layers...")
        _t_min_afferent = torch.tensor(_T_MIN_INPUT, dtype=torch.float32, device=device)
        _t_min_efferent = torch.tensor(_T_MAX_INPUT, dtype=torch.float32, device=device)

        for layer_idx, layer in enumerate(model.layers_list):
            if isinstance(layer, SpikingDense):
                _t_max_efferent = _t_min_efferent + torch.tensor(1.0, dtype=torch.float32, device=device)
                
                if hasattr(layer, 'set_time_params'):
                    layer.set_time_params(_t_min_afferent, _t_min_efferent, _t_max_efferent)
                else:
                    print(f"Warning: Layer {layer_idx} ({getattr(layer, 'name', 'unnamed')}) does not have set_time_params method. Skipping initial time setting.")

                _t_min_afferent = _t_min_efferent.clone()
                _t_min_efferent = _t_max_efferent.clone()
        print("Initial static time parameters set.")
    else:
        print("Warning: SNNModel does not have 'layers_list' attribute. Skipping initial static time parameter setting.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    scheduler = ExponentialLR(optimizer, gamma=_LR_SCHEDULER_GAMMA)

    train_losses, val_losses, train_accuracies, val_accuracies, lrs_over_epochs = [], [], [], [], []
    best_val_acc_during_training = 0.0
    patience_counter = 0
    best_model_state_dict = None
    actual_stopped_epoch = _NUM_EPOCHS

    print(f"Starting training for up to {_NUM_EPOCHS} epochs...")
    for epoch in range(_NUM_EPOCHS):
        epoch_start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                            _TRAINING_GAMMA, _T_MIN_INPUT, _T_MAX_INPUT, _LAMBDA_L1)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        
        if _LR_SCHEDULER_GAMMA < 1.0:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        lrs_over_epochs.append(current_lr)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{_NUM_EPOCHS}] | LR: {current_lr:.1e} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"Time: {epoch_duration:.2f}s")

        if val_acc > best_val_acc_during_training + _EARLY_STOPPING_MIN_DELTA:
            best_val_acc_during_training = val_acc
            best_model_state_dict = model.state_dict().copy()
            patience_counter = 0
            print(f"  Validation accuracy improved to {best_val_acc_during_training:.4f}. Resetting patience counter.")
        else:
            patience_counter += 1
            print(f"  Validation accuracy did not improve significantly. Patience counter: {patience_counter}/{_EARLY_STOPPING_PATIENCE}")
        
        if patience_counter >= _EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            actual_stopped_epoch = epoch + 1
            break
    
    if best_model_state_dict:
        print("Loading the model state with the highest validation accuracy during training...")
        model.load_state_dict(best_model_state_dict)
    else:
        print("Warning: No better model state was saved during training. Using the model from the last epoch.")
        best_val_acc_during_training = val_accuracies[-1] if val_accuracies else 0.0

    final_val_loss, final_val_acc, final_labels, final_preds = evaluate_model(model, val_loader, criterion, device)
    target_report_names = [f'Class {i}' for i in range(_OUTPUT_SIZE)]
    if _OUTPUT_SIZE == 4:
        target_report_names = ['Neutral(0)', 'Sad(1)', 'Fear(2)', 'Happy(3)']

    report_str = classification_report(final_labels, final_preds, target_names=target_report_names, digits=4, zero_division=0)
    print("\nClassification Report (based on validation set - using the best model):")
    print(report_str)

    filename_prefix_str = build_filename_prefix(all_params)
    
    run_dir_suffix = "_".join(filename_prefix_str.split('_')[:-1])
    run_specific_output_dir = os.path.join(OUTPUT_DIR_BASE, f"run_{run_id}_{run_dir_suffix}")
    if not os.path.exists(run_specific_output_dir):
        os.makedirs(run_specific_output_dir, exist_ok=True)
    
    plot_history(train_losses, val_losses, train_accuracies, val_accuracies, lrs_over_epochs,
                 filename_prefix_str, run_specific_output_dir, actual_stopped_epoch, _NUM_EPOCHS)
    save_model_torch(model, filename_prefix_str, run_specific_output_dir)
    save_params(all_params, filename_prefix_str, run_specific_output_dir)

    report_filename = os.path.join(run_specific_output_dir, f"report_{filename_prefix_str}.txt")
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Current hyperparameters:\n{json.dumps(current_hyperparams, indent=2, ensure_ascii=False)}\n\n")
        fixed_params_to_show = {k: v for k, v in fixed_params_dict_input.items()
                                if k not in ['FEATURE_DIR', 'OUTPUT_DIR_BASE']}
        f.write(f"Fixed parameters (partial):\n{json.dumps(fixed_params_to_show, indent=2, ensure_ascii=False)}\n\n")
        f.write(f"Actual stopped epoch: {actual_stopped_epoch} / {_NUM_EPOCHS}\n")
        f.write(f"Best validation accuracy during training: {best_val_acc_during_training:.4f}\n")
        f.write(f"Validation accuracy after loading best model: {final_val_acc:.4f}\n\n")
        f.write("Classification Report (validation set - best model):\n")
        f.write(report_str)
    print(f"Classification report saved to: {report_filename}")

    return final_val_acc

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

    keys, values = zip(*hyperparameter_grid.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    num_combinations = len(hyperparam_combinations)
    print(f"Will perform training for {num_combinations} hyperparameter combinations.")

    best_accuracy_overall = 0.0
    best_hyperparams_combo_overall = None
    all_results_summary = []
    
    overall_start_time = time.time()

    for i, params_combo in enumerate(hyperparam_combinations):
        print(f"\n{'='*30} Starting combination {i+1}/{num_combinations} {'='*30}")
        run_start_time = time.time()
        
        validation_accuracy = run_training_session(params_combo, fixed_params_dict, i + 1)
        
        run_end_time = time.time()
        run_duration_seconds = run_end_time - run_start_time
        run_duration_str = time.strftime("%H:%M:%S", time.gmtime(run_duration_seconds))

        print(f"Run ID {i+1} completed. Time taken: {run_duration_str} ({run_duration_seconds:.2f} seconds). Final validation accuracy: {validation_accuracy:.4f}")

        current_run_all_params = {**fixed_params_dict, **params_combo}
        all_results_summary.append({
            'Run ID': i + 1,
            'Hyperparameter Combination': params_combo,
            'Final Validation Accuracy': validation_accuracy,
            'Time Taken (seconds)': round(run_duration_seconds, 2),
            'Time Taken (H:M:S)': run_duration_str
        })
        if validation_accuracy > best_accuracy_overall:
            best_accuracy_overall = validation_accuracy
            best_hyperparams_combo_overall = params_combo
            print(f"*** New overall best validation accuracy: {best_accuracy_overall:.4f} (from run ID: {i + 1}) ***")
        print(f"{'='*30} Ending combination {i+1}/{num_combinations} {'='*30}")

    overall_end_time = time.time()
    total_duration_seconds = overall_end_time - overall_start_time
    total_duration_str = time.strftime("%H:%M:%S", time.gmtime(total_duration_seconds))

    print(f"\n--- All {num_combinations} hyperparameter experiments completed ---")
    print(f"Total time taken: {total_duration_str} ({total_duration_seconds:.2f} seconds)")

    if best_hyperparams_combo_overall:
        print(f"Best validation accuracy (overall): {best_accuracy_overall:.4f}")
        print(f"Corresponding best hyperparameter combination (overall):\n{json.dumps(best_hyperparams_combo_overall, indent=2, ensure_ascii=False)}")
        
        summary_file_name = f"grid_search_summary_DynamicTime_2HL_{time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_file_path = os.path.join(OUTPUT_DIR_BASE, summary_file_name)
        
        serializable_best_hyperparams = {}
        for k, v_best in best_hyperparams_combo_overall.items():
            if isinstance(v_best, np.integer):
                serializable_best_hyperparams[k] = int(v_best)
            elif isinstance(v_best, np.floating):
                serializable_best_hyperparams[k] = float(v_best)
            else:
                serializable_best_hyperparams[k] = v_best
        
        serializable_all_results = []
        for res in all_results_summary:
            s_params_combo = {}
            for k_p, v_p in res['Hyperparameter Combination'].items():
                if isinstance(v_p, np.integer):
                    s_params_combo[k_p] = int(v_p)
                elif isinstance(v_p, np.floating):
                    s_params_combo[k_p] = float(v_p)
                else:
                    s_params_combo[k_p] = v_p
            serializable_all_results.append({**res, 'Hyperparameter Combination': s_params_combo})

        summary_data = {
            "Best Validation Accuracy (Overall)": best_accuracy_overall,
            "Best Hyperparameter Combination (Overall)": serializable_best_hyperparams,
            "All Run Results (sorted by accuracy)": sorted(serializable_all_results, key=lambda x: x['Final Validation Accuracy'], reverse=True),
            "Fixed Parameters (partial)": {k: v for k, v in fixed_params_dict.items()
                                       if k not in ['FEATURE_DIR', 'OUTPUT_DIR_BASE']},
            "Early Stopping Criterion": "Validation Accuracy",
            "Early Stopping Patience Epochs": EARLY_STOPPING_PATIENCE,
            "Early Stopping Min Delta": EARLY_STOPPING_MIN_DELTA,
            "Total Time Taken": total_duration_str,
            "Total Time Taken (seconds)": round(total_duration_seconds, 2)
        }
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"Summary report saved to: {summary_file_path}")
    else:
        print("No successful training runs or all runs resulted in zero accuracy.")