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

# --- Fixed Parameters ---
FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features_NoBandpass_Fixed_BaselineCorrected"
OUTPUT_DIR_BASE = r"paper_result"
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

# --- Hyperparameter Search Space ---
hyperparameter_grid = {
    'LEARNING_RATE': [5e-4],
    'BATCH_SIZE': [8],
    'HIDDEN_UNITS_1': [512],
    'HIDDEN_UNITS_2': [256],
}

# Load feature data
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
        raise ValueError(f"No feature files found in directory {feature_dir}.")
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    label_mapping = {-1: 0, 0: 1, 1: 2}
    valid_labels_indices = np.isin(combined_labels, list(label_mapping.keys()))
    combined_features_filtered = combined_features[valid_labels_indices]
    combined_labels_filtered = combined_labels[valid_labels_indices]
    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels_filtered], dtype=np.int64)
    return combined_features_filtered, mapped_labels

# TTFS encoding
def ttfs_encode(features: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
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
            input_dim_for_layer = m.kernel.shape[0]
            if input_dim_for_layer > 0:
                stddev = 1.0 / np.sqrt(input_dim_for_layer)
                with torch.no_grad():
                    m.kernel.data.normal_(mean=0.0, std=stddev)

# Train one epoch
def train_epoch(model: SNNModel, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device, epoch: int,
                gamma_ttfs: float, current_t_min_input: float, current_t_max_input: float) -> Tuple[float, float]:
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

        # Update time parameters as per the paper
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
                        if min_ti_for_layer is not None:
                            # Paper's update rule: t_max += max(gamma * (t_max - min_ti) - (t_max - t_min), 0)
                            current_t_max = layer.t_max.clone().detach()
                            current_t_min = layer.t_min.clone().detach()
                            delta_t_max = gamma_ttfs * (current_t_max - min_ti_for_layer) - (current_t_max - current_t_min)
                            delta_t_max = torch.clamp(delta_t_max, min=0.0)
                            new_t_max_layer = current_t_max + delta_t_max
                        else:
                            new_t_max_layer = layer.t_max.clone().detach()
                        k += 1
                    else:  # Output layer
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

# Build filename prefix
def build_filename_prefix(params: Dict[str, Union[int, float]]) -> str:
    lr_val = params.get('LEARNING_RATE', fixed_parameters.get('LEARNING_RATE'))
    lr_str = f"{lr_val:.0e}".replace('-', 'm').replace('+', '')
    gamma_ttfs_val = params.get('TRAINING_GAMMA', fixed_parameters.get('TRAINING_GAMMA'))
    gamma_str = str(gamma_ttfs_val).replace('.', 'p')
    lr_decay_gamma_val = params.get('LR_SCHEDULER_GAMMA', fixed_parameters.get('LR_SCHEDULER_GAMMA', 'NA'))
    lr_decay_gamma_str = f"_lrdecay{str(lr_decay_gamma_val).replace('.', 'p')}" if lr_decay_gamma_val != 'NA' else ""
    prefix = (f"lr{lr_str}_bs{params['BATCH_SIZE']}_epochsMax{params['NUM_EPOCHS']}"
              f"_h1_{params['HIDDEN_UNITS_1']}_h2_{params['HIDDEN_UNITS_2']}"
              f"_gammaTTFS{gamma_str}{lr_decay_gamma_str}_seed{params['RANDOM_SEED']}")
    return prefix

# Plot training history
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
    print(f"Training history and learning rate plot saved as {filename}")

# Save model
def save_model_torch(model: SNNModel, filename_prefix: str, save_dir: str):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_{filename_prefix}_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to: {save_path}")

# Save parameters
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
    with open(save_path, 'w') as f: json.dump(serializable_params, f, indent=4)
    print(f"Training parameters saved to: {save_path}")

# Training and evaluation function
def run_training_session(current_hyperparams: Dict, fixed_params_dict: Dict, run_id: int):
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

    all_params_for_naming_and_saving = all_params.copy()
    all_params_for_naming_and_saving['LR_SCHEDULER_GAMMA'] = _LR_SCHEDULER_GAMMA
    all_params_for_naming_and_saving['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))'

    base_prefix = build_filename_prefix(all_params_for_naming_and_saving)
    run_specific_output_dir_name = f"{base_prefix}_val_loss_stop_customInit"
    run_specific_output_dir = os.path.join(OUTPUT_DIR_BASE, run_specific_output_dir_name)

    if not os.path.exists(run_specific_output_dir):
        os.makedirs(run_specific_output_dir, exist_ok=True)

    filename_prefix = f"final_best_val_loss_early_stop_customInit"

    print(f"\n--- Starting Run ID: {run_id} ---")
    print(f"Full parameters (for this run): {all_params_for_naming_and_saving}")
    print(f"Initial learning rate: {LEARNING_RATE_INITIAL}, Learning rate decay Gamma: {_LR_SCHEDULER_GAMMA}")
    print(f"Weight initialization: N(0, 1/sqrt(N_in))")
    print(f"Early stopping settings (based on validation loss): Patience={_EARLY_STOPPING_PATIENCE}, Min Delta={_EARLY_STOPPING_MIN_DELTA}")
    print(f"Results will be saved in: {run_specific_output_dir}")

    save_params(all_params_for_naming_and_saving, filename_prefix, run_specific_output_dir)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    features_data, labels_data = load_features_from_mat(FEATURE_DIR)
    print(f"Data loaded successfully. Feature shape: {features_data.shape}, Label shape: {labels_data.shape}")

    unique_labels_data, counts_data = np.unique(labels_data, return_counts=True)
    stratify_option = labels_data if all(count >= 2 for count in counts_data[counts_data > 0]) and len(unique_labels_data[counts_data > 0]) >= 2 else None
    if stratify_option is None: print("Warning: Dataset does not meet stratified sampling conditions. Using random sampling.")
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(
        features_data, labels_data, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=stratify_option
    )
    print(f"Training set size: {len(X_train_orig)}")
    print(f"Validation set size: {len(X_val_orig)}")

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

    print("Applying custom weight initialization N(0, 1/sqrt(N_in))...")
    model.apply(custom_weight_init)
    print("Custom weight initialization completed.")

    model.to(device)
    print(f"Total trainable parameters in model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL)
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

    print(f"Starting training for up to {_MAX_NUM_EPOCHS} epochs...")
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

        scheduler.step()

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{_MAX_NUM_EPOCHS}] | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Time: {epoch_end_time - epoch_start_time:.2f} sec")

        if val_loss < best_val_loss - _EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"  Validation loss improved to {best_val_loss:.4f}. Resetting patience counter.")
        else:
            patience_counter += 1
            print(f"  Validation loss did not improve significantly. Patience counter: {patience_counter}/{_EARLY_STOPPING_PATIENCE}")

        if patience_counter >= _EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. Max patience {_EARLY_STOPPING_PATIENCE} reached (based on validation loss).")
            stopped_epoch = epoch + 1
            break

    if stopped_epoch is None:
        stopped_epoch = _MAX_NUM_EPOCHS
        print(f"Training completed for {_MAX_NUM_EPOCHS} epochs.")
        if best_model_state_dict is None or (val_losses and val_losses[-1] < best_val_loss - _EARLY_STOPPING_MIN_DELTA):
             if val_losses:
                best_val_loss = val_losses[-1]
                best_model_state_dict = copy.deepcopy(model.state_dict())
                print(f"  Training ended, recording model state from last epoch, validation loss: {best_val_loss:.4f}")

    if best_model_state_dict is not None:
        print("Loading model weights with best performance on validation set (based on loss).")
        model.load_state_dict(best_model_state_dict)
    else:
        print("Warning: Best model state not found (based on validation loss). Using the last epoch's model.")

    end_time_run = time.time()
    print(f"--- SNN Training Completed (Run ID: {run_id}), Total Time: {end_time_run - start_time_run:.2f} sec, Actual Epochs: {stopped_epoch} ---")

    save_model_torch(model, filename_prefix, run_specific_output_dir)
    plot_history(train_losses, val_losses, train_accuracies, val_accuracies, learning_rates_over_epochs,
                 filename_prefix, run_specific_output_dir, stopped_epoch=stopped_epoch)

    final_val_loss, final_val_acc, final_labels, final_preds = evaluate_model(model, val_loader, criterion, device)
    print(f"Final validation loss (from best model based on loss): {final_val_loss:.4f}")
    print(f"Final validation accuracy (from best model based on loss): {final_val_acc:.4f}")

    report_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
    report = classification_report(final_labels, final_preds, target_names=report_names, digits=4, zero_division=0)
    print("\nClassification Report (based on validation set and best model based on loss):")
    print(report)

    report_filename = os.path.join(run_specific_output_dir, f"classification_report_{filename_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_filename, 'w') as f:
        params_to_save_report = {**all_params_for_naming_and_saving, 'stopped_epoch': stopped_epoch, 'best_validation_accuracy_achieved': final_val_acc, 'weight_initialization_note': 'N(0, 1/sqrt(N_in)) applied after __init__'}
        f.write(f"Hyperparameters: {json.dumps(params_to_save_report, indent=4)}\n\n")
        f.write(f"Final Validation Loss (best model on val_loss): {final_val_loss:.4f}\n")
        f.write(f"Final Validation Accuracy (best model on val_loss): {final_val_acc:.4f}\n\n")
        f.write("Classification Report (on validation set with best model based on val_loss):\n")
        f.write(report)
    print(f"Classification report saved to: {report_filename}")
    print(f"--- Run ID: {run_id} Completed ---")
    return final_val_acc

# Main program
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
        'LR_SCHEDULER_GAMMA': LR_SCHEDULER_GAMMA
    }

    keys, values = zip(*hyperparameter_grid.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    num_combinations = len(hyperparam_combinations)
    print(f"Will perform training for {num_combinations} hyperparameter combinations.")
    print(f"Fixed parameters (partial): TRAINING_GAMMA (TTFS) = {TRAINING_GAMMA}, MAX_NUM_EPOCHS = {NUM_EPOCHS}, LR_SCHEDULER_GAMMA = {LR_SCHEDULER_GAMMA}")
    print(f"Weight initialization: N(0, 1/sqrt(N_in))")
    print(f"Early stopping parameters (based on validation loss): Patience = {EARLY_STOPPING_PATIENCE}, Min Delta = {EARLY_STOPPING_MIN_DELTA}")

    best_accuracy_overall = 0.0
    best_hyperparams_combo_overall = None
    all_results = []

    for i, params_combo_iter in enumerate(hyperparam_combinations):
        validation_accuracy_for_run = run_training_session(current_hyperparams=params_combo_iter,
                                                           fixed_params_dict=fixed_parameters,
                                                           run_id=i+1)

        full_params_for_log = {**fixed_parameters, **params_combo_iter}
        full_params_for_log['WEIGHT_INITIALIZATION'] = 'N(0, 1/sqrt(N_in))'

        all_results.append({'id': i+1, 'params': full_params_for_log, 'best_validation_accuracy_for_this_run (from best_loss_model)': validation_accuracy_for_run})

        if validation_accuracy_for_run > best_accuracy_overall:
            best_accuracy_overall = validation_accuracy_for_run
            best_hyperparams_combo_overall = full_params_for_log

    print("\n--- All Hyperparameter Trials Completed ---")
    if best_hyperparams_combo_overall:
        print(f"Best validation accuracy across all runs (from model chosen by loss-based early stopping): {best_accuracy_overall:.4f}")
        print(f"Corresponding best hyperparameter combination: {json.dumps(best_hyperparams_combo_overall, indent=2)}")

        summary_file_suffix = f"grid_search_summary_val_loss_early_stop_no_regularization_customInit.json"
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
        print(f"Summary of all runs saved to: {summary_file}")
    else:
        print("No successful training runs or all runs resulted in zero accuracy.")

    print("Script execution completed.")