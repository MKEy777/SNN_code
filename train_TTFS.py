# train_snn.py
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

# 从单独的文件导入 SNN 模型定义
from TTFS_ORIGN import SNNModel, SpikingDense

# --- Configuration ---
FEATURE_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features"
if not os.path.isdir(FEATURE_DIR):
    print(f"ERROR: Feature directory not found: {FEATURE_DIR}")
    exit()

# Model Hyperparameters
INPUT_SIZE = 4216
HIDDEN_UNITS_1 = 512
HIDDEN_UNITS_2 = 128
OUTPUT_SIZE = 3
X_N_HIDDEN = 1.0
X_N_OUTPUT = 1.0
T_MIN_INPUT = 0.0
T_MAX_INPUT = 1.0

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 50
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

# --- Data Loading and Preparation ---

def load_features_from_mat(feature_dir):
    """Loads features and labels, converting features to float32."""
    all_features = []
    all_labels = []
    mat_files = glob.glob(os.path.join(feature_dir, "*_features.mat"))
    if not mat_files:
        print(f"ERROR: No '*_features.mat' files found in {feature_dir}")
        exit()
    print(f"Found {len(mat_files)} feature files. Loading...")
    for fpath in mat_files:
        try:
            mat_data = loadmat(fpath)
            if 'features' in mat_data and 'labels' in mat_data:
                features = mat_data['features']
                labels = mat_data['labels'].flatten()
                if features.shape[1] != INPUT_SIZE: continue
                if features.shape[0] != len(labels): continue
                # 使用 float32
                all_features.append(features.astype(np.float32))
                all_labels.append(labels)
        except Exception as e: print(f"Error loading {os.path.basename(fpath)}: {e}. Skip.")
    if not all_features:
        print("ERROR: No valid data loaded.")
        exit()
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)
    print(f"Loaded total {combined_features.shape[0]} segments.")
    label_mapping = {-1: 0, 0: 1, 1: 2}
    mapped_labels = np.array([label_mapping[lbl] for lbl in combined_labels], dtype=np.int64) # Labels remain int64
    print("Label mapping applied.")
    print(f"Unique mapped labels: {np.unique(mapped_labels)}")
    return combined_features, mapped_labels

def ttfs_encode(features, t_min=T_MIN_INPUT, t_max=T_MAX_INPUT):
    """Encodes normalized features [0, 1] into spike times using TTFS (output float32)."""
    features = np.clip(features, 0.0, 1.0)
    spike_times = t_max - features * (t_max - t_min)
    # 输出 float32 张量
    return torch.tensor(spike_times, dtype=torch.float32)

class EncodedEEGDataset(Dataset):
    """Dataset returning TTFS encoded features (float32)."""
    def __init__(self, encoded_features, labels):
        # encoded_features should be a float32 tensor
        self.features = encoded_features
        self.labels = torch.tensor(labels, dtype=torch.long) # Labels remain LongTensor
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

# --- Training and Evaluation Functions ---
# (No change needed here as they operate on tensors whose dtype is determined by the model/data)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the SNN model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for features, labels in dataloader:
        # Ensure data sent to device maintains intended dtype (float32 for features)
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(features) # Model expects float32, outputs float32
        loss = criterion(outputs, labels) # Criterion handles input dtype
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    """Evaluates the SNN model."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs, _ = model(features) # Model expects float32, outputs float32
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

# --- Main Execution ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data (Now loads as float32)
    features, labels = load_features_from_mat(FEATURE_DIR)

    # 2. Split Data
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"Training set size: {len(X_train_orig)}")
    print(f"Testing set size: {len(X_test_orig)}")

    # 3. Normalize Features (Input/Output is float32)
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train_orig)
    X_test_norm = scaler.transform(X_test_orig)
    print("Features normalized.")

    # 4. Encode Normalized Features using TTFS (Output is float32)
    X_train_encoded = ttfs_encode(X_train_norm)
    X_test_encoded = ttfs_encode(X_test_norm)
    print("Normalized features encoded into spike times.")

    # 5. Create Datasets and DataLoaders (Handles float32 features)
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    test_dataset = EncodedEEGDataset(X_test_encoded, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 6. Initialize SNN Model, Loss, Optimizer
    # Model will be initialized with float32 parameters from the definitions file
    # Explicitly set model to float32 just in case, though parameters should already be float32
    model = SNNModel().to(device).to(torch.float32)

    # Add layers (layers themselves are initialized with float32 now)
    model.add(SpikingDense(units=HIDDEN_UNITS_1, name='dense1', input_dim=INPUT_SIZE, X_n=X_N_HIDDEN, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=HIDDEN_UNITS_2, name='dense2', X_n=X_N_HIDDEN, kernel_initializer='glorot_uniform'))
    model.add(SpikingDense(units=OUTPUT_SIZE, name='output', outputLayer=True, X_n=X_N_OUTPUT, kernel_initializer='glorot_uniform'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting SNN Training (float32) ---")
    start_time = time.time()
    best_test_acc = 0.0

    # 7. Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        if test_acc > best_test_acc:
             best_test_acc = test_acc
             # torch.save(model.state_dict(), 'best_snn_model_float32.pth')

    end_time = time.time()
    print(f"\n--- SNN Training Finished ---")
    print(f"Time: {end_time - start_time:.2f}s | Best Test Acc: {best_test_acc:.4f}")

    # 8. Final Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    final_test_loss, final_test_acc, final_labels, final_preds = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Acc: {final_test_acc:.4f}")
    report_target_names = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    try:
        print("\nClassification Report:")
        print(classification_report(final_labels, final_preds, target_names=report_target_names, digits=4, zero_division=0))
    except ValueError as e: print(f"\nCould not generate report: {e}")

    print("\nScript finished.")