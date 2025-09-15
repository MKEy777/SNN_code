"""
SNN 教师模型训练与诊断分析集成脚本 (版本A基础 + 诊断模式)

功能:
1. 训练模式 (`--mode train`): 
   - 遵循版本A的训练逻辑（min更新, val_acc早停, gamma=10.0, 无类别平衡）。
   - 同时保存验证准确率最高的模型和验证损失最低的模型。

2. 诊断模式 (`--mode analyze`):
   - 加载并对比“最高准确率”和“最低损失”两个模型。
   - 分析并可视化内部脉冲动力学，以诊断模型行为。

使用方法:
- 训练: python your_script_name.py --mode train
- 诊断: python your_script_name.py --mode analyze
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Union, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json
import itertools
import copy
import argparse

# --- 1. SNN模型定义 (版本A) ---
EPSILON = 1e-9

def call_spiking_layer(tj: torch.Tensor, W: torch.Tensor, D_i: torch.Tensor,
                       t_min_prev: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor) -> torch.Tensor:
    current_device = tj.device
    W = W.to(dtype=torch.float32)
    D_i = D_i.to(dtype=torch.float32)
    t_min_prev = t_min_prev.to(dtype=torch.float32, device=current_device)
    t_min = t_min.to(dtype=torch.float32, device=current_device)
    t_max = t_max.to(dtype=torch.float32, device=current_device)
    
    threshold = t_max - t_min - D_i
    ti = torch.matmul(tj - t_min, W) + threshold + t_min
    ti = torch.where(ti < t_max, ti, t_max)
    return ti

class SpikingDense(nn.Module):
    def __init__(self, units: int, name: str, outputLayer: bool = False,
                 input_dim: Optional[int] = None,
                 kernel_initializer='glorot_uniform'):
        super(SpikingDense, self).__init__()
        self.units = units
        self.name = name
        self.outputLayer = outputLayer
        self.input_dim = input_dim
        self.initializer = kernel_initializer
        self.D_i = nn.Parameter(torch.zeros(units, dtype=torch.float32))
        self.kernel = None 

        self.register_buffer('t_min_prev', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('t_min', torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor([1.0], dtype=torch.float32))

        self.built = False
        if self.input_dim is not None:
            self.build(torch.Size([0, self.input_dim]))

    def _initialize_weights(self):
        if self.initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.kernel)
        else:
            nn.init.xavier_normal_(self.kernel)
        with torch.no_grad():
            self.D_i.zero_()

    def build(self, input_shape: torch.Size):
        if self.built: return
        in_dim = input_shape[-1]
        self.input_dim = in_dim
        device = self.D_i.device
        self.kernel = nn.Parameter(torch.empty(self.input_dim, self.units, device=device, dtype=torch.float32))
        self._initialize_weights()
        self.built = True

    def set_time_params(self, t_min_prev: torch.Tensor, t_min: torch.Tensor, t_max: torch.Tensor):
        self.t_min_prev.copy_(t_min_prev)
        self.t_min.copy_(t_min)
        self.t_max.copy_(t_max)

    def forward(self, tj: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.built:
            self.build(tj.shape)
        
        current_device = self.D_i.device
        tj = tj.to(current_device)
        min_ti_output = None
        
        if self.outputLayer:
            time_diff = self.t_min - self.t_min_prev
            safe_time_diff = torch.where(time_diff == 0, torch.tensor(EPSILON, device=current_device), time_diff)
            alpha = self.D_i / safe_time_diff
            output = alpha * time_diff + torch.matmul(self.t_min - tj, self.kernel)
        else:
            output = call_spiking_layer(tj, self.kernel, self.D_i, self.t_min_prev, self.t_min, self.t_max)
            with torch.no_grad():
                mask = torch.isfinite(output) & (output < self.t_max)
                spikes = output[mask]
                if spikes.numel() > 0:
                    min_ti_output = torch.min(spikes).detach().unsqueeze(0)
                else:
                    min_ti_output = self.t_max.clone().detach().unsqueeze(0)
        return output, min_ti_output

class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()
        self.layers_list = nn.ModuleList()

    def add(self, layer: nn.Module):
        self.layers_list.append(layer)

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        current_input = x
        min_ti_list: List[Optional[torch.Tensor]] = []
        hidden_outputs: List[torch.Tensor] = []

        target_device = x.device
        if self.layers_list:
            try:
                target_device = next(self.parameters()).device
            except StopIteration:
                pass 
        
        current_input = current_input.to(target_device)

        for i, layer in enumerate(self.layers_list):
            layer = layer.to(target_device)
            if isinstance(layer, SpikingDense):
                output_from_layer, min_ti = layer(current_input)
                if not layer.outputLayer:
                    min_ti_list.append(min_ti)
                    hidden_outputs.append(output_from_layer)
                current_input = output_from_layer
            else:
                current_input = layer(current_input)

        final_output = current_input
        
        return {
            "final_output": final_output,
            "min_ti_list": min_ti_list,
            "hidden_outputs": hidden_outputs
        }


HYPERPARAMETER_GRID = {
    'LEARNING_RATE': [2e-4],
    'BATCH_SIZE': [8],
    'HIDDEN_UNITS_1': [256],
    'HIDDEN_UNITS_2': [128],
    'LAMBDA_L1': [0],
    # 【新增】时间蒸馏损失的强度系数
    'LAMBDA_TIME_DISTILL': [0.001] 
}

# 【新增】根据诊断结果设定的目标平均脉冲时间
TARGET_SPIKE_TIMES = {
    'dense_1': 17.5,
    'dense_2': 55.0
}

PREPROCESSED_DATA_FILE = "selected_features_k300.pt"
OUTPUT_DIR_BASE = "SNN_Teacher_Training_TimeDistill" # 使用新目录
BEST_ACC_MODEL_SAVE_PATH = "teacher_snn_best_acc_td.pth"
BEST_LOSS_MODEL_SAVE_PATH = "teacher_snn_best_loss_td.pth"
OVERALL_SEARCH_REPORT_SAVE_PATH = "teacher_snn_hyper_search_summary_td.json"

T_MIN_INPUT = 0.0; T_MAX_INPUT = 1.0; TRAINING_GAMMA = 10.0; NUM_CLASSES = 3; NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20; EARLY_STOPPING_MIN_DELTA = 0.0001; RANDOM_SEED = 42; TEST_SPLIT_SIZE = 0.1

# --- 3. 工具函数 ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ttfs_encode(features: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
    features = np.clip(features, 0.0, 1.0)
    return torch.tensor(t_max - features * (t_max - t_min), dtype=torch.float32)

class EncodedEEGDataset(Dataset):
    def __init__(self, encoded_features: torch.Tensor, labels: np.ndarray):
        self.features = encoded_features.to(dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self) -> int:
        return len(self.labels)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

def generate_run_report(run_output_dir, run_id, params, history, best_val_acc, y_true, y_pred, stopped_epoch):
    os.makedirs(run_output_dir, exist_ok=True)
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.plot(epochs_range, history['train_loss'], label='Training Loss'); plt.plot(epochs_range, history['val_loss'], label='Validation Loss'); plt.title(f'Run {run_id} - Loss (Stopped at {stopped_epoch})'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 2); plt.plot(epochs_range, history['train_acc'], label='Training Accuracy'); plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy'); plt.title(f'Run {run_id} - Accuracy (Best: {best_val_acc:.4f})'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 3); plt.plot(epochs_range, history['lr'], label='Learning Rate'); plt.title(f'Run {run_id} - Learning Rate'); plt.legend(); plt.grid(True); plt.yscale('log')
    plt.tight_layout(); plt.savefig(os.path.join(run_output_dir, "training_history.png")); plt.close()

    with open(os.path.join(run_output_dir, "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"{'='*20} 训练报告: 运行 ID {run_id} {'='*20}\n\n")
        f.write("1. 训练参数:\n" + json.dumps(params, indent=4) + "\n\n" + "="*50 + "\n\n")
        f.write("2. 最终模型在验证集上的性能评估:\n\n")
        class_names = ['负向(0)', '中性(1)', '正向(2)']
        f.write("混淆矩阵:\n" + np.array2string(confusion_matrix(y_true, y_pred)) + "\n\n")
        f.write("分类报告:\n" + classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    print(f"  已为此运行生成详细报告和图表于: '{run_output_dir}'")

# --- 4. 训练与评估函数 (版本A) ---
def train_epoch_snn(model, dataloader, criterion, optimizer, device, 
                    gamma_ttfs, lambda_l1, lambda_time_distill, target_spike_times):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        result = model(features)
        outputs = result["final_output"]
        min_ti_list = result["min_ti_list"]
        hidden_outputs = result["hidden_outputs"]

        # 1. 计算主损失
        primary_loss = criterion(outputs, labels)
        
        # 2. 计算L1正则化
        l1_reg = lambda_l1 * sum(torch.sum(torch.abs(layer.kernel)) for layer in model.layers_list if isinstance(layer, SpikingDense) and layer.kernel is not None)

        # 3. 【核心新增】计算目标时间蒸馏损失 L_distill_time
        time_distill_loss = torch.tensor(0.0, device=device)
        hidden_layers = [layer for layer in model.layers_list if isinstance(layer, SpikingDense) and not layer.outputLayer]
        
        for i, h_out in enumerate(hidden_outputs):
            layer_name = hidden_layers[i].name
            if layer_name in target_spike_times:
                target_time = target_spike_times[layer_name]
                firing_mask = h_out < hidden_layers[i].t_max
                fired_spikes = h_out[firing_mask]
                
                if fired_spikes.numel() > 0:
                    # 计算与目标时间的MSE损失
                    layer_distill_loss = torch.mean(torch.square(fired_spikes - target_time))
                    time_distill_loss += layer_distill_loss
        
        # 4. 合并总损失
        loss = primary_loss + l1_reg + lambda_time_distill * time_distill_loss
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        with torch.no_grad():
            current_t_min_layer = torch.tensor(T_MAX_INPUT, dtype=torch.float32, device=device)
            t_min_prev_layer = torch.tensor(T_MIN_INPUT, dtype=torch.float32, device=device)
            k = 0
            for i, layer in enumerate(model.layers_list):
                if isinstance(layer, SpikingDense) and not layer.outputLayer:
                    min_ti_for_layer = min_ti_list[k] if k < len(min_ti_list) and min_ti_list[k] is not None else None
                    base_interval = torch.tensor(1.0, dtype=torch.float32, device=device)
                    min_ti_val = None
                    if min_ti_for_layer is not None and min_ti_for_layer.numel() > 0:
                        positive_spikes = min_ti_for_layer[min_ti_for_layer > 1e-9]
                        if positive_spikes.numel() > 0:
                            min_ti_val = torch.min(positive_spikes) # 版本A特性: 使用min    
                    if min_ti_val is not None and layer.t_max > min_ti_val:
                        dynamic_term = gamma_ttfs * (layer.t_max - min_ti_val)
                        new_t_max_layer = current_t_min_layer + torch.maximum(base_interval, torch.clamp(dynamic_term, min=0.0))
                    else:
                        new_t_max_layer = current_t_min_layer + base_interval
                    k += 1
                    
                    layer.set_time_params(t_min_prev_layer, current_t_min_layer, new_t_max_layer)
                    t_min_prev_layer = current_t_min_layer.clone()
                    current_t_min_layer = new_t_max_layer.clone()
                elif isinstance(layer, SpikingDense) and layer.outputLayer:
                    new_t_max_layer = current_t_min_layer + torch.tensor(1.0, dtype=torch.float32, device=device)
                    layer.set_time_params(t_min_prev_layer, current_t_min_layer, new_t_max_layer)

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    return running_loss / total_samples, correct_predictions / total_samples

def evaluate_model_snn(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            result = model(features)
            outputs = result["final_output"]
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return running_loss / total_samples, correct_predictions / total_samples, all_labels, all_preds

# --- 5. 训练会话与诊断函数 ---

def run_training_session(params, run_id, total_runs, device, full_data, base_output_dir):
    run_output_dir = os.path.join(base_output_dir, f"run_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"开始运行 [{run_id}/{total_runs}] | 参数: {params}")
    
    features_norm, labels = full_data
    INPUT_SIZE = features_norm.shape[1]
    
    X_train, X_val, y_train, y_val = train_test_split(
        features_norm.numpy(), labels.numpy(), test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=labels.numpy()
    )
    
    X_train_encoded = ttfs_encode(X_train, T_MIN_INPUT, T_MAX_INPUT)
    X_val_encoded = ttfs_encode(X_val, T_MIN_INPUT, T_MAX_INPUT)
    
    train_dataset = EncodedEEGDataset(X_train_encoded, y_train)
    val_dataset = EncodedEEGDataset(X_val_encoded, y_val)
    train_loader = DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['BATCH_SIZE'], shuffle=False)
    
    # 版本A特性: 外部add模式
    model = SNNModel()
    model.add(SpikingDense(units=params['HIDDEN_UNITS_1'], name='dense_1', input_dim=INPUT_SIZE))
    model.add(SpikingDense(units=params['HIDDEN_UNITS_2'], name='dense_2', input_dim=params['HIDDEN_UNITS_1']))
    model.add(SpikingDense(units=NUM_CLASSES, name='dense_output', input_dim=params['HIDDEN_UNITS_2'], outputLayer=True))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss() # 版本A特性: 无类别权重
    optimizer = optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0)

    best_val_acc, best_val_loss = 0.0, float('inf')
    best_acc_model_state, best_loss_model_state = None, None
    patience_counter = 0
    stopped_epoch = NUM_EPOCHS
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch_snn(model, train_loader, criterion, optimizer, device, TRAINING_GAMMA, params['LAMBDA_L1'], params['LAMBDA_TIME_DISTILL'], TARGET_SPIKE_TIMES)
        val_loss, val_acc, final_y_true, final_y_pred = evaluate_model_snn(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']; history['train_loss'].append(train_loss); history['train_acc'].append(train_acc); history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['lr'].append(current_lr)
        print(f"  轮次 [{epoch+1:>3}/{NUM_EPOCHS}] | LR: {current_lr:.2e} | 训练损失: {train_loss:.4f}, Acc: {train_acc:.4f} | 验证损失: {val_loss:.4f}, Acc: {val_acc:.4f}")
        scheduler.step()

        if val_acc > best_val_acc + EARLY_STOPPING_MIN_DELTA:
            best_val_acc = val_acc; best_acc_model_state = copy.deepcopy(model.state_dict()); patience_counter = 0
        else:
            patience_counter += 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_loss_model_state = copy.deepcopy(model.state_dict())
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  在第 {epoch+1} 轮触发早停。"); stopped_epoch = epoch + 1; break
            
    if best_acc_model_state:
        model.load_state_dict(best_acc_model_state)
        _, _, final_y_true, final_y_pred = evaluate_model_snn(model, val_loader, criterion, device)
        generate_run_report(run_output_dir, run_id, params, history, best_val_acc, final_y_true, final_y_pred, stopped_epoch)

    # 确保即使训练提前结束，我们也有一个best_loss_model
    if best_loss_model_state is None and best_acc_model_state is not None:
        best_loss_model_state = best_acc_model_state
        
    return best_val_acc, best_acc_model_state, best_loss_model_state

# --- 诊断函数 (新增) ---
def analyze_snn_dynamics(model, dataloader, device):
    model.eval(); model.to(device)
    hidden_layers = [layer for layer in model.layers_list if isinstance(layer, SpikingDense) and not layer.outputLayer]
    num_hidden_layers = len(hidden_layers)
    if num_hidden_layers == 0: return {}
    
    total_neurons_per_layer = [layer.units for layer in hidden_layers]
    fired_counts_per_layer = [0] * num_hidden_layers
    spike_times_per_layer = [[] for _ in range(num_hidden_layers)]
    total_samples = 0

    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device); total_samples += features.shape[0]
            result = model(features)
            hidden_outputs = result["hidden_outputs"]
            for i, hidden_output in enumerate(hidden_outputs):
                layer = hidden_layers[i]
                firing_mask = hidden_output < layer.t_max
                fired_counts_per_layer[i] += torch.sum(firing_mask).item()
                fired_spikes = hidden_output[firing_mask]
                if fired_spikes.numel() > 0:
                    spike_times_per_layer[i].extend(fired_spikes.cpu().numpy().flatten())

    analysis_results = {}
    for i in range(num_hidden_layers):
        layer_name = hidden_layers[i].name
        avg_firing_rate = fired_counts_per_layer[i] / (total_samples * total_neurons_per_layer[i]) if total_samples > 0 else 0
        spike_times = np.array(spike_times_per_layer[i])
        analysis_results[layer_name] = {
            'avg_firing_rate': avg_firing_rate, 'spike_times_raw': spike_times,
            'mean_spike_time': np.mean(spike_times) if len(spike_times) > 0 else float('nan'),
        }
    return analysis_results

def plot_dynamics_comparison(stats1, stats2, model1_name, model2_name, layer_names):
    num_layers = len(layer_names)
    fig, axes = plt.subplots(num_layers, 2, figsize=(16, 5 * num_layers), squeeze=False)
    for i, name in enumerate(layer_names):
        rates = [stats1[name]['avg_firing_rate'] * 100, stats2[name]['avg_firing_rate'] * 100]
        sns.barplot(x=[model1_name, model2_name], y=rates, ax=axes[i, 0])
        axes[i, 0].set_title(f'{name} - Average Firing Rate (%)'); axes[i, 0].set_ylabel('Firing Rate (%)')
        
        sns.histplot(data=stats1[name]['spike_times_raw'], ax=axes[i, 1], color='blue', label=model1_name, kde=True, stat='density', alpha=0.6)
        sns.histplot(data=stats2[name]['spike_times_raw'], ax=axes[i, 1], color='red', label=model2_name, kde=True, stat='density', alpha=0.6)
        axes[i, 1].set_title(f'{name} - Spike Time Distribution'); axes[i, 1].set_xlabel('Spike Time'); axes[i, 1].legend()

    # 修改后的代码
    plt.tight_layout(pad=3.0)
    save_path = 'dynamics_comparison.png'
    plt.savefig(save_path)
    print(f"\n诊断图表已成功保存至: '{save_path}'")
    plt.close() # 加上这行是个好习惯，可以释放内存


# --- 6. 主执行逻辑 (新增诊断模式) ---
def main():
    parser = argparse.ArgumentParser(description="SNN 教师模型训练与诊断分析集成脚本 (版本A)")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'analyze'], help="选择模式: 'train' 或 'analyze'")
    args = parser.parse_args()

    set_seed(RANDOM_SEED); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"使用的设备: {device}")

    if args.mode == 'train':
        print("\n--- 模式: 训练 (版本A逻辑) ---")
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
        data = torch.load(PREPROCESSED_DATA_FILE)
        features, labels = data['features_normalized'], data['labels']
        
        keys, values = zip(*HYPERPARAMETER_GRID.items())
        hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_overall_acc, best_overall_params = 0.0, None
        best_acc_state_overall, best_loss_state_overall = None, None

        for i, params in enumerate(hyperparam_combinations):
            val_acc, best_acc_state, best_loss_state = run_training_session(params, i + 1, len(hyperparam_combinations), device, (features, labels), OUTPUT_DIR_BASE)
            if val_acc > best_overall_acc:
                best_overall_acc = val_acc; best_overall_params = params
                best_acc_state_overall = best_acc_state
                best_loss_state_overall = best_loss_state
                
        print("\n" + "="*60 + "\n--- 超参数搜索完成 ---")

        if best_overall_params:
            print(f"全局最佳验证准确率: {best_overall_acc:.4f}"); print("最佳超参数组合:\n" + json.dumps(best_overall_params, indent=4))
            
            torch.save(best_acc_state_overall, BEST_ACC_MODEL_SAVE_PATH)
            print(f"\n已将最高准确率模型权重保存至: '{BEST_ACC_MODEL_SAVE_PATH}'")
            torch.save(best_loss_state_overall, BEST_LOSS_MODEL_SAVE_PATH)
            print(f"已将最低损失模型权重保存至: '{BEST_LOSS_MODEL_SAVE_PATH}'")

            report = {'best_validation_accuracy': best_overall_acc, 'best_hyperparameters': best_overall_params, 'search_space': HYPERPARAMETER_GRID}
            with open(OVERALL_SEARCH_REPORT_SAVE_PATH, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"超参数搜索总摘要报告已保存至: '{OVERALL_SEARCH_REPORT_SAVE_PATH}'")
        else:
            print("所有训练运行均未产生有效结果。")

    elif args.mode == 'analyze':
        print("\n--- 模式: 诊断 ---")
        if not all(os.path.exists(p) for p in [BEST_ACC_MODEL_SAVE_PATH, BEST_LOSS_MODEL_SAVE_PATH, OVERALL_SEARCH_REPORT_SAVE_PATH]):
            print("错误: 找不到必要的模型或配置文佳。请先以 'train' 模式运行脚本。")
            return

        with open(OVERALL_SEARCH_REPORT_SAVE_PATH, 'r') as f:
            best_params = json.load(f)['best_hyperparameters']
        print("已加载超参数用于构建模型结构:", best_params)

        data = torch.load(PREPROCESSED_DATA_FILE)
        features, labels = data['features_normalized'], data['labels']
        INPUT_SIZE = features.shape[1]
        
        _, X_val, _, y_val = train_test_split(features.numpy(), labels.numpy(), test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=labels.numpy())
        val_loader = DataLoader(EncodedEEGDataset(ttfs_encode(X_val, T_MIN_INPUT, T_MAX_INPUT), y_val), batch_size=best_params['BATCH_SIZE'])
        
        model_acc = SNNModel(); model_acc.add(SpikingDense(best_params['HIDDEN_UNITS_1'], 'd1', input_dim=INPUT_SIZE)); model_acc.add(SpikingDense(best_params['HIDDEN_UNITS_2'], 'd2', input_dim=best_params['HIDDEN_UNITS_1'])); model_acc.add(SpikingDense(NUM_CLASSES, 'out', input_dim=best_params['HIDDEN_UNITS_2'], outputLayer=True))
        model_acc.load_state_dict(torch.load(BEST_ACC_MODEL_SAVE_PATH, map_location=device))
        
        model_loss = SNNModel(); model_loss.add(SpikingDense(best_params['HIDDEN_UNITS_1'], 'd1', input_dim=INPUT_SIZE)); model_loss.add(SpikingDense(best_params['HIDDEN_UNITS_2'], 'd2', input_dim=best_params['HIDDEN_UNITS_1'])); model_loss.add(SpikingDense(NUM_CLASSES, 'out', input_dim=best_params['HIDDEN_UNITS_2'], outputLayer=True))
        model_loss.load_state_dict(torch.load(BEST_LOSS_MODEL_SAVE_PATH, map_location=device))
        
        print("正在分析 '最高准确率模型'...")
        stats_acc = analyze_snn_dynamics(model_acc, val_loader, device)
        print("正在分析 '最低损失模型'...")
        stats_loss = analyze_snn_dynamics(model_loss, val_loader, device)

        if stats_acc and stats_loss:
            layer_names = list(stats_acc.keys())
            plot_dynamics_comparison(stats_acc, stats_loss, "Best Acc Model", "Best Loss Model", layer_names)
        else:
            print("未能提取隐藏层统计数据，无法生成图表。")


if __name__ == '__main__':
    main()