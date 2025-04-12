import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from utils.load_dataset_deap_ORIGN import *
from module.TTFS_ORIGN import *

# 配置类，管理超参数和设置
class Config:
    def __init__(self):
        # 数据相关
        self.data_dir = "C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN_code\\dataset"   # DEAP数据集路径
        self.emotion_dim = 'valence'     # 情感维度: 'valence' 或 'arousal'
        self.window_sec = 4              # 时间窗口长度（秒）
        self.batch_size = 32             # 批次大小
        self.z_score_normalize = True    # 是否进行Z-score标准化
        
        # 模型相关
        self.input_size = 32             # EEG通道数
        self.hidden_size = 256           # 隐藏层神经元数
        self.output_size = 2             # 输出类别数（高/低）
        self.time_steps = 512            # 时间步长（采样率128Hz * 4秒）
        
        # 训练相关
        self.epochs = 100                # 训练轮数
        self.learning_rate = 0.001       # 学习率
        self.weight_decay = 1e-4         # 权重衰减
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TTFS编码参数
        self.p = 0.0                     # 数据最小值
        self.q = 1.0                     # 数据最大值
        self.noise = 0.01                # 噪声标准差

# 根据情感维度设置标签类型
def get_label_type(emotion_dim):
    if emotion_dim == 'valence':
        return [0, 2]  # valence, 2类
    elif emotion_dim == 'arousal':
        return [1, 2]  # arousal, 2类
    else:
        raise ValueError("emotion_dim must be 'valence' or 'arousal'")

# TTFS 编码函数，将 EEG 数据转换为脉冲时间
def convert_to_ttfs(x, p, q, noise):
    x = (x - p) / (q - p)           # 归一化到 [0, 1]
    x = 1 - x                       # 反转值
    x = torch.clamp(x + torch.randn_like(x) * noise, min=0)  # 添加噪声并限制范围
    return x

# 处理时间序列的前向传播
def process_time_sequence(model, x, device):
    batch_size = x.shape[0]
    time_steps = x.shape[2]
    hidden_states = torch.zeros(batch_size, model.layers[0].units).to(device)
    for t in range(time_steps):
        current_input = x[:, :, t]
        hidden_states = model(current_input)
    return hidden_states

# 训练一个 epoch
def train_epoch(model, train_loader, optimizer, device, cfg):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    pbar = tqdm(train_loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        data = convert_to_ttfs(data, cfg.p, cfg.q, cfg.noise)  # TTFS 编码
        optimizer.zero_grad()
        output = process_time_sequence(model, data, device)
        loss = model.loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1).cpu().numpy()
        all_predictions.extend(pred)
        all_targets.extend(target.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = accuracy_score(all_targets, all_predictions)
    epoch_f1 = f1_score(all_targets, all_predictions, average='macro')
    return epoch_loss, epoch_acc, epoch_f1

# 验证函数
def validate(model, test_loader, device, cfg):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    pbar = tqdm(test_loader, desc='Validating')
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            data = convert_to_ttfs(data, cfg.p, cfg.q, cfg.noise)  # TTFS 编码
            output = process_time_sequence(model, data, device)
            loss = model.loss_fn(output, target).item()
            total_loss += loss
            pred = output.argmax(dim=1).cpu().numpy()
            all_predictions.extend(pred)
            all_targets.extend(target.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss:.4f}'})
    epoch_loss = total_loss / len(test_loader)
    epoch_acc = accuracy_score(all_targets, all_predictions)
    epoch_f1 = f1_score(all_targets, all_predictions, average='macro')
    return epoch_loss, epoch_acc, epoch_f1

# 主函数
def main():
    cfg = Config()
    print(f"使用设备: {cfg.device}")
    print(f"训练情绪维度: {cfg.emotion_dim}")

    # 加载数据
    label_type = get_label_type(cfg.emotion_dim)
    train_loader, test_loader = dataset_prepare(
        window_length_sec=cfg.window_sec,
        n_subjects=26,
        single_subject=False,
        load_all=True,
        only_EEG=True,
        label_type=label_type,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        z_score_normalize=cfg.z_score_normalize
    )

    # 构建 SNN 模型
    model = SNNModel().to(cfg.device)
    model.layers.append(SpikingDense(cfg.hidden_size, cfg.input_size))          # 输入层 -> 隐藏层
    model.layers.append(SpikingDense(cfg.output_size, cfg.hidden_size, outputLayer=True))  # 隐藏层 -> 输出层

    # 设置优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # 训练循环
    best_val_acc = 0
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, cfg.device, cfg)
        val_loss, val_acc, val_f1 = validate(model, test_loader, cfg.device, cfg)
        print(f'\n训练指标 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'验证指标 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"新的最佳验证准确率: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
