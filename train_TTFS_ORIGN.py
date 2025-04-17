import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from utils.load_dataset_deap_ORIGN import *
from module.TTFS_ORIGN import *

class Config:
    def __init__(self):
        # 数据相关
        self.data_dir = "C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN_code\\dataset"
        self.emotion_dim = 'valence'
        self.window_sec = 4
        self.batch_size = 32
        self.z_score_normalize = False
        # 模型相关
        self.input_size = 32 * 512       
        self.hidden_size = 256
        self.output_size = 2
        
        # 训练相关
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TTFS编码参数
        self.p = 0.0
        self.q = 1.0
        self.noise = 0.01

def get_label_type(emotion_dim):
    if emotion_dim == 'valence':
        return [0, 2]
    elif emotion_dim == 'arousal':
        return [1, 2]
    else:
        raise ValueError("emotion_dim must be 'valence' or 'arousal'")

def convert_to_ttfs(x):
    x_min = x.min()
    x_max = x.max()
    x_normalized = (x - x_min) / (x_max - x_min + 1e-8)  # 加入小值避免除零

    return 1 - x_normalized

    
 
def train_epoch(model, train_loader, optimizer, device, cfg):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    pbar = tqdm(train_loader, desc='Training')
    
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        # 展平数据: (batch_size, channels, time_steps) -> (batch_size, channels * time_steps)
        data = data.reshape(data.size(0), -1)
        data = convert_to_ttfs(data)
        
        optimizer.zero_grad()
        output = model(data)  
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

def validate(model, test_loader, device, cfg):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    pbar = tqdm(test_loader, desc='Validating')
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            # 展平数据
            data = data.reshape(data.size(0), -1)
            data = convert_to_ttfs(data)
            
            output = model(data)  # 直接前向传播
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
        use_z_score=cfg.z_score_normalize
    )

    # 构建模型
    model = SNNModel().to(cfg.device)
    model.layers.append(SpikingDense(cfg.hidden_size, cfg.input_size))
    model.layers.append(SpikingDense(cfg.output_size, cfg.hidden_size, outputLayer=True))
    model.to(cfg.device)
    
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