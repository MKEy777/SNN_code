import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from dataset_prepare import dataset_prepare
from model import SNNModel, SpikingDense

# 超参数设置
class Config:
    def __init__(self):
        # 数据相关
        self.data_dir = "dataset/"  # 数据集路径
        self.emotion_dim = 'valence'  # 'valence' 或 'arousal'
        self.window_sec = 4  
        self.batch_size = 32
        
        # 模型相关
        self.input_size = 32     # EEG通道数
        self.hidden_size = 256   # 隐藏层神经元数
        self.output_size = 2     # 分类数
        self.time_steps = 512    # 时间步长
        
        # 训练相关
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_time_sequence(model, x, device):
    """
    按时间步处理序列数据
    x: [batch_size, input_size, time_steps]
    """
    batch_size = x.shape[0]
    time_steps = x.shape[2]
    
    # 初始化隐藏状态
    hidden_states = torch.zeros(batch_size, model.layers[0].units).to(device)
    
    # 逐时间步处理
    for t in range(time_steps):
        # 获取当前时间步的输入 [batch_size, input_size]
        current_input = x[:, :, t]
        # 更新隐藏状态
        hidden_states = model(current_input)
    
    return hidden_states

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc='Training')
    for data, target in pbar:
        # data shape: [batch_size, input_size, time_steps]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 按时间步处理序列
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

def validate(model, test_loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(test_loader, desc='Validating')
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            # 按时间步处理序列
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

def main():
    cfg = Config()
    print(f"使用设备: {cfg.device}")
    print(f"训练情绪维度: {cfg.emotion_dim}")

    # 准备数据
    label_type = [0, 2] if cfg.emotion_dim == 'valence' else [1, 2]
    try:
        train_loader, test_loader = dataset_prepare(
            window_length_sec=cfg.window_sec,
            n_subjects=26,
            single_subject=False,
            load_all=True,
            only_EEG=True,
            label_type=label_type,
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            normalize=True
        )
        print("数据加载成功!")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # 构建模型 (32-256-2)
    try:
        model = SNNModel().to(cfg.device)
        model.layers.append(SpikingDense(cfg.hidden_size, cfg.input_size))  # 32 -> 256
        model.layers.append(SpikingDense(cfg.output_size, cfg.hidden_size, outputLayer=True))  # 256 -> 2
        print("模型构建成功!")
        
        # 打印模型结构
        print("\n模型结构:")
        print(f"Input Layer: {cfg.input_size} neurons")
        print(f"Hidden Layer: {cfg.hidden_size} neurons")
        print(f"Output Layer: {cfg.output_size} neurons")
        print(f"Time steps: {cfg.time_steps}\n")
        
    except Exception as e:
        print(f"模型构建失败: {str(e)}")
        return

    # 优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # 训练循环
    best_val_acc = 0
    print("\n开始训练...")
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, cfg.device
        )
        
        val_loss, val_acc, val_f1 = validate(
            model, test_loader, cfg.device
        )
        
        # 打印训练信息
        print(f'\n训练指标 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}')
        print(f'验证指标 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"新的最佳验证准确率: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
