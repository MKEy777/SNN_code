import torch
import torch.nn as nn
import torch.optim as optim
from utils.load_dataset_deap import dataset_prepare
from module.TTFS import SNNTTFSLayer
import numpy as np
from tqdm import tqdm

class SNNTTFS(nn.Module):
    def __init__(self, time_steps=512):
        super().__init__()
        self.time_steps = time_steps
        
        # Network layers: 32-200-2
        self.layer1 = SNNTTFSLayer(32, 200)
        self.layer2 = SNNTTFSLayer(200, 2, is_output_layer=True)
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize neuron states
        self.layer1.set_neuron_state(batch_size, device)
        self.layer2.set_neuron_state(batch_size, device)
        
        # Store output membrane potentials
        outputs = []
        
        # Process each time step
        for t in range(self.time_steps):
            current_input = x[:, t, :]  # (batch_size, 32)
            
            # Forward pass through layers
            hidden = self.layer1(current_input, t)
            output = self.layer2(hidden, t)
            outputs.append(output)
        
        # Stack outputs and get final prediction
        outputs = torch.stack(outputs, dim=1)  # (batch_size, time_steps, 2)
        final_output = outputs[:, -1, :]  # Use last timestep's output
        
        return final_output

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 创建 tqdm 进度条
    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        # 计算当前批次的 loss 和 accuracy
        batch_loss = loss.item()
        pred = output.argmax(dim=1)
        batch_correct = pred.eq(target).sum().item()
        batch_total = target.size(0)
        batch_acc = batch_correct / batch_total
        
        # 累积总计值
        total_loss += batch_loss
        correct += batch_correct
        total += batch_total
        
        # 更新 tqdm 进度条的描述，实时显示 loss 和 accuracy
        progress_bar.set_postfix({
            'batch_loss': f'{batch_loss:.4f}',
            'batch_acc': f'{batch_acc:.4f}'
        })
    
    # 计算平均 loss 和 accuracy
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    
    # 关闭进度条并返回结果
    progress_bar.close()
    return avg_loss, avg_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 创建 tqdm 进度条
    progress_bar = tqdm(test_loader, desc="Evaluating", leave=True)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            # 计算当前批次的 loss 和 accuracy
            batch_loss = loss.item()
            pred = output.argmax(dim=1)
            batch_correct = pred.eq(target).sum().item()
            batch_total = target.size(0)
            batch_acc = batch_correct / batch_total
            
            # 累积总计值
            total_loss += batch_loss
            correct += batch_correct
            total += batch_total
            
            # 更新 tqdm 进度条的描述，实时显示 loss 和 accuracy
            progress_bar.set_postfix({
                'batch_loss': f'{batch_loss:.4f}',
                'batch_acc': f'{batch_acc:.4f}'
            })
    
    # 计算平均 loss 和 accuracy
    avg_loss = total_loss / len(test_loader)
    avg_acc = correct / total
    
    # 关闭进度条并返回结果
    progress_bar.close()
    return avg_loss, avg_acc

def main():
    # Hyperparameters
    num_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load DEAP dataset
    data_dir = "C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN_code\\dataset"
    label_type = [0, 2]  # [0,2] for valence, [1,2] for arousal
    
    train_loader, test_loader = dataset_prepare(
        window_length_sec=4,
        n_subjects=26,
        single_subject=False,
        load_all=True,
        only_EEG=True,
        label_type=label_type,
        data_dir=data_dir,
        batch_size=batch_size,
        normalize=True
    )
    
    # Initialize model
    model = SNNTTFS().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f'Best Test Acc: {best_acc:.4f}')

if __name__ == "__main__":
    main()