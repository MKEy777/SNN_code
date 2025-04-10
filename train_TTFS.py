import argparse
import time
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from snn_module import SNNModel, SpikingDense

# ### 数据集加载

def load_mnist_data(args):
    """加载并预处理 MNIST 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # 展平为 (784,)
        transforms.Lambda(lambda x: 1 - x),  # TTFS 转换：1 - x
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 添加噪声（可选）
    if args.noise > 0:
        train_dataset.data = torch.clamp(train_dataset.data + torch.randn_like(train_dataset.data.float()) * args.noise, 0, 1)
        test_dataset.data = torch.clamp(test_dataset.data + torch.randn_like(test_dataset.data.float()) * args.noise, 0, 1)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, test_loader

# ### 训练和测试函数

def train(model, train_loader, optimizer, device, epoch, args):
    """训练一个 epoch"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'第 {epoch+1}/{args.epochs} 轮, 批次 {batch_idx}, 损失: {loss.item():.4f}')

def test(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100. * correct / total
    return accuracy

# ### 主函数

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='SNN 在 MNIST 上的训练')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--testing', type=str, default='True', help='是否执行测试 (True|False)')
    parser.add_argument('--save', type=str, default='False', help='训练后是否保存模型 (True|False)')
    parser.add_argument('--noise', type=float, default=0.0, help='噪声标准差')
    parser.add_argument('--time_bits', type=int, default=0, help='时间量化位数 (0 表示禁用)')
    parser.add_argument('--weight_bits', type=int, default=0, help='权重量化位数 (0 表示禁用)')
    parser.add_argument('--w_min', type=float, default=-1.0, help='权重最小值')
    parser.add_argument('--w_max', type=float, default=1.0, help='权重最大值')
    args = parser.parse_args()
    args.testing = args.testing.lower() == 'true'
    args.save = args.save.lower() == 'true'
    args.model_name = 'MNIST-FC2'

    # 设备设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 鲁棒性参数
    robustness_params = {
        'noise': args.noise,
        'time_bits': args.time_bits,
        'weight_bits': args.weight_bits,
        'w_min': args.w_min,
        'w_max': args.w_max,
    }

    # 加载数据
    train_loader, test_loader = load_mnist_data(args)

    # 定义模型
    model = SNNModel()
    model.layers.append(SpikingDense(units=256, input_dim=784, robustness_params=robustness_params))
    model.layers.append(SpikingDense(units=256, input_dim=256, robustness_params=robustness_params))
    model.layers.append(SpikingDense(units=10, input_dim=256, outputLayer=True, robustness_params=robustness_params))
    model.to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    start_time = time.time()
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, device, epoch, args)
        if args.testing:
            accuracy = test(model, test_loader, device)
            print(f'第 {epoch+1}/{args.epochs} 轮, 测试准确率: {accuracy:.2f}%')

    # 保存模型
    if args.save:
        save_path = f"{args.model_name}.pt"
        torch.save(model.state_dict(), save_path)
        print(f'模型已保存至 {save_path}')

    print(f'总训练时间: {time.time() - start_time:.2f} 秒')

if __name__ == '__main__':
    main()