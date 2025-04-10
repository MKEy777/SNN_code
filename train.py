import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from utils.load_dataset_deap import dataset_prepare
from module.LIF import  SNNRateLayer, SurrogateGradientFunction


class SNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=200, output_dim=2):
        super(SNN, self).__init__()

        # 第一层参数
        self.layer1_params = {
            'thresh': 0.5,
            'tau_mem': 10.0,
            'tau_syn': 5.0,
            'dt': 1.0,
            'reset_mode': 'soft',
            'grad_type': 'multi_gaussian',
            'lens': 0.5,
            'gamma': 1.0,
            'hight': 0.15,
            'learn_params': True
        }

        # 第二层参数
        self.layer2_params = {
            'thresh': 1.0,
            'tau_mem': 20.0,
            'tau_syn': 10.0,
            'dt': 1.0,
            'reset_mode': 'soft',
            'grad_type': 'multi_gaussian',
            'lens': 0.4,
            'gamma': 1.2,
            'hight': 0.2,
            'learn_params': True
        }

        # 创建两层网络 32-> 200 -> 2
        self.fc1 = SNNRateLayer(input_dim, hidden_dim, **self.layer1_params)
        self.fc2 = SNNRateLayer(hidden_dim, output_dim, **self.layer2_params)

    def set_neuron_state(self, batch_size, device):
        # 重置第一层状态
        self.fc1.neurons.syn = torch.zeros(batch_size, self.fc1.neurons.num_neurons, device=device)
        self.fc1.neurons.mem = torch.zeros(batch_size, self.fc1.neurons.num_neurons, device=device)

        # 重置第二层状态
        self.fc2.neurons.syn = torch.zeros(batch_size, self.fc2.neurons.num_neurons, device=device)
        self.fc2.neurons.mem = torch.zeros(batch_size, self.fc2.neurons.num_neurons, device=device)

    def forward(self, input):
        # input shape: [batch_size, seq_length, input_dim]
        batch_size, seq_length, input_dim = input.shape
        device = input.device

        # 初始化神经元状态
        self.set_neuron_state(batch_size, device)

        # 存储输出
        output = torch.zeros(batch_size, self.fc2.neurons.num_neurons, device=device)

        # 存储每个时间步的状态（如果需要）
        all_spikes_1 = []
        all_mems_1 = []
        all_spikes_2 = []
        all_mems_2 = []

        # 按时间步处理
        for t in range(seq_length):
            input_t = input[:, t, :]

            # 第一层
            weighted_1 = torch.matmul(input_t, self.fc1.weight)
            alpha_syn_1 = torch.exp(-self.fc1.neurons.dt / self.fc1.neurons.tau_syn)
            alpha_mem_1 = torch.exp(-self.fc1.neurons.dt / self.fc1.neurons.tau_mem)
            input_scale_1 = (1 - alpha_syn_1) * self.fc1.neurons.tau_syn
            self.fc1.neurons.syn = alpha_syn_1 * self.fc1.neurons.syn + input_scale_1 * weighted_1
            self.fc1.neurons.mem = alpha_mem_1 * self.fc1.neurons.mem + \
                                   (1 - alpha_mem_1) * self.fc1.neurons.syn

            spike_1 = SurrogateGradientFunction.apply(
                self.fc1.neurons.mem - self.fc1.neurons.thresh,
                self.fc1.neurons.grad_type,
                self.fc1.neurons.lens,
                self.fc1.neurons.gamma,
                self.fc1.neurons.hight
            )

            if self.fc1.neurons.reset_mode == 'soft':
                self.fc1.neurons.mem = self.fc1.neurons.mem - spike_1 * self.fc1.neurons.thresh
            else:
                self.fc1.neurons.mem = self.fc1.neurons.mem * (1 - spike_1)

            # 第二层
            weighted_2 = torch.matmul(spike_1, self.fc2.weight)
            alpha_syn_2 = torch.exp(-self.fc2.neurons.dt / self.fc2.neurons.tau_syn)
            alpha_mem_2 = torch.exp(-self.fc2.neurons.dt / self.fc2.neurons.tau_mem)
            input_scale_2 = (1 - alpha_syn_2) * self.fc2.neurons.tau_syn
            self.fc2.neurons.syn = alpha_syn_2 * self.fc2.neurons.syn + input_scale_2 * weighted_2
            self.fc2.neurons.mem = alpha_mem_2 * self.fc2.neurons.mem + \
                                   (1 - alpha_mem_2) * self.fc2.neurons.syn

            spike_2 = SurrogateGradientFunction.apply(
                self.fc2.neurons.mem - self.fc2.neurons.thresh,
                self.fc2.neurons.grad_type,
                self.fc2.neurons.lens,
                self.fc2.neurons.gamma,
                self.fc2.neurons.hight
            )

            if self.fc2.neurons.reset_mode == 'soft':
                self.fc2.neurons.mem = self.fc2.neurons.mem - spike_2 * self.fc2.neurons.thresh
            else:
                self.fc2.neurons.mem = self.fc2.neurons.mem * (1 - spike_2)

            if t > 0:
                output += self.fc2.neurons.mem

            all_spikes_1.append(spike_1)
            all_mems_1.append(self.fc1.neurons.mem)
            all_spikes_2.append(spike_2)
            all_mems_2.append(self.fc2.neurons.mem)

        output = output / seq_length

        all_states = {
            'spikes_1': torch.stack(all_spikes_1, dim=1),
            'mems_1': torch.stack(all_mems_1, dim=1),
            'spikes_2': torch.stack(all_spikes_2, dim=1),
            'mems_2': torch.stack(all_mems_2, dim=1)
        }

        return output, all_states

def train(model, train_loader, optimizer, criterion, device, epochs=10):
    """
    Train the SNN model.
    
    Args:
        model: SNN model instance
        train_loader: DataLoader for training data
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device to run on (CPU/GPU)
        epochs: Number of training epochs
    
    Returns:
        train_losses: List of average losses per epoch
        train_accuracies: List of accuracies per epoch
    """
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Reset neuron states at the start of each batch
            model.set_neuron_state(data.size(0), device)

            # Forward pass: [batch_size, seq_length, input_dim] -> [batch_size, output_dim]
            output, _ = model(data)
            loss = criterion(output, target)  # Output is membrane potential, not probabilities

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute predictions (argmax over output_dim)
            pred = torch.argmax(output, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return train_losses, train_accuracies

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the SNN model on the test set.
    
    Args:
        model: SNN model instance
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on (CPU/GPU)
    
    Returns:
        test_loss: Average test loss
        test_accuracy: Test accuracy
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Reset neuron states
            model.set_neuron_state(data.size(0), device)

            # Forward pass
            output, _ = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()

            # Compute predictions
            pred = torch.argmax(output, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# Main function
def main():
    """
    Main script to load data, initialize model, train, evaluate, and visualize results.
    """
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data directory (adjust to your local path)
    data_dir = "C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN_code\\dataset"

    # Load and prepare data
    train_loader, test_loader = dataset_prepare(
        window_length_sec=4,          # 4-second windows (512 time points at 128Hz)
        n_subjects=26,               # Number of subjects for training
        single_subject=False,        # Use multiple subjects
        load_all=True,               # Load all data and split into train/test
        only_EEG=True,               # Use only EEG channels
        label_type=[0, 2],           # Valence, 2 classes (0 or 1)
        data_dir=data_dir,
        batch_size=batch_size,
        normalize=True               # Apply Z-score normalization
    )

    # Initialize model
    input_dim = 32   # Number of EEG channels
    hidden_dim = 200 # Hidden layer size
    output_dim = 2   # Number of classes (binary valence)
    model = SNN(input_dim, hidden_dim, output_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    train_losses, train_accuracies = train(model, train_loader, optimizer, criterion, device, epochs)

    # Evaluate the model
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

    # Visualize training progress
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

