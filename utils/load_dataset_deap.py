import os
import numpy as np
import pickle as cPickle
from sklearn.utils import shuffle
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

##############################################
# 离散化情绪标签：支持2类、3类、4类
##############################################
def labels_quantization(labels, num_classes):
    """
    将连续情绪标签离散化
    参数:
        labels: 原始标签，形状 (样本数, 2)
        num_classes: 分类数 (2、3或4)
    返回:
        离散化后的标签数组
    """
    if num_classes == 2:
        median_val = 5
        labels_val = np.zeros(labels.shape[0])
        labels_val[labels[:, 0] > median_val] = 1

        labels_arousal = np.zeros(labels.shape[0])
        labels_arousal[labels[:, 1] > median_val] = 1

        return np.array([labels_val, labels_arousal])

    elif num_classes == 3:
        low_value = 3
        high_value = 6
        labels_val = np.zeros(labels.shape[0])
        labels_val[(labels[:, 0] > low_value) & (labels[:, 0] <= high_value)] = 1
        labels_val[labels[:, 0] > high_value] = 2

        labels_arousal = np.zeros(labels.shape[0])
        labels_arousal[(labels[:, 1] > low_value) & (labels[:, 1] <= high_value)] = 1
        labels_arousal[labels[:, 1] > high_value] = 2

        return np.array([labels_val, labels_arousal])

    else:  # 4类
        median_val = 5
        labels_all = np.zeros(labels.shape[0])
        labels_all[(labels[:, 0] > median_val) & (labels[:, 1] <= median_val)] = 1
        labels_all[(labels[:, 0] <= median_val) & (labels[:, 1] > median_val)] = 2
        labels_all[(labels[:, 0] > median_val) & (labels[:, 1] > median_val)] = 3
        return labels_all

##############################################
# 加载并切片数据（带基线校正）
##############################################
def load_with_path(filepaths, label_type=[0, 2], only_phys=False, only_EEG=True, window_length_sec=4):
    """
    加载多个被试数据并切片，包含基线校正
    参数:
        filepaths: 数据文件路径列表
        label_type: [标签索引 (0: valence, 1: arousal), 分类数]
        only_phys / only_EEG: 选择生理信号或EEG
        window_length_sec: 窗口时长（秒）
    返回:
        all_data: 切片后的EEG数据 (样本数, 通道数, 时间点数)
        all_labels_final: 离散化标签
    """
    all_data = []
    all_labels = []

    for filepath in filepaths:
        with open(filepath, 'rb') as f:
            loaddata = cPickle.load(f, encoding="latin1")
        labels = loaddata['labels']
        data = loaddata['data'].astype(np.float32)

        # 选择信号类型
        if only_phys:
            data = data[:, 32:, :]
        elif only_EEG:
            data = data[:, :32, :]

        # 提取前3秒基线数据（128Hz * 3s = 384个时间点）
        baseline_data = data[:, :, :384]

        # 计算每个trial和每个通道的基线均值
        baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)

        # 对后续数据进行基线校正
        corrected_data = data[:, :, 384:] - baseline_mean

        # 固定窗口切片
        sample_length = window_length_sec * 128
        num_samples = (corrected_data.shape[2] - sample_length) // sample_length + 1

        segmented_data = []
        repeated_labels = []
        for trial in range(corrected_data.shape[0]):
            trial_data = corrected_data[trial]
            trial_label = labels[trial]
            for i in range(num_samples):
                start = i * sample_length
                end = start + sample_length
                segmented_data.append(trial_data[:, start:end])
                repeated_labels.append(trial_label)

        all_data.append(np.array(segmented_data))
        all_labels.append(np.array(repeated_labels))

    all_data = np.vstack(all_data)
    all_labels = np.vstack(all_labels)

    # 标签离散化处理
    if label_type[1] == 1:
        processed_labels = labels_quantization(all_labels, 4)
    elif label_type[1] == 2:
        processed_labels = labels_quantization(all_labels, 2)
    elif label_type[1] == 3:
        processed_labels = labels_quantization(all_labels, 3)
    else:
        processed_labels = labels_quantization(all_labels, 4)

    # 根据valence或arousal选择标签
    if processed_labels.ndim == 2:
        all_labels_final = processed_labels[label_type[0]].squeeze()
    else:
        all_labels_final = processed_labels

    return all_data, all_labels_final

##############################################
# 加载DEAP数据
##############################################
def load_DEAP(data_dir, n_subjects=26, single_subject=False, load_all=False, only_phys=False, only_EEG=True,
              label_type=[0, 2], window_length_sec=4):
    """
    加载DEAP数据
    参数:
        data_dir: 数据文件目录
        n_subjects: 训练的被试数
        single_subject: 是否单个被试
        load_all: 是否加载所有被试
    返回:
        单被试或全部被试数据及标签；或训练/测试数据集
    """
    filenames = os.listdir(data_dir)
    filepaths = [os.path.join(data_dir, f) for f in filenames]

    if single_subject:
        train_paths = [filepaths[n_subjects - 1]]
        train_names = [filenames[n_subjects - 1]]
        train_data, train_labels = load_with_path(train_paths, label_type, only_phys, only_EEG, window_length_sec)
        return train_data, train_labels, train_names

    if load_all:
        train_paths = filepaths
        train_names = filenames
        train_data, train_labels = load_with_path(train_paths, label_type, only_phys, only_EEG, window_length_sec)
        return train_data, train_labels, train_names

    # 划分训练和测试
    filepaths, filenames = shuffle(filepaths, filenames, random_state=29)
    train_paths = filepaths[:n_subjects]
    test_paths = filepaths[n_subjects:]
    train_names = filenames[:n_subjects]
    test_names = filenames[n_subjects:]
    train_data, train_labels = load_with_path(train_paths, label_type, only_phys, only_EEG, window_length_sec)
    test_data, test_labels = load_with_path(test_paths, label_type, only_phys, only_EEG, window_length_sec)

    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    return train_data, train_labels, train_names, test_data, test_labels, test_names

##############################################
# 构建PyTorch DataLoader
##############################################
def dataset_prepare(window_length_sec=4, n_subjects=26, single_subject=False, load_all=False,
                    only_phys=False, only_EEG=True, label_type=[0, 2], data_dir="...",
                    batch_size=64, normalize=True):
    """
    准备训练/测试集 DataLoader
    参数:
        window_length_sec, n_subjects, single_subject, load_all, only_phys, only_EEG, label_type, data_dir, batch_size, normalize
    返回:
        train_loader, test_loader
    """
    if single_subject:
        data, labels, _ = load_DEAP(data_dir, n_subjects, single_subject=True, load_all=False,
                                    label_type=label_type, only_phys=only_phys, only_EEG=only_EEG,
                                    window_length_sec=window_length_sec)
    elif load_all:
        data, labels, _ = load_DEAP(data_dir, n_subjects, single_subject=False, load_all=True,
                                    label_type=label_type, only_phys=only_phys, only_EEG=only_EEG,
                                    window_length_sec=window_length_sec)
    else:
        train_data, train_labels, _, test_data, test_labels, _ = load_DEAP(
            data_dir, n_subjects, single_subject=False, load_all=False, label_type=label_type,
            only_phys=only_phys, only_EEG=only_EEG, window_length_sec=window_length_sec)
        data = train_data
        labels = train_labels

    # 若加载所有数据则划分训练/验证集
    if load_all and not single_subject:
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=42, shuffle=True)
    else:
        train_data = data
        test_data = None
        train_labels = labels
        test_labels = None

    # Z-score标准化
    def z_score_normalize(data_array):
        for i in range(data_array.shape[0]):
            sample = data_array[i]
            mean = np.mean(sample, axis=1, keepdims=True)
            std = np.std(sample, axis=1, keepdims=True)
            std[std == 0] = 1e-6
            data_array[i] = (sample - mean) / std
        return data_array

    if normalize:
        train_data = z_score_normalize(train_data)
        if test_data is not None:
            test_data = z_score_normalize(test_data)

    # 转换数据形状为 [样本数, 时间点数, 通道数]
    train_data = np.transpose(train_data, (0, 2, 1))
    if test_data is not None:
        test_data = np.transpose(test_data, (0, 2, 1))

    # 构建 DataLoader
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.LongTensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if test_data is not None:
        test_dataset = TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_labels))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    print("Train data shape:", train_data.shape)
    if test_data is not None:
        print("Test data shape:", test_data.shape)

    return train_loader, test_loader

##############################################
# 示例调用
##############################################
if __name__ == '__main__':
    data_dir = "C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN_code\\dataset"
    train_loader, test_loader = dataset_prepare(
        window_length_sec=4,
        n_subjects=26,
        single_subject=False,
        load_all=True,
        only_EEG=True,
        label_type=[0, 2],
        data_dir=data_dir,
        batch_size=32,
        normalize=True
    )

    # 检查第一个batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        break