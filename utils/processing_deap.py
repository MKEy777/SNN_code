import os
import numpy as np
import pickle as cPickle
from scipy.signal import butter, filtfilt
from scipy.io import savemat

# --------------------------------------------
# 离散化情绪标签：二分类逻辑
# --------------------------------------------
def labels_quantization(labels, num_classes):
    """
    将连续情绪标签离散化为二分类 (高/低)。
    参数:
        labels: 原始标签，形状 (试验次数, 2)，包含 valence 和 arousal 值。
        num_classes: 固定为 2。
    返回:
        形状为 (2, 试验次数) 的数组，第一行是二分类 Valence，第二行是二分类 Arousal。
    """
    median_val = 5  # 中值 5 作为阈值
    labels_val = np.zeros(labels.shape[0])
    labels_val[labels[:, 0] > median_val] = 1  # valence > 5 为高 (1)

    labels_arousal = np.zeros(labels.shape[0])
    labels_arousal[labels[:, 1] > median_val] = 1  # arousal > 5 为高 (1)

    return np.array([labels_val, labels_arousal])  # 形状 (2, n_trials)

# --------------------------------------------
# 带通滤波器函数
# --------------------------------------------
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    对数据应用带通滤波器。
    参数:
        data: 输入数据，形状 (通道数, 时间点数)。
        lowcut: 下截止频率。
        highcut: 上截止频率。
        fs: 采样频率。
        order: 滤波器阶数。
    返回:
        滤波后的数据。
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=1)
    return y

# --------------------------------------------
# 处理单个被试数据
# --------------------------------------------
def process_subject_data(filepath, bands, fs=128, window_sec=4, step_sec=2, only_EEG=True):
    """
    加载、滤波并分割单个被试数据。
    参数:
        filepath: .dat 文件路径。
        bands: 频带列表 [(低1, 高1), (低2, 高2), ...]。
        fs: 采样频率。
        window_sec: 窗口长度（秒）。
        step_sec: 步长（秒）。
        only_EEG: 是否仅使用前 32 个 EEG 通道。
    返回:
        tuple: (segmented_data, segmented_labels)
               segmented_data 形状: (n_segments, len(bands), 32, window_samples)
               segmented_labels 形状: (n_segments, 2)
    """
    print(f"正在处理文件: {filepath}")
    with open(filepath, 'rb') as f:
        loaddata = cPickle.load(f, encoding="latin1")

    labels_raw = loaddata['labels'][:, :2]  # (trials, 2)
    data_raw = loaddata['data'].astype(np.float32)  # (trials, channels, time)

    # 仅使用 EEG 通道
    if only_EEG:
        data_raw = data_raw[:, :32, :]
        print(f"  仅使用 EEG 通道。数据形状: {data_raw.shape}")

    num_trials, num_channels, total_timepoints = data_raw.shape

    # 基线校正
    baseline_samples = 3 * fs
    baseline_data = data_raw[:, :, :baseline_samples]
    baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
    data_corrected = data_raw[:, :, baseline_samples:] - baseline_mean
    print(f"  基线已校正。数据形状: {data_corrected.shape}")

    # 滤波并堆叠频带
    filtered_data_all_bands = []
    for low, high in bands:
        print(f"  应用滤波器: {low}-{high} Hz")
        filtered_band_data = np.zeros_like(data_corrected)
        for trial in range(num_trials):
            filtered_band_data[trial, :, :] = bandpass_filter(data_corrected[trial, :, :], low, high, fs)
        filtered_data_all_bands.append(filtered_band_data)

    filtered_data_stacked = np.stack(filtered_data_all_bands, axis=1)  # (n_trials, 4, 32, timepoints - baseline_samples)
    print(f"  滤波数据已堆叠。形状: {filtered_data_stacked.shape}")

    # 二分类标签
    labels_quantized_trial_all = labels_quantization(labels_raw, 2)
    print(f"  二分类标签形状: {labels_quantized_trial_all.shape}")

    # 滑动窗口分割
    window_samples = window_sec * fs  # 512
    step_samples = step_sec * fs
    _, num_bands, num_channels, segmentable_timepoints = filtered_data_stacked.shape

    all_segments = []
    all_segment_labels_quantized = []

    for trial_idx in range(num_trials):
        trial_data = filtered_data_stacked[trial_idx]  # (4, 32, timepoints)
        trial_label_pair_quantized = labels_quantized_trial_all[:, trial_idx]

        start = 0
        while start + window_samples <= segmentable_timepoints:
            segment = trial_data[:, :, start:start + window_samples]  # (4, 32, 512)
            all_segments.append(segment)
            all_segment_labels_quantized.append(trial_label_pair_quantized)
            start += step_samples

    segmented_data_np = np.array(all_segments)  # (n_segments, 4, 32, 512)
    segmented_labels_np = np.array(all_segment_labels_quantized)  # (n_segments, 2)
    print(f"  分割完成。数据形状: {segmented_data_np.shape}")
    print(f"  标签形状: {segmented_labels_np.shape}")

    return segmented_data_np, segmented_labels_np

# --------------------------------------------
# 主程序
# --------------------------------------------
if __name__ == '__main__':
    data_dir = "C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN_code\\dataset"
    output_dir = "C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN_code\\PerSession_MAT_Deap"
    fs = 128
    bands = [(4, 8), (8, 13), (13, 30), (30, 45)]  # 4 个频带
    window_sec = 4
    step_sec = 2
    only_EEG = True

    os.makedirs(output_dir, exist_ok=True)

    filenames = [f for f in os.listdir(data_dir) if f.endswith('.dat') and f.startswith('s')]
    filenames.sort()

    print(f"找到 {len(filenames)} 个被试文件")
    for filename in filenames:
        subject_filepath = os.path.join(data_dir, filename)
        subject_id = filename.split('.')[0]

        segmented_data, segmented_labels = process_subject_data(
            subject_filepath, bands, fs, window_sec, step_sec, only_EEG
        )

        mat_data = {
            'data': segmented_data,  # (n_segments, 4, 32, 512)
            'labels': segmented_labels  # (n_segments, 2)
        }

        output_filename = f"{subject_id}_win{window_sec}s_step{step_sec}s_bands_binaryVA.mat"
        output_filepath = os.path.join(output_dir, output_filename)
        savemat(output_filepath, mat_data)
        print(f"  保存到: {output_filepath}\n")

    print("处理完成。")