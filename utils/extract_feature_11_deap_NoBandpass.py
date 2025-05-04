import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import os
import time

# 特征名称 (11个特征)
FEATURE_NAMES = [
    "Mean PTP", "Mean Square Value", "Variance", "PSD Sum", "Max PSD Value",
    "Freq at Max PSD", "Hjorth Mobility", "Hjorth Complexity", "Shannon Entropy",
    "C0 Complexity (FWHM)", "Power Spectrum Entropy"
]

# --- 参数设定 ---
INPUT_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN_code\PerSession_MAT_Deap_nofilter"
OUTPUT_DIR_FEATURES = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN_code\deap_features_NoFilter"
FS = 128  # DEAP 数据集的采样频率 (Hz)
N_CHANNELS = 32  # DEAP 数据集中的 EEG 通道数
N_FEATURES_PER_CHANNEL = len(FEATURE_NAMES) # 每个通道提取的特征数量
# 每个段的总特征数 = 通道数 * 每个通道的特征数
EXPECTED_TOTAL_FEATURES = N_CHANNELS * N_FEATURES_PER_CHANNEL

# --- 特征计算函数  ---

def calculate_mean_ptp(data, sub_window_size=FS):
    """计算平均峰峰值 (Peak-to-Peak)"""
    valid_data = data[np.isfinite(data)] # 忽略 NaN 或 Inf
    if len(valid_data) < 2:
        return 0.0
    sub_window_size = min(sub_window_size, len(valid_data))
    if sub_window_size <= 1:
        return np.ptp(valid_data)
    ptp_values = []
    for i in range(0, len(valid_data) - sub_window_size + 1):
        sub_window = valid_data[i : i + sub_window_size]
        if len(sub_window) >= 2:
            ptp_values.append(np.ptp(sub_window))
    return np.mean(ptp_values) if ptp_values else np.ptp(valid_data)

def calculate_msv(data):
    """计算均方值 (Mean Square Value)"""
    valid_data = data[np.isfinite(data)]
    return np.mean(np.square(valid_data)) if len(valid_data) > 0 else 0.0

def calculate_var(data):
    """计算方差 (Variance)"""
    valid_data = data[np.isfinite(data)]
    return np.var(valid_data) if len(valid_data) >= 2 else 0.0

def calculate_psd_sum(data, fs=FS):
    """计算功率谱密度 (PSD) 的总和"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2:
        return 0.0
    nperseg = min(256, len(valid_data))
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
        return np.sum(Pxx) if Pxx is not None and len(Pxx) > 0 else 0.0
    except ValueError:
         return 0.0

def calculate_max_psd(data, fs=FS):
    """计算最大 PSD 值及其对应的频率"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2:
        return 0.0, 0.0
    nperseg = min(256, len(valid_data))
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
        if Pxx is None or len(Pxx) == 0:
            return 0.0, 0.0
        max_idx = np.argmax(Pxx)
        return Pxx[max_idx], f[max_idx]
    except ValueError:
        return 0.0, 0.0

def calculate_hjorth_mobility(data):
    """计算 Hjorth 活动性 (Mobility)"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2:
        return 0.0
    var_data = np.var(valid_data)
    if var_data < 1e-10:
        return 0.0
    diff1 = np.diff(valid_data)
    if len(diff1) < 2:
        return 0.0
    var_diff1 = np.var(diff1)
    ratio = var_diff1 / var_data
    return np.sqrt(max(0, ratio))

def calculate_hjorth_complexity(data):
    """计算 Hjorth 复杂度 (Complexity)"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 4:
        return 0.0
    mobility_data = calculate_hjorth_mobility(valid_data)
    if mobility_data < 1e-10:
        return 0.0
    diff1 = np.diff(valid_data)
    if len(diff1) < 3:
        return 0.0
    var_diff1 = np.var(diff1)
    if var_diff1 < 1e-10:
        return 0.0
    diff2 = np.diff(diff1)
    if len(diff2) < 2:
        return 0.0
    var_diff2 = np.var(diff2)
    ratio_diff = var_diff2 / var_diff1
    mobility_diff1 = np.sqrt(max(0, ratio_diff))
    return mobility_diff1 / mobility_data

def calculate_shannon_entropy(data, bins=10):
    """计算香农熵 (Shannon Entropy)"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0 or np.ptp(valid_data) < 1e-10:
        return 0.0
    actual_bins = max(1, min(int(bins), len(np.unique(valid_data))))
    hist, bin_edges = np.histogram(valid_data, bins=actual_bins, density=False)
    counts = hist[hist > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts / len(valid_data)
    return -np.sum(probs * np.log2(probs[probs > 1e-10])) # 避免 log(0)

def calculate_c0_complexity(data, fs=FS):
    """计算 C0 复杂度 (功率谱峰值的半高全宽 FWHM)"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2 or np.ptp(valid_data) < 1e-10:
        return 0.0
    nperseg = min(256, len(valid_data))
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
        if Pxx is None or len(Pxx) == 0 or np.all(Pxx < 1e-10):
            return 0.0
        max_idx = np.argmax(Pxx)
        max_power = Pxx[max_idx]
        if max_power < 1e-10:
            return 0.0
        half_max = max_power / 2.0
        indices_above_half = np.where(Pxx >= half_max)[0]
        if len(indices_above_half) == 0:
             return 0.0
        fwhm = f[indices_above_half[-1]] - f[indices_above_half[0]]
        return fwhm
    except ValueError:
        return 0.0

def calculate_power_spectrum_entropy(data, fs=FS):
    """计算功率谱熵 (Power Spectrum Entropy)"""
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2 or np.ptp(valid_data) < 1e-10:
        return 0.0
    nperseg = min(256, len(valid_data))
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
        if Pxx is None or len(Pxx) == 0:
            return 0.0
        Pxx_nz = Pxx[Pxx > 1e-10]
        if len(Pxx_nz) == 0:
            return 0.0
        Pxx_sum = np.sum(Pxx_nz)
        if Pxx_sum < 1e-10:
            return 0.0
        Pxx_norm = Pxx_nz / Pxx_sum
        Pxx_norm = Pxx_norm[Pxx_norm > 1e-10]
        if len(Pxx_norm) == 0:
             return 0.0
        return -np.sum(Pxx_norm * np.log(Pxx_norm[Pxx_norm > 1e-10])) # 使用自然对数，避免log(0)
    except ValueError:
        return 0.0

# --- MODIFIED Feature Extraction Function (No Filtering) ---

def extract_features_for_segment(segment_data_raw, fs=FS):
    """
    为单个数据段（所有通道）提取特征。
    直接在输入的原始 (基线校正后) 数据上操作，不进行额外滤波。

    Args:
        segment_data_raw (np.ndarray): 单个数据段的原始 EEG 数据。
                                       形状: (通道数, 窗口样本数) 即 (32, 512)
        fs (int): 采样频率。

    Returns:
        np.ndarray: 该数据段展平后的特征向量。
                    形状: (通道数 * 每个通道的特征数,) 即 (32 * 11 = 352,)
    """
    all_segment_features = [] # 存储该段所有特征的列表
    n_channels, window_len = segment_data_raw.shape

    # 仅遍历每个通道 (因为不需要区分频带了)
    for c in range(n_channels):
        # 获取该通道的原始时间序列数据 (未滤波)
        ts_raw = segment_data_raw[c, :]

        # --- 在原始时间序列 (ts_raw) 上计算 11 个特征 ---
        features = []
        # 调用特征计算函数，传入 ts_raw
        features.append(calculate_mean_ptp(ts_raw, sub_window_size=fs))
        features.append(calculate_msv(ts_raw))
        features.append(calculate_var(ts_raw))
        features.append(calculate_psd_sum(ts_raw, fs=fs))
        max_psd_val, freq_at_max = calculate_max_psd(ts_raw, fs=fs)
        features.append(max_psd_val)
        features.append(freq_at_max)
        features.append(calculate_hjorth_mobility(ts_raw))
        features.append(calculate_hjorth_complexity(ts_raw))
        features.append(calculate_shannon_entropy(ts_raw))
        features.append(calculate_c0_complexity(ts_raw, fs=fs))
        features.append(calculate_power_spectrum_entropy(ts_raw, fs=fs))
        # --- 特征计算结束 ---

        # 将当前通道的特征添加到总列表中
        all_segment_features.extend(features)

    # 检查提取的特征数量是否符合预期 (32 * 11 = 352)
    if len(all_segment_features) != EXPECTED_TOTAL_FEATURES:
         print(f"警告：提取了 {len(all_segment_features)} 个特征，预期为 {EXPECTED_TOTAL_FEATURES} (通道数 * 特征数/通道)。")
         # 可以选择填充或截断以匹配预期长度
         expected_len = EXPECTED_TOTAL_FEATURES
         actual_len = len(all_segment_features)
         if actual_len < expected_len:
             padding = np.full(expected_len - actual_len, np.nan) # 使用 NaN 填充
             all_segment_features.extend(padding)
         elif actual_len > expected_len:
             all_segment_features = all_segment_features[:expected_len] # 截断

    # 将列表转换为 NumPy 数组返回
    return np.array(all_segment_features, dtype=np.float32)

# --- MODIFIED File Processing Function ---

def process_single_file(fpath, output_dir_features, fs=FS):
    """
    从预处理后的 .mat 文件加载数据，为每个数据段提取特征(无额外滤波)，并保存结果。
    """
    fname = os.path.basename(fpath)
    # 修改输出文件名以反映特征是直接提取的
    output_fname = os.path.splitext(fname)[0].replace('_nofilter_binaryVA', '') + "_direct_features.mat"
    output_fpath = os.path.join(output_dir_features, output_fname)

    if os.path.exists(output_fpath):
        print(f"  跳过 {fname}，输出文件已存在: {output_fname}")
        return

    print(f"  正在加载数据文件: {fname}")
    try:
        mat_data = loadmat(fpath)
        # 数据形状是 (分段数, 通道数, 窗口样本数)
        seg_X_raw = mat_data['data']
        # 标签形状是 (分段数, 2)
        seg_y = mat_data['labels']
    except Exception as e:
        print(f"  加载 {fname} 时出错: {e}。跳过此文件。")
        return

    n_segments_in_file, n_chans, n_samples = seg_X_raw.shape

    if n_segments_in_file == 0:
        print(f"  跳过 {fname}，因为它不包含任何数据段。")
        return

    if n_chans != N_CHANNELS:
        print(f"  警告：在 {fname} 中发现 {n_chans} 个通道，预期为 {N_CHANNELS}。")

    file_features_list = []
    file_labels_list = []
    processed_segment_count = 0
    skipped_segment_count = 0

    print(f"  开始处理 {n_segments_in_file} 个数据段 (无额外滤波)...")
    segment_time_start = time.time()

    for seg_idx in range(n_segments_in_file):
        # 获取当前段的原始数据：形状 (通道数, 窗口样本数)
        segment_data_raw = seg_X_raw[seg_idx, :, :]
        label = seg_y[seg_idx]

        # 提取特征（直接在原始数据上操作）
        features_vector = extract_features_for_segment(segment_data_raw, fs=fs)

        if np.isnan(features_vector).any():
            print(f"    数据段 {seg_idx+1} 产生了 NaN 特征。跳过此段。")
            skipped_segment_count += 1
            continue

        file_features_list.append(features_vector)
        file_labels_list.append(label)
        processed_segment_count += 1

        if (processed_segment_count + skipped_segment_count) % 50 == 0 or seg_idx == n_segments_in_file - 1:
             current_time = time.time()
             total_segments_processed = processed_segment_count + skipped_segment_count
             if total_segments_processed > 0:
                 avg_time_per_segment = (current_time - segment_time_start) / total_segments_processed
                 print(f"    已处理数据段 {total_segments_processed}/{n_segments_in_file}... (平均 {avg_time_per_segment:.3f} 秒/段)")


    if file_features_list:
        # 特征矩阵形状: (有效段数, 352)
        features_matrix = np.vstack(file_features_list)
        # 标签矩阵形状: (有效段数, 2)
        labels_matrix = np.array(file_labels_list, dtype=seg_y.dtype)

        print(f"  准备保存特征矩阵，形状: {features_matrix.shape}")
        print(f"  准备保存标签矩阵，形状: {labels_matrix.shape}")

        savemat(output_fpath, {'features': features_matrix, 'labels': labels_matrix}, do_compression=True)
        print(f"  已将 {fname} 的特征保存到 {output_fname}")
        print(f"  处理了 {processed_segment_count} 个有效段，跳过了 {skipped_segment_count} 个段。")

    else:
        print(f"  未能为 {fname} 提取或保存任何有效特征。")

# --- 主执行函数 (路径更新，逻辑基本不变) ---

def run_deap_processing(input_dir, output_dir_features):
    """运行 DEAP 数据处理的主函数"""
    start_time_total = time.time()

    if not os.path.exists(output_dir_features):
        print(f"正在创建输出目录: {output_dir_features}")
        os.makedirs(output_dir_features)

    try:
        mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat') and os.path.isfile(os.path.join(input_dir, f))])
    except FileNotFoundError:
        print(f"错误：输入目录 '{input_dir}' 不存在。请检查路径。")
        return

    if not mat_files:
        print(f"错误：在 '{input_dir}' 中未找到 .mat 文件。请确保预处理脚本已正确运行。")
        return

    total_files = len(mat_files)
    print(f"在 '{input_dir}' 中找到 {total_files} 个预处理文件。开始提取特征 (无额外滤波)...")

    for i, fname in enumerate(mat_files):
        print(f"\n--- 正在处理文件 {i+1}/{total_files}: {fname} ---")
        start_file_time = time.time()
        fpath = os.path.join(input_dir, fname)
        process_single_file(fpath, output_dir_features, fs=FS)
        file_elapsed = time.time() - start_file_time
        print(f"  完成处理 {fname}，耗时 {file_elapsed:.2f} 秒。")

    elapsed_total = time.time() - start_time_total
    print(f"\n所有 {total_files} 个文件的特征提取完成，总耗时 {elapsed_total:.2f} 秒。")
    print(f"特征已保存到: {output_dir_features}")

# --- 脚本入口 ---
if __name__ == "__main__":
    run_deap_processing(INPUT_DIR, OUTPUT_DIR_FEATURES)
    print("\n脚本执行完毕。")