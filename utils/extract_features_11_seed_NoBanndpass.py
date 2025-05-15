import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import os
import time

# 特征名称列表
FEATURE_NAMES = [
    "Mean PTP", "Mean Square Value", "Variance", "PSD Sum", "Max PSD Value",
    "Freq at Max PSD", "Hjorth Mobility", "Hjorth Complexity", "Shannon Entropy",
    "C0 Complexity (FWHM)", "Power Spectrum Entropy"
]

# 输入目录：现在指向 processing 脚本的输出目录
INPUT_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\PerSession_MAT_NoBandpass_Fixed_BaselineCorrected"
# 输出目录：保存提取的特征
OUTPUT_DIR_FEATURES = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features_NoBandpass_Fixed_BaselineCorrected"
FS = 200 # 采样率 (Hz)
N_FEATURES_PER_CHANNEL = 11
N_CHANNELS = 62 # 通道数

# --- 特征计算函数 ---

def calculate_mean_ptp(data, sub_window_size=FS):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return 0.0
    sub_window_size = min(sub_window_size, len(valid_data))
    if sub_window_size <= 1: return np.ptp(valid_data)

    ptp_values = []
    for i in range(0, len(valid_data) - sub_window_size + 1):
        sub_window = valid_data[i : i + sub_window_size]
        if len(sub_window) >= 2: ptp_values.append(np.ptp(sub_window))

    return np.mean(ptp_values) if ptp_values else np.ptp(valid_data)

def calculate_msv(data):
    valid_data = data[np.isfinite(data)]
    return np.mean(np.square(valid_data)) if len(valid_data) > 0 else 0.0

def calculate_var(data):
    valid_data = data[np.isfinite(data)]
    return np.var(valid_data) if len(valid_data) >= 2 else 0.0

def calculate_psd_sum(data, fs=FS):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return 0.0
    nperseg = min(256, len(valid_data))
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
    except ValueError: # Handle cases where segment length might be too short for welch
        return 0.0
    return np.sum(Pxx) if Pxx is not None and len(Pxx) > 0 else 0.0

def calculate_max_psd(data, fs=FS):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return 0.0, 0.0
    nperseg = min(256, len(valid_data))
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
    except ValueError:
         return 0.0, 0.0
    if Pxx is None or len(Pxx) == 0: return 0.0, 0.0
    max_idx = np.argmax(Pxx)
    return Pxx[max_idx], f[max_idx]

def calculate_hjorth_mobility(data):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return 0.0
    var_data = np.var(valid_data)
    if var_data < 1e-10: return 0.0
    diff1 = np.diff(valid_data)
    if len(diff1) < 2: return 0.0 # Need at least 2 points in diff1 for variance
    var_diff1 = np.var(diff1)
    ratio = var_diff1 / var_data
    return np.sqrt(max(0, ratio)) # Ensure non-negative ratio

def calculate_hjorth_complexity(data):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 4: return 0.0 # Need enough points for diff2
    mobility_data = calculate_hjorth_mobility(valid_data)
    if mobility_data < 1e-10: return 0.0

    diff1 = np.diff(valid_data)
    if len(diff1) < 3: return 0.0 # Need enough points for diff2 variance
    var_diff1 = np.var(diff1)
    if var_diff1 < 1e-10: return 0.0

    diff2 = np.diff(diff1)
    if len(diff2) < 2: return 0.0 # Need at least 2 points in diff2 for variance
    var_diff2 = np.var(diff2)
    ratio_diff = var_diff2 / var_diff1
    mobility_diff1 = np.sqrt(max(0, ratio_diff)) # Ensure non-negative ratio
    return mobility_diff1 / mobility_data

def calculate_shannon_entropy(data, bins=10):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0 or np.ptp(valid_data) < 1e-10: return 0.0
    hist, bin_edges = np.histogram(valid_data, bins=max(1, int(bins)), density=False)
    counts = hist[hist > 0]
    if len(counts) == 0: return 0.0
    probs = counts / len(valid_data)
    # Ensure probabilities are valid (should be handled by hist>0, but added check)
    valid_probs = probs[(probs > 1e-10) & (probs <= 1.0)]
    if len(valid_probs) == 0: return 0.0
    return -np.sum(valid_probs * np.log2(valid_probs))

def calculate_c0_complexity(data, fs=FS):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2 or np.ptp(valid_data) < 1e-10: return 0.0
    nperseg = min(256, len(valid_data))
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
    except ValueError:
        return 0.0
    if Pxx is None or len(Pxx) == 0 or np.all(Pxx < 1e-10): return 0.0
    max_idx = np.argmax(Pxx)
    max_power = Pxx[max_idx]
    if max_power < 1e-10: return 0.0
    half_max = max_power / 2.0
    indices_above_half = np.where(Pxx > half_max)[0]
    if len(indices_above_half) == 0: return 0.0
    # Ensure indices are within bounds of f
    if indices_above_half[-1] >= len(f) or indices_above_half[0] >= len(f): return 0.0
    fwhm = f[indices_above_half[-1]] - f[indices_above_half[0]]
    return max(0, fwhm) # Ensure FWHM is non-negative

def calculate_power_spectrum_entropy(data, fs=FS):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2 or np.ptp(valid_data) < 1e-10: return 0.0
    nperseg = min(256, len(valid_data))
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
    except ValueError:
        return 0.0
    if Pxx is None or len(Pxx) == 0: return 0.0
    Pxx_filtered = Pxx[Pxx > 1e-10] # Filter small values early
    if len(Pxx_filtered) == 0: return 0.0
    Pxx_sum = np.sum(Pxx_filtered)
    if Pxx_sum < 1e-10: return 0.0 # Avoid division by zero or near-zero
    Pxx_norm = Pxx_filtered / Pxx_sum
    # Filter again after normalization just in case of numerical issues
    Pxx_norm = Pxx_norm[Pxx_norm > 1e-10]
    if len(Pxx_norm) == 0: return 0.0
    return -np.sum(Pxx_norm * np.log(Pxx_norm))


# --- 特征提取 ---
def extract_features_for_segment(segment_data, fs=FS):
    """
    为单个数据段提取特征。
    输入 segment_data 的形状: (channels, window_len)
    输出形状: (channels * N_FEATURES_PER_CHANNEL,)
    """
    all_segment_features = []
    if segment_data.ndim != 2:
         print(f"错误：extract_features_for_segment 期望二维数组，但接收到形状 {segment_data.shape}")
         return np.zeros(N_CHANNELS * N_FEATURES_PER_CHANNEL, dtype=np.float32) # Return zeros on shape mismatch

    channels, window_len = segment_data.shape
    if channels != N_CHANNELS:
        print(f"错误：期望 {N_CHANNELS} 个通道，但数据段有 {channels} 个通道。")
        return np.zeros(N_CHANNELS * N_FEATURES_PER_CHANNEL, dtype=np.float32) # Return zeros on channel mismatch

    for c in range(channels):
        ts = segment_data[c, :] # Time series for channel c
        # Handle potential all-NaN or constant series within feature functions
        features = []
        features.append(calculate_mean_ptp(ts, sub_window_size=fs))
        features.append(calculate_msv(ts))
        features.append(calculate_var(ts))
        features.append(calculate_psd_sum(ts, fs=fs))
        max_psd_val, freq_at_max = calculate_max_psd(ts, fs=fs)
        features.append(max_psd_val)
        features.append(freq_at_max)
        features.append(calculate_hjorth_mobility(ts))
        features.append(calculate_hjorth_complexity(ts))
        features.append(calculate_shannon_entropy(ts))
        features.append(calculate_c0_complexity(ts, fs=fs))
        features.append(calculate_power_spectrum_entropy(ts, fs=fs))

        # Append calculated features for this channel
        all_segment_features.extend(features)

    # Convert to numpy array and check final shape
    final_features = np.array(all_segment_features, dtype=np.float32)
    expected_len = N_CHANNELS * N_FEATURES_PER_CHANNEL
    if len(final_features) != expected_len:
        print(f"警告：特征向量长度不匹配。期望 {expected_len}，得到 {len(final_features)}。进行填充/截断。")
        # Pad with zeros or truncate if length is incorrect
        padded_features = np.zeros(expected_len, dtype=np.float32)
        actual_len = min(len(final_features), expected_len)
        padded_features[:actual_len] = final_features[:actual_len]
        return padded_features

    # Replace any remaining NaNs or Infs with 0 before returning
    final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)

    return final_features


# --- 文件处理 ---

def process_single_file(fpath, output_dir_features, fs=FS):
    """处理单个包含分段数据的 .mat 文件以提取特征"""
    fname = os.path.basename(fpath)
    print(f"  开始处理文件: {fname}")

    # <<< 修改 >>> 调整输出文件名的生成逻辑
    # 从输入文件名 '..._no_bandpass_baseline_corrected.mat'
    # 生成输出文件名 '..._no_bandpass_features.mat'
    base_name = fname.replace('_no_bandpass_baseline_corrected.mat', '')
    output_fname = f"{base_name}_no_bandpass_features.mat"
    output_fpath = os.path.join(output_dir_features, output_fname)

    try:
        mat_data = loadmat(fpath)
        # 检查所需的数据键是否存在
        if 'seg_X' not in mat_data or 'seg_y' not in mat_data:
             print(f"  跳过 {fname}: 未找到 'seg_X' 或 'seg_y' 键。")
             return

        seg_X = mat_data['seg_X']
        seg_y_raw = mat_data['seg_y']
        # 确保标签是一维数组
        seg_y = seg_y_raw.flatten()

        # 验证输入数据 seg_X 的维度和通道数
        if seg_X.ndim != 3:
            print(f"  跳过 {fname}: 期望三维数据 (segments, channels, timepoints)，但得到 {seg_X.ndim}D，形状为 {seg_X.shape}")
            return
        if seg_X.shape[1] != N_CHANNELS:
             print(f"  跳过 {fname}: 期望 {N_CHANNELS} 个通道，但得到 {seg_X.shape[1]} 个通道。")
             return
        if seg_X.shape[0] != len(seg_y):
             print(f"  警告 {fname}: 数据段数量 ({seg_X.shape[0]}) 与标签数量 ({len(seg_y)}) 不匹配。")
             # 可以选择跳过或继续处理较小的数量
             min_len = min(seg_X.shape[0], len(seg_y))
             seg_X = seg_X[:min_len, :, :]
             seg_y = seg_y[:min_len]
             if min_len == 0:
                 print(f"  跳过 {fname}: 数据或标签数量为零。")
                 return

        n_segments_in_file = seg_X.shape[0]
        if n_segments_in_file == 0:
            print(f"  跳过 {fname}，因为它不包含任何有效数据段。")
            return

        file_features_list = []
        file_labels_list = []

        print(f"    处理 {n_segments_in_file} 个数据段...")
        segments_processed = 0
        segments_skipped = 0
        for seg_idx in range(n_segments_in_file):
            if (seg_idx + 1) % 100 == 0 or seg_idx == 0 or seg_idx == n_segments_in_file - 1: # 调整打印频率
                 print(f"      正在处理数据段 {seg_idx + 1}/{n_segments_in_file}...")

            try:
                # 获取一个数据段的数据: (channels, timepoints)
                segment_data = seg_X[seg_idx, :, :]
                label = seg_y[seg_idx]

                # 提取特征
                features_vector = extract_features_for_segment(segment_data, fs=fs)

                # 检查特征提取是否成功 (extract_features_for_segment 内部已处理形状问题)
                # 检查是否有 NaN 或 Inf (虽然内部已替换，但可加额外检查)
                if np.any(np.isinf(features_vector)) or np.any(np.isnan(features_vector)):
                    print(f"    警告：文件 {fname} 中数据段 {seg_idx+1} 的特征包含 NaN/Inf，跳过该段。")
                    segments_skipped += 1
                    continue # 跳过此数据段

                file_features_list.append(features_vector)
                file_labels_list.append(label)
                segments_processed += 1

            except Exception as e:
                print(f"    处理文件 {fname} 中的数据段 {seg_idx + 1} 时出错: {e}")
                segments_skipped += 1
                continue # 跳到下一个数据段

        if file_features_list:
            try:
                # 将列表堆叠成最终的特征矩阵和标签向量
                features_matrix = np.vstack(file_features_list)
                labels_vector = np.array(file_labels_list, dtype=np.int32) # 确保标签是整数类型

                # 对特征矩阵形状进行最终检查
                expected_cols = N_CHANNELS * N_FEATURES_PER_CHANNEL
                if features_matrix.shape[1] != expected_cols:
                     print(f"  错误：文件 {fname} 的最终特征矩阵列数不正确 ({features_matrix.shape[1]}，期望 {expected_cols})。跳过保存。")
                     return
                if features_matrix.shape[0] != len(labels_vector):
                     print(f"  错误：文件 {fname} 的最终特征数量 ({features_matrix.shape[0]}) 与标签数量 ({len(labels_vector)}) 不匹配。跳过保存。")
                     return

                # 保存特征和标签到 .mat 文件
                savemat(output_fpath, {'features': features_matrix, 'labels': labels_vector}, do_compression=True)
                print(f"  成功处理 {segments_processed} 个数据段，跳过 {segments_skipped} 个。")
                print(f"  已将文件 {fname} 的特征 ({features_matrix.shape}) 保存到 {output_fname}")

            except Exception as e:
                print(f"  堆叠特征或保存文件 {output_fname} 时出错: {e}")
        else:
            print(f"  文件 {fname} 没有成功提取任何有效特征。跳过了 {segments_skipped} 个（或全部）数据段。")

    except FileNotFoundError:
        print(f"  错误：无法找到文件 {fpath}")
    except Exception as e:
        print(f"  加载或处理文件 {fname} 时发生意外错误: {e}")


# --- 主执行逻辑 ---

def run_minimal_processing(input_dir, output_dir_features):
    """主函数，遍历输入目录中的文件并处理它们"""
    start_time_total = time.time()
    print(f"开始特征提取过程...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir_features}")

    if not os.path.exists(output_dir_features):
        try:
            print(f"创建输出目录: {output_dir_features}")
            os.makedirs(output_dir_features)
        except OSError as e:
            print(f"错误：无法创建输出目录 '{output_dir_features}': {e}")
            return # 无法创建输出目录，退出

    try:
        # <<< 修改 >>> 更新文件后缀以匹配文件 2 的输出
        target_suffix = '_no_bandpass_baseline_corrected.mat'
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        mat_files = sorted([f for f in all_files if f.endswith(target_suffix)])

    except FileNotFoundError:
        print(f"错误：输入目录未找到: '{input_dir}'")
        return
    except Exception as e:
        print(f"列出输入目录 '{input_dir}' 中的文件时出错: {e}")
        return

    if not mat_files:
        # <<< 修改 >>> 更新未找到文件时的错误消息
        print(f"错误：在 '{input_dir}' 中未找到以 '{target_suffix}' 结尾的 .mat 文件。")
        print("请确保 processing 脚本已成功运行并将输出文件放在此目录中。")
        return

    total_files = len(mat_files)
    print(f"\n在 '{input_dir}' 中找到 {total_files} 个待处理文件。")
    print("-" * 30)

    files_processed_successfully = 0
    files_failed = 0
    for i, fname in enumerate(mat_files):
        print(f"\n处理文件 {i+1}/{total_files}: {fname}")
        fpath = os.path.join(input_dir, fname)
        try:
            process_single_file(fpath, output_dir_features, fs=FS)
            files_processed_successfully += 1
        except Exception as e:
            # 捕获 process_single_file 中未处理的意外错误
            print(f"!! 处理文件 {fname} 时发生顶层错误: {e}")
            files_failed += 1
        print("-" * 30)


    elapsed_total = time.time() - start_time_total
    print("\n" + "=" * 40)
    print(f"特征提取完成。")
    print(f"总计处理文件: {total_files}")
    print(f"  成功: {files_processed_successfully}")
    print(f"  失败/跳过: {files_failed + (total_files - files_processed_successfully - files_failed)}") # 修正计数逻辑
    print(f"总耗时: {elapsed_total:.2f} 秒 ({elapsed_total/60:.2f} 分钟)。")
    print(f"特征文件已保存到: {output_dir_features}")
    print("=" * 40)

if __name__ == "__main__":
    # 运行处理流程
    run_minimal_processing(INPUT_DIR, OUTPUT_DIR_FEATURES)
    print("\n脚本执行完毕。")