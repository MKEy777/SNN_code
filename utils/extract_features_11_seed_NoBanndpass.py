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

INPUT_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\PerSession_MAT_NoBandpass_Fixed"
OUTPUT_DIR_FEATURES = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features_NoBandpass_Fixed"
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
    f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
    return np.sum(Pxx) if Pxx is not None and len(Pxx) > 0 else 0.0

def calculate_max_psd(data, fs=FS):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return 0.0, 0.0
    nperseg = min(256, len(valid_data))
    f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
    if Pxx is None or len(Pxx) == 0: return 0.0, 0.0
    max_idx = np.argmax(Pxx)
    return Pxx[max_idx], f[max_idx]

def calculate_hjorth_mobility(data):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return 0.0
    var_data = np.var(valid_data)
    if var_data < 1e-10: return 0.0
    diff1 = np.diff(valid_data)
    if len(diff1) < 2: return 0.0
    var_diff1 = np.var(diff1)
    ratio = var_diff1 / var_data
    return np.sqrt(max(0, ratio))

def calculate_hjorth_complexity(data):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 4: return 0.0
    mobility_data = calculate_hjorth_mobility(valid_data)
    if mobility_data < 1e-10: return 0.0
    diff1 = np.diff(valid_data)
    if len(diff1) < 3: return 0.0
    var_diff1 = np.var(diff1)
    if var_diff1 < 1e-10: return 0.0
    diff2 = np.diff(diff1)
    if len(diff2) < 2: return 0.0
    var_diff2 = np.var(diff2)
    ratio_diff = var_diff2 / var_diff1
    mobility_diff1 = np.sqrt(max(0, ratio_diff))
    return mobility_diff1 / mobility_data

def calculate_shannon_entropy(data, bins=10):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0 or np.ptp(valid_data) < 1e-10: return 0.0
    hist, bin_edges = np.histogram(valid_data, bins=max(1, int(bins)), density=False)
    counts = hist[hist > 0]
    if len(counts) == 0: return 0.0
    probs = counts / len(valid_data)
    return -np.sum(probs * np.log2(probs[probs > 1e-10]))

def calculate_c0_complexity(data, fs=FS):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2 or np.ptp(valid_data) < 1e-10: return 0.0
    nperseg = min(256, len(valid_data))
    f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
    if Pxx is None or len(Pxx) == 0 or np.all(Pxx < 1e-10): return 0.0
    max_idx = np.argmax(Pxx)
    max_power = Pxx[max_idx]
    if max_power < 1e-10: return 0.0
    half_max = max_power / 2.0
    indices_above_half = np.where(Pxx > half_max)[0]
    if len(indices_above_half) == 0: return 0.0
    return f[indices_above_half[-1]] - f[indices_above_half[0]]

def calculate_power_spectrum_entropy(data, fs=FS):
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2 or np.ptp(valid_data) < 1e-10: return 0.0
    nperseg = min(256, len(valid_data))
    f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
    if Pxx is None or len(Pxx) == 0: return 0.0
    Pxx_filtered = Pxx[Pxx > 1e-10]
    if len(Pxx_filtered) == 0: return 0.0
    Pxx_sum = np.sum(Pxx_filtered)
    if Pxx_sum < 1e-10: return 0.0
    Pxx_norm = Pxx_filtered / Pxx_sum
    Pxx_norm = Pxx_norm[Pxx_norm > 1e-10]
    return -np.sum(Pxx_norm * np.log(Pxx_norm)) if len(Pxx_norm) > 0 else 0.0

# --- 特征提取 (Minimal) ---
def extract_features_for_segment(segment_data, fs=FS):
    """
    为单个数据段提取特征。
    输入 segment_data 的形状: (channels, window_len)
    输出形状: (channels * N_FEATURES_PER_CHANNEL,)
    """
    all_segment_features = []
    # 期望形状为 (channels, window_len)
    if segment_data.ndim != 2:
         print(f"错误：extract_features_for_segment 期望二维数组，但接收到形状 {segment_data.shape}")
         # 返回全零数组或进行其他适当的错误处理
         return np.zeros(N_CHANNELS * N_FEATURES_PER_CHANNEL, dtype=np.float32)

    channels, window_len = segment_data.shape

    for c in range(channels):
        ts = segment_data[c, :] # 直接访问通道数据
        # 对完全无效输入的检查已被替换为假设输入有效
        # 或者让后续计算处理可能出现的 NaN 值
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

        # 不检查 NaN/Inf，假设结果是有限的，或者由 np.array 转换处理
        all_segment_features.extend(features)

    # 不进行最终长度检查
    # 假设期望的最终类型是 float32
    # 期望长度: N_CHANNELS * N_FEATURES_PER_CHANNEL
    final_features = np.array(all_segment_features, dtype=np.float32)
    expected_len = N_CHANNELS * N_FEATURES_PER_CHANNEL
    if len(final_features) != expected_len:
        print(f"警告：特征向量长度不匹配。期望 {expected_len}，得到 {len(final_features)}。进行填充/截断。")
        # 如果长度错误，用零填充或截断
        padded_features = np.zeros(expected_len, dtype=np.float32)
        actual_len = min(len(final_features), expected_len)
        padded_features[:actual_len] = final_features[:actual_len]
        return padded_features

    return final_features


# --- 文件处理 (Minimal) ---

def process_single_file(fpath, output_dir_features, fs=FS):
    fname = os.path.basename(fpath)

    output_fname = os.path.splitext(fname)[0].replace('_no_bandpass', '') + "_no_bandpass_features.mat"
    output_fpath = os.path.join(output_dir_features, output_fname)

    try: # 添加 try-except 块以增强鲁棒性
        mat_data = loadmat(fpath)
        if 'seg_X' not in mat_data or 'seg_y' not in mat_data:
             print(f"  跳过 {fname}: 未找到 'seg_X' 或 'seg_y' 键。")
             return
        seg_X = mat_data['seg_X']
        seg_y_raw = mat_data['seg_y']
        seg_y = seg_y_raw.flatten()

        # <<< 修改 >>> 检查三维形状: (segments, channels, timepoints)
        if seg_X.ndim != 3:
            print(f"  跳过 {fname}: 期望三维数据 (segments, channels, timepoints)，但得到 {seg_X.ndim}D，形状为 {seg_X.shape}")
            return
        if seg_X.shape[1] != N_CHANNELS:
             print(f"  跳过 {fname}: 期望 {N_CHANNELS} 个通道，但得到 {seg_X.shape[1]}")
             return

        n_segments_in_file = seg_X.shape[0]
        if n_segments_in_file == 0:
            print(f"  跳过 {fname}，因为它不包含任何数据段。")
            return

        file_features_list = []
        file_labels_list = []

        print(f"  正在处理 {n_segments_in_file} 个数据段...")
        for seg_idx in range(n_segments_in_file):
            if (seg_idx + 1) % 50 == 0 or seg_idx == 0 or seg_idx == n_segments_in_file - 1: # 调整了打印频率
                 print(f"    正在处理数据段 {seg_idx + 1}/{n_segments_in_file}...")

            try: # 为处理单个数据段添加 try-except
                # <<< 修改 >>> 索引假设形状为 (segments, channels, timepoints)
                segment_data = seg_X[seg_idx, :, :] # 获取一个数据段的数据: (channels, timepoints)
                label = seg_y[seg_idx]

                features_vector = extract_features_for_segment(segment_data, fs=fs)

                if features_vector is not None: # 检查特征提取是否成功
                     file_features_list.append(features_vector)
                     file_labels_list.append(label)
                else:
                     print(f"    警告：未能为文件 {fname} 中的数据段 {seg_idx+1} 提取特征。")

            except Exception as e:
                print(f"    处理文件 {fname} 中的数据段 {seg_idx + 1} 时出错: {e}")
                continue # 跳过到下一个数据段

        if file_features_list:
            try: # 为堆叠特征和保存文件添加 try-except
                features_matrix = np.vstack(file_features_list)
                labels_vector = np.array(file_labels_list, dtype=np.int32)

                # 对矩阵形状进行最终检查
                expected_cols = N_CHANNELS * N_FEATURES_PER_CHANNEL
                if features_matrix.shape[1] != expected_cols:
                     print(f"  错误：文件 {fname} 的最终特征矩阵列数不正确 ({features_matrix.shape[1]}，期望 {expected_cols})。跳过保存。")
                     return

                savemat(output_fpath, {'features': features_matrix, 'labels': labels_vector}, do_compression=True)
                print(f"  已将文件 {fname} 的特征 ({features_matrix.shape}) 保存到 {output_fname}")

            except Exception as e:
                print(f"  堆叠特征或保存文件 {output_fname} 时出错: {e}")
        else:
            print(f"  文件 {fname} 没有提取或保存有效的特征。")

    except Exception as e:
        print(f"加载或处理文件 {fname} 时出错: {e}")


# --- 主执行逻辑 (Minimal) ---

def run_minimal_processing(input_dir, output_dir_features):
    start_time_total = time.time()

    if not os.path.exists(output_dir_features):
        print(f"创建输出目录: {output_dir_features}")
        os.makedirs(output_dir_features)

    try: # 为列出文件添加 try-except
        # <<< 修改 >>> 现在查找 '_no_bandpass.mat' 文件
        mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_no_bandpass.mat') and os.path.isfile(os.path.join(input_dir, f))])
    except FileNotFoundError:
        print(f"错误：输入目录未找到: '{input_dir}'")
        return
    except Exception as e:
        print(f"列出输入目录 '{input_dir}' 中的文件时出错: {e}")
        return


    if not mat_files:
        print(f"错误：在 '{input_dir}' 中未找到 '*_no_bandpass.mat' 文件。") # 调整了错误消息
        return

    total_files = len(mat_files)
    print(f"在 '{input_dir}' 中找到 {total_files} 个待处理文件。正在处理...")

    for i, fname in enumerate(mat_files):
        print(f"正在处理文件 {i+1}/{total_files}: {fname}")
        fpath = os.path.join(input_dir, fname)
        process_single_file(fpath, output_dir_features, fs=FS)

    elapsed_total = time.time() - start_time_total
    print(f"\n处理完成 {total_files} 个文件，耗时 {elapsed_total:.2f} 秒。")
    print(f"输出已保存到: {output_dir_features}")

if __name__ == "__main__":
    run_minimal_processing(INPUT_DIR, OUTPUT_DIR_FEATURES)
    print("脚本执行完毕。")