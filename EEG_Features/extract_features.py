import numpy as np
from scipy import signal
from scipy.linalg import svd
from scipy.io import loadmat, savemat
from itertools import permutations
import warnings
import os
import time
import nolds
import traceback # 导入 traceback 模块用于打印详细错误信息
import multiprocessing # 导入 multiprocessing 库
from functools import partial # 导入 partial 用于简化 map 参数传递

# 尝试导入 EntropyHub，如果未安装/找到，则处理潜在的导入错误
try:
    # Note: Ensure EntropyHub is correctly installed in your environment
    # pip install EntropyHub
    import EntropyHub as eh
    entropy_hub_available = True
except ImportError:
    entropy_hub_available = False
    print("警告：未找到 EntropyHub 库。Kolmogorov Entropy 特征将返回 NaN。")
    print("请尝试安装： pip install EntropyHub")

# --- 全局参数 ---
# !! 修改为你实际的输入和输出路径 !!
INPUT_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\PerSession_MAT" # 输入 MAT 文件目录
OUTPUT_DIR_FEATURES = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features" # 输出特征文件的目录

FS = 200 # 采样频率 (Hz)
N_FEATURES_PER_COMBINATION = 17 # 每个频带/通道组合提取的特征数量
N_BANDS = 4 # 频带数量 (例如 delta, theta, alpha, beta)
N_CHANNELS = 62 # EEG 通道数量
EXPECTED_TOTAL_FEATURES = N_BANDS * N_CHANNELS * N_FEATURES_PER_COMBINATION  # 预期总特征数 4 * 62 * 17 = 4216

# --- 特征提取函数 ---
# (包括 calculate_mean_ptp 到 calculate_power_spectrum_entropy)
# (这些函数与之前版本相同，这里为完整性再次包含)

def calculate_mean_ptp(data, sub_window_size=FS):
    """计算子窗口的平均峰峰值 (Peak-to-Peak)，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan
    valid_data = data[np.isfinite(data)] # 仅使用有限（非 NaN, 非 Inf）数据
    if len(valid_data) == 0: return np.nan

    if sub_window_size <= 0 or not isinstance(sub_window_size, int):
        sub_window_size = FS # 使用默认采样频率作为窗口大小
    sub_window_size = min(sub_window_size, len(valid_data))
    if sub_window_size <= 1:
         return np.ptp(valid_data) if len(valid_data) >= 2 else 0.0

    ptp_values = []
    for i in range(0, len(valid_data) - sub_window_size + 1):
        sub_window = valid_data[i : i + sub_window_size]
        if len(sub_window) >= 2:
            ptp_val = np.ptp(sub_window)
            if not np.isnan(ptp_val):
                ptp_values.append(ptp_val)

    if not ptp_values:
        return np.ptp(valid_data) if len(valid_data) >= 2 else 0.0
    return np.mean(ptp_values)

def calculate_msv(data):
    """计算均方值 (Mean Square Value)，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0: return np.nan
    return np.mean(np.square(valid_data))

def calculate_var(data):
    """计算方差 (Variance)，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return 0.0
    return np.var(valid_data)

def calculate_psd_sum(data, fs=FS):
    """使用 Welch 方法计算功率谱密度 (PSD) 总和，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return np.nan
    nperseg = min(256, len(valid_data))
    if nperseg == 0: return np.nan
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
        if Pxx is None or len(Pxx) == 0: return np.nan
        return np.sum(Pxx)
    except ValueError:
        return np.nan # Handle cases like constant input after NaN removal

def calculate_max_psd(data, fs=FS):
    """使用 Welch 方法计算最大 PSD 值及其对应的频率，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan, np.nan
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return np.nan, np.nan
    nperseg = min(256, len(valid_data))
    if nperseg == 0: return np.nan, np.nan
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
        if Pxx is None or len(Pxx) == 0: return np.nan, np.nan
        max_idx = np.argmax(Pxx)
        # Ensure index is valid for both Pxx and f
        if max_idx < len(Pxx) and max_idx < len(f):
             return Pxx[max_idx], f[max_idx]
        else:
             # This case should theoretically not happen if Pxx is not empty
             return np.nan, np.nan
    except ValueError:
        return np.nan, np.nan

def calculate_hjorth_mobility(data):
    """计算 Hjorth 移动性 (Mobility)，处理 NaNs/Infs 和零方差。"""
    if data is None or len(data) < 2: return np.nan
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return np.nan
    var_data = np.var(valid_data)
    # Handle constant signal
    if var_data < 1e-10: return 0.0
    # Use np.gradient for potentially more stable derivative estimate? Or stick to diff.
    diff1 = np.diff(valid_data)
    # Need variance of diff, requires at least 2 points in diff1 -> 3 points in valid_data
    if len(diff1) < 2: return np.nan # Need at least 3 points in original data for var(diff1)
    var_diff1 = np.var(diff1)
    # Ratio can be negative due to precision, ensure it's non-negative
    ratio = var_diff1 / var_data
    return np.sqrt(max(0, ratio))

def calculate_hjorth_complexity(data):
    """计算 Hjorth 复杂度 (Complexity)，处理 NaNs/Infs 和零方差。"""
    if data is None or len(data) < 3: return np.nan # Need 3 points for first diff var
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 3: return np.nan

    mobility_data = calculate_hjorth_mobility(valid_data)
    # Handle NaN or zero mobility
    if np.isnan(mobility_data) or mobility_data < 1e-10: return 0.0

    diff1 = np.diff(valid_data)
    # Need variance of diff1 -> requires len(diff1) >= 2 -> len(valid_data) >= 3
    if len(diff1) < 2: return np.nan # Should be caught by initial length check

    var_diff1 = np.var(diff1)
    # Handle constant first difference
    if var_diff1 < 1e-10: return 0.0

    # Need variance of diff2 -> requires len(diff2) >= 2 -> len(diff1) >= 3 -> len(valid_data) >= 4
    if len(valid_data) < 4: return np.nan # Need 4 points for complexity

    diff2 = np.diff(diff1)
    if len(diff2) < 2: return np.nan # Should be caught by length check

    var_diff2 = np.var(diff2)
    # Mobility of the first difference
    mobility_diff1 = np.sqrt(max(0, var_diff2 / var_diff1))

    # Complexity = Mobility(diff1) / Mobility(data)
    return mobility_diff1 / mobility_data

def calculate_shannon_entropy(data, bins=10):
    """基于直方图计算香农熵 (Shannon Entropy)，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0: return np.nan
    # Handle constant signal
    if np.ptp(valid_data) < 1e-10: return 0.0
    try:
        # Ensure bins is positive integer
        bins = max(1, int(bins))
        hist, bin_edges = np.histogram(valid_data, bins=bins, density=False)
        counts = hist[hist > 0]
        if len(counts) == 0: return 0.0 # Should only happen if valid_data is empty
        total_count = len(valid_data) # Use total count, not sum(counts)
        if total_count == 0: return 0.0
        probs = counts / total_count
        # Use log base 2 for bits
        return -np.sum(probs * np.log2(probs))
    except ValueError: # Catches issues in histogram (e.g., bad range if data is weird)
        return np.nan

def calculate_approx_entropy(data, m=2, r_coeff=0.2):
    """计算近似熵 (Approximate Entropy, ApEn)，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan
    valid_data = data[np.isfinite(data)]
    N = len(valid_data)
    # Ensure m is positive integer
    m = max(1, int(m))
    if N < m + 1: return np.nan # Need N >= m+1 for phi(m+1) calculation

    std_dev = np.std(valid_data)
    # Handle constant signal
    if std_dev < 1e-10: return 0.0
    # Ensure r_coeff is positive
    r_coeff = max(1e-10, r_coeff)
    r = r_coeff * std_dev

    def _phi(m_dim, N_local, r_local, data_local):
        # Create embedded vectors efficiently
        shape = (N_local - m_dim + 1, m_dim)
        strides = (data_local.strides[0], data_local.strides[0])
        try:
             # Ensure data is contiguous for stride tricks
             if not data_local.flags['C_CONTIGUOUS']:
                 data_local = np.ascontiguousarray(data_local)
             X = np.lib.stride_tricks.as_strided(data_local, shape=shape, strides=strides)
        except ValueError:
             # Fallback if stride tricks fail
             X = np.array([data_local[i:i + m_dim] for i in range(N_local - m_dim + 1)])
             if X.shape[0] != shape[0] or X.shape[1] != shape[1]:
                 return -np.inf # Cannot form sequences

        n_seq = X.shape[0]
        if n_seq == 0: return -np.inf

        # Calculate Chebyshev distance matrix (can be memory intensive)
        # Consider using libraries like scipy.spatial.distance.cdist if memory becomes an issue
        try:
            # Broadcasting approach
            dist = np.max(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2)
        except MemoryError:
            # Fallback to loop if memory is insufficient (much slower)
            dist = np.zeros((n_seq, n_seq))
            for i in range(n_seq):
                for j in range(i, n_seq): # Optimize by calculating only upper triangle
                    d = np.max(np.abs(X[i] - X[j]))
                    dist[i, j] = d
                    dist[j, i] = d

        # Count pairs within distance r
        # Avoid log(0) by adding small epsilon or filtering zeros
        C_i = np.sum(dist <= r_local, axis=1) / n_seq
        C_i_filtered = C_i[C_i > 1e-10] # Filter zeros
        if len(C_i_filtered) == 0: return -np.inf

        # Return mean of log(C_i)
        return np.mean(np.log(C_i_filtered))

    # Ensure data is contiguous array for stride tricks within _phi
    valid_data_cont = np.ascontiguousarray(valid_data)

    phi_m = _phi(m, N, r, valid_data_cont)
    phi_m_plus_1 = _phi(m + 1, N, r, valid_data_cont)

    if np.isinf(phi_m) or np.isinf(phi_m_plus_1): return np.nan

    apen = phi_m - phi_m_plus_1
    # ApEn should be non-negative
    return max(0, apen)

def calculate_c0_complexity(data, fs=FS):
    """计算 C0 复杂度 (功率谱半峰全宽 FWHM)，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return np.nan
    # Handle constant signal separately (zero bandwidth)
    if np.ptp(valid_data) < 1e-10: return 0.0

    nperseg = min(256, len(valid_data))
    if nperseg == 0: return np.nan
    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
        if Pxx is None or len(Pxx) == 0 or np.all(Pxx < 1e-10): return 0.0

        max_idx = np.argmax(Pxx)
        max_power = Pxx[max_idx]
        if max_power < 1e-10: return 0.0 # Should be caught by np.all check

        half_max = max_power / 2.0

        # Find indices where power is less than or equal to half max
        indices_below_half = np.where(Pxx <= half_max)[0]

        # Find left frequency
        left_indices = indices_below_half[indices_below_half < max_idx]
        f_left = f[left_indices[-1]] if len(left_indices) > 0 else f[0]

        # Find right frequency
        right_indices = indices_below_half[indices_below_half > max_idx]
        f_right = f[right_indices[0]] if len(right_indices) > 0 else f[-1]

        bandwidth = f_right - f_left
        return max(0, bandwidth) # Ensure non-negative
    except (ValueError, IndexError):
        # Catch errors during Welch or indexing
        return np.nan

def calculate_correlation_dimension(data, emb_dim=10):
    """使用 nolds 库计算关联维数 (Correlation Dimension)，处理 NaNs/Infs。"""
    # Note: nolds installation required: pip install nolds
    if data is None: return np.nan
    valid_data = data[np.isfinite(data)]
    # Ensure emb_dim is reasonable
    emb_dim = max(1, int(emb_dim))
    # Check length requirement based on nolds documentation/common practice
    if len(valid_data) < emb_dim * 10: return np.nan
    # Handle constant signal
    if np.ptp(valid_data) < 1e-10: return 0.0
    try:
        # Consider setting r_min, r_max, etc. if default range is problematic
        corr_dim_val = nolds.corr_dim(valid_data, emb_dim=emb_dim, debug=False)
        # Check for NaN/Inf result from nolds
        return corr_dim_val if np.isfinite(corr_dim_val) else np.nan
    except Exception as e:
        # print(f"DEBUG: nolds.corr_dim failed: {e}") # Uncomment for debugging
        return np.nan

def calculate_lyapunov_exponent(data, emb_dim=10, lag=None, min_tsep=None, tau=1, min_neighbors=20):
    """使用 nolds 库计算最大李雅普诺夫指数 (Largest Lyapunov Exponent)，处理 NaNs/Infs。"""
    # Note: nolds installation required: pip install nolds
    if data is None: return np.nan
    valid_data = data[np.isfinite(data)]
    N = len(valid_data)
    # Ensure emb_dim is reasonable
    emb_dim = max(1, int(emb_dim))
    # Check length requirement
    if N < emb_dim * 15: return np.nan
    # Handle constant signal
    if np.ptp(valid_data) < 1e-10: return 0.0

    # Estimate lag if not provided
    if lag is None:
        try:
            detrended_data = valid_data - np.mean(valid_data)
            if np.var(detrended_data) < 1e-10: raise ValueError("Variance is zero after detrending.")
            autocorr = np.correlate(detrended_data, detrended_data, mode='full')
            autocorr = autocorr[N-1:] # Take positive lags
            if autocorr[0] < 1e-10: raise ValueError("Autocorrelation at lag 0 is zero.")
            autocorr /= autocorr[0] # Normalize

            # Find first zero crossing or first drop below 1/e
            zero_crossings = np.where(np.diff(np.signbit(autocorr)))[0] # More robust sign check
            if len(zero_crossings) > 0:
                lag = zero_crossings[0] + 1
            else:
                lag_1_e = np.where(autocorr < 1/np.e)[0]
                if len(lag_1_e) > 0:
                    lag = lag_1_e[0] + 1 # Index + 1 for lag
                else:
                    lag = max(1, int(0.1 * emb_dim)) # Heuristic fallback
            # Ensure lag is reasonable relative to embedding dimension and length
            max_possible_lag = (N - 1) // (emb_dim if emb_dim > 0 else 1)
            lag = min(lag, max(1, max_possible_lag if max_possible_lag > 0 else 1))
            lag = max(lag, 1) # Final check: lag >= 1
        except Exception as lag_err:
            # print(f"DEBUG: Lag estimation failed: {lag_err}") # Uncomment for debugging
            lag = max(1, int(0.1*emb_dim) if emb_dim > 0 else 1)
            lag = max(lag, 1)

    # Estimate min_tsep if not provided
    if min_tsep is None:
        min_tsep = lag * emb_dim # Common heuristic

    try:
        # Validate parameters before passing to nolds
        lag = max(1, int(round(lag)))
        min_tsep = max(0, int(round(min_tsep))) # min_tsep can be 0 in nolds
        min_neighbors = max(1, int(min_neighbors))
        tau = max(1, int(tau))

        # Check sufficient length for nolds algorithm
        required_len = (emb_dim - 1) * lag + min_tsep + 1
        if N < required_len:
             # print(f"DEBUG: Data length {N} insufficient for lyap_r (needs {required_len}).")
             return np.nan

        # Call nolds function
        lyap_r_val = nolds.lyap_r(valid_data, emb_dim=emb_dim, lag=lag,
                                  min_tsep=min_tsep, tau=tau,
                                  min_neighbors=min_neighbors,
                                  debug=False, fit='poly') # Use polynomial fit
        # Check result validity
        return lyap_r_val if np.isfinite(lyap_r_val) else np.nan
    except Exception as e:
        # print(f"DEBUG: nolds.lyap_r failed: {e}") # Uncomment for debugging
        return np.nan

def calculate_kolmogorov_entropy(data, m=2, tau=1, r=None):
    """使用 EntropyHub 库计算 K2 熵 (Kolmogorov Entropy 估计值)，处理 NaNs/Infs。"""
    global entropy_hub_available
    if not entropy_hub_available: return np.nan
    if data is None: return np.nan
    valid_data = data[np.isfinite(data)]
    # Ensure m and tau are positive integers
    m = max(1, int(m))
    tau = max(1, int(tau))
    # Check length requirement (K2En usually needs significant data)
    if len(valid_data) < (m + 1) * tau * 10:
        return np.nan
    # Handle constant signal
    if np.ptp(valid_data) < 1e-10: return 0.0

    # Estimate radius r if not provided
    if r is None:
        std_dev = np.std(valid_data)
        if std_dev < 1e-9: return 0.0 # Already checked by ptp, but safe
        r = 0.2 * std_dev

    # Validate radius r
    if not np.isfinite(r) or r <= 0:
        # print(f"DEBUG: Invalid radius r={r} for K2En.") # Uncomment for debugging
        return np.nan

    try:
        # Ensure data is in list format for EntropyHub
        if isinstance(valid_data, np.ndarray):
             valid_data_list = valid_data.flatten().tolist()
        else:
             valid_data_list = list(valid_data) # Assume iterable

        # Call EntropyHub K2En
        # Note: K2En can be computationally expensive
        k2_results = eh.K2En(valid_data_list, m=m, tau=tau, r=r)
        # K2En returns (K2En, Av_Rad) - we need the first value
        k2_entropy = k2_results[0]

        # Validate result
        if np.isfinite(k2_entropy) and k2_entropy >= 0:
            return k2_entropy
        else:
            # print(f"DEBUG: K2En returned invalid value: {k2_entropy}") # Uncomment for debugging
            return np.nan
    except Exception as e:
        # print(f"DEBUG: EntropyHub K2En failed: {e}") # Uncomment for debugging
        return np.nan

def calculate_permutation_entropy(data, m=3, delay=1):
    """计算排列熵 (Permutation Entropy, PE)，处理 NaNs/Infs。"""
    if data is None: return np.nan
    valid_data = data[np.isfinite(data)]
    N = len(valid_data)
    # Ensure m and delay are positive integers
    m = max(1, int(m))
    delay = max(1, int(delay))
    # Calculate required length for one pattern
    required_length = (m - 1) * delay + 1
    if N < required_length: return np.nan
    # Handle constant signal (entropy is 0)
    if np.ptp(valid_data) < 1e-10: return 0.0

    # Use sliding window view for efficiency if possible
    try:
        shape = (N - required_length + 1, m)
        strides = (valid_data.strides[0] * delay, valid_data.strides[0]) # Stride by delay then by 1
        # Ensure data is contiguous for stride tricks
        if not valid_data.flags['C_CONTIGUOUS']:
            valid_data = np.ascontiguousarray(valid_data)
        # Create overlapping sequences using strides
        sequences = np.lib.stride_tricks.as_strided(valid_data[::delay],
                                                   shape=shape,
                                                   strides=(valid_data.strides[0] * delay, valid_data.strides[0]))
        sequences = valid_data[:N - (m - 1) * delay].reshape(-1, 1) + np.arange(m) * delay * valid_data.strides[0]

        # Trying a different stride approach
        itemsize = valid_data.itemsize
        sequences = np.lib.stride_tricks.as_strided(valid_data,
                                      shape=(N - (m - 1) * delay, m),
                                      strides=(itemsize, itemsize * delay))

    except ValueError:
        # Fallback to list comprehension if stride tricks fail
        sequences = np.array([valid_data[i:i + required_length:delay] for i in range(N - required_length + 1)])
        # Verify shape after fallback
        if sequences.ndim != 2 or sequences.shape[1] != m:
            return np.nan # Could not form sequences correctly

    num_sequences = sequences.shape[0]
    if num_sequences == 0: return 0.0

    # Get permutations (ordinal patterns)
    # This argsort step can be a bottleneck
    try:
        perms = np.argsort(sequences, axis=1)
        # Convert permutations to tuples or strings to be hashable for counting
        # Using view casting for potential speedup if arrays are simple types
        hashable_perms = perms.view(f'|S{perms.shape[1] * perms.itemsize}')
    except TypeError:
        # Fallback if view casting fails
        hashable_perms = [tuple(p) for p in perms]


    # Count occurrences of each pattern
    counts = {}
    for p_hash in hashable_perms:
        counts[p_hash] = counts.get(p_hash, 0) + 1

    # Calculate probabilities
    probs = np.array(list(counts.values()), dtype=float) / num_sequences
    # Filter zero probabilities (shouldn't happen if counts > 0)
    probs = probs[probs > 1e-10]
    if len(probs) == 0: return 0.0

    # Calculate entropy (using natural log)
    pe = -np.sum(probs * np.log(probs))

    # Optional: Normalize by log(m!)
    # from math import factorial, log
    # max_entropy = log(factorial(m)) if m > 1 else 0
    # if max_entropy > 1e-10:
    #     pe /= max_entropy

    return pe

def calculate_singular_spectrum_entropy(data, window_size=None):
    """计算奇异谱熵 (Singular Spectrum Entropy, SSE)，处理 NaNs/Infs。"""
    if data is None: return np.nan
    valid_data = data[np.isfinite(data)]
    N = len(valid_data)
    if N < 2: return np.nan # Need at least 2 points
    # Handle constant signal
    if np.ptp(valid_data) < 1e-10: return 0.0 # Entropy is 0

    # Determine window size (embedding dimension L)
    if window_size is None:
        window_size = N // 2 # Common choice
    window_size = max(2, min(int(round(window_size)), N - 1)) # Ensure 2 <= L <= N-1

    # Calculate number of lagged vectors (K)
    K = N - window_size + 1
    if K < 1: return np.nan # Should not happen if L <= N-1

    # Create trajectory matrix using stride tricks
    try:
        shape = (K, window_size)
        strides = (valid_data.itemsize, valid_data.itemsize) # Stride by 1 element in both dims
        # Ensure data is contiguous
        if not valid_data.flags['C_CONTIGUOUS']:
            valid_data = np.ascontiguousarray(valid_data)
        X = np.lib.stride_tricks.as_strided(valid_data, shape=shape, strides=strides)
    except ValueError as stride_err:
        # print(f"DEBUG: Stride error in SSE: {stride_err}") # Uncomment for debugging
        return np.nan

    # Perform SVD - only need singular values (s)
    try:
        s = svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.nan # SVD failed

    # Filter small/zero singular values and normalize
    s = s[s > 1e-10] # Tolerance for floating point precision
    if len(s) == 0: return 0.0

    s_sum = np.sum(s)
    if s_sum < 1e-10: return 0.0 # Should be covered by filtering s, but check sum

    norm_s = s / s_sum
    # Filter again after normalization for safety
    norm_s = norm_s[norm_s > 1e-10]
    if len(norm_s) == 0: return 0.0

    # Calculate entropy (using natural log)
    sse = -np.sum(norm_s * np.log(norm_s))
    return sse

def calculate_power_spectrum_entropy(data, fs=FS):
    """使用 Welch 方法计算功率谱熵 (Power Spectrum Entropy, PSE)，处理 NaNs/Infs。"""
    if data is None or len(data) == 0: return np.nan
    valid_data = data[np.isfinite(data)]
    if len(valid_data) < 2: return np.nan
    # Handle constant signal
    if np.ptp(valid_data) < 1e-10: return 0.0 # PSD is a spike, entropy near 0

    # Use appropriate segment length for Welch
    nperseg = min(256, len(valid_data))
    if nperseg < 2: return np.nan # Welch needs at least 2 points per segment

    try:
        f, Pxx = signal.welch(valid_data, fs=fs, nperseg=nperseg, detrend='linear')
        # Check for valid PSD output
        if Pxx is None or len(Pxx) == 0 : return np.nan

        # Filter near-zero power values before normalization
        Pxx = Pxx[Pxx > 1e-10] # Tolerance
        if len(Pxx) == 0: return 0.0

        # Normalize to get probability distribution
        Pxx_sum = np.sum(Pxx)
        # Check sum before division
        if Pxx_sum < 1e-10: return 0.0

        Pxx_norm = Pxx / Pxx_sum
        # Filter again after normalization if needed (shouldn't be necessary)
        Pxx_norm = Pxx_norm[Pxx_norm > 1e-10]
        if len(Pxx_norm) == 0: return 0.0

        # Calculate entropy (using natural log)
        pse = -np.sum(Pxx_norm * np.log(Pxx_norm))
        return pse
    except ValueError:
        # Catch Welch errors (e.g., if segment length is too small after NaN removal)
        return np.nan

# --- 单个数据段特征提取主函数 ---
def extract_features_for_segment(segment_data, fs=FS):
    """为单个数据段（包含所有频带和通道）提取所有特征。"""
    all_segment_features = []
    # Validate input shape
    if not isinstance(segment_data, np.ndarray) or segment_data.ndim != 3 or \
       segment_data.shape[0] != N_BANDS or segment_data.shape[1] != N_CHANNELS:
        # print(f"DEBUG: Invalid segment shape: {segment_data.shape}") # Debugging
        return None

    bands, channels, window_len = segment_data.shape

    # Iterate through bands and channels
    for b in range(bands):
        for c in range(channels):
            ts = segment_data[b, c, :] # Time series for this band/channel

            # Check if timeseries is valid (not all NaN/Inf)
            if np.all(np.isnan(ts)) or not np.any(np.isfinite(ts)):
                # Use zeros as placeholders if data is invalid
                features = [0.0] * N_FEATURES_PER_COMBINATION
            else:
                # Calculate all 17 features for the valid timeseries
                # Encapsulate in try-except for robustness? Overhead concern.
                try:
                    features = [
                        calculate_mean_ptp(ts, fs=fs), # Pass fs if needed by func
                        calculate_msv(ts),
                        calculate_var(ts),
                        calculate_psd_sum(ts, fs=fs),
                        *calculate_max_psd(ts, fs=fs), # Unpack max_psd, freq_at_max
                        calculate_hjorth_mobility(ts),
                        calculate_hjorth_complexity(ts),
                        calculate_shannon_entropy(ts),
                        calculate_approx_entropy(ts),
                        calculate_c0_complexity(ts, fs=fs),
                        calculate_correlation_dimension(ts),
                        calculate_lyapunov_exponent(ts),
                        calculate_kolmogorov_entropy(ts),
                        calculate_permutation_entropy(ts),
                        calculate_singular_spectrum_entropy(ts),
                        calculate_power_spectrum_entropy(ts, fs=fs)
                    ]
                    # Verify correct number of features returned (robustness check)
                    if len(features) != N_FEATURES_PER_COMBINATION:
                         # Log this error, indicates a bug in a calculation function
                         # print(f"ERROR: Feature count mismatch for band {b}, channel {c}. Got {len(features)}")
                         features = [0.0] * N_FEATURES_PER_COMBINATION # Fallback
                except Exception as calc_err:
                     # Catch errors during feature calculation for this specific series
                     # print(f"ERROR calculating features for band {b}, channel {c}: {calc_err}")
                     features = [0.0] * N_FEATURES_PER_COMBINATION # Fallback on error

            # Convert to float64 array for safe NaN/Inf handling
            feature_array = np.array(features, dtype=np.float64)
            # Replace any potential NaN/Inf results with 0.0
            # This handles NaNs from calculations or library issues
            if np.any(~np.isfinite(feature_array)):
                # print(f"DEBUG: NaN/Inf detected in features for band {b}, channel {c}. Replacing.")
                feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Append the cleaned feature vector to the list for the whole segment
            all_segment_features.extend(feature_array.tolist())

    # After iterating through all bands/channels, convert the full list to a float32 vector
    final_features_vector = np.array(all_segment_features, dtype=np.float32)

    # Final check: ensure the total number of features matches expectations
    if len(final_features_vector) != EXPECTED_TOTAL_FEATURES:
        # print(f"ERROR: Final feature vector length mismatch ({len(final_features_vector)} vs {EXPECTED_TOTAL_FEATURES})")
        return None # Indicate failure for this segment

    return final_features_vector


# --- 处理单个文件的函数 (用于并行化) ---
def process_single_file(fpath, output_dir_features, fs=FS):
    """
    Loads a single MAT file, extracts features for all its segments,
    and saves the results to a separate output MAT file.
    Designed to be called by multiprocessing.Pool.
    """
    fname = os.path.basename(fpath)
    pid = os.getpid() # Get process ID for logging
    start_time_file = time.time()
    file_processed_segments = 0
    file_skipped_segments = 0
    error_message = None
    n_segments_in_file = 'N/A' # Initialize for logging

    # Define output path based on input filename
    output_fname = os.path.splitext(fname)[0] + "_features.mat"
    output_fpath = os.path.join(output_dir_features, output_fname)

    # --- Optional: Skip processing if output file already exists ---
    # Useful for resuming interrupted runs. Uncomment to enable.
    # if os.path.exists(output_fpath):
    #     print(f"[P:{pid}] Skip: Output exists for {fname}", flush=True)
    #     # Return success indication but 0 processed/skipped if skipping
    #     return fname, 0, 0, "Output exists, skipped."
    # --- End Optional Skip ---

    print(f"--- [P:{pid}] Processing: {fname} ---", flush=True)

    try:
        # --- Load Data ---
        try:
             mat_data = loadmat(fpath)
        except FileNotFoundError:
             error_message = f"File not found: {fpath}"
             # Print error immediately in worker for visibility
             print(f"[P:{pid}] ERROR {fname}: {error_message}", flush=True)
             # Return immediately on critical load error
             return fname, 0, 0, error_message
        except Exception as load_err:
             error_message = f"Load error {fname}: {load_err}"
             print(f"[P:{pid}] ERROR {fname}: {error_message}", flush=True)
             return fname, 0, 0, error_message

        # --- Validate Input Data Structure ---
        if 'seg_X' not in mat_data or 'seg_y' not in mat_data:
            error_message = f"Missing 'seg_X' or 'seg_y' key in {fname}"
            print(f"[P:{pid}] WARN {fname}: {error_message}", flush=True)
            return fname, 0, 0, error_message

        seg_X = mat_data['seg_X']
        seg_y = mat_data['seg_y'].flatten() # Ensure label array is 1D

        # Validate shapes and types (add more checks if needed)
        if not isinstance(seg_X, np.ndarray) or seg_X.ndim != 4 or \
           seg_X.shape[1] != N_BANDS or seg_X.shape[2] != N_CHANNELS:
            error_message = f"Incorrect 'seg_X' shape/type: {seg_X.shape}, {type(seg_X)} in {fname}"
            print(f"[P:{pid}] WARN {fname}: {error_message}", flush=True)
            return fname, 0, 0, error_message
        if not isinstance(seg_y, np.ndarray) or len(seg_y) != seg_X.shape[0]:
             error_message = f"Incorrect 'seg_y' type or length mismatch in {fname}"
             print(f"[P:{pid}] WARN {fname}: {error_message}", flush=True)
             return fname, 0, 0, error_message

        n_segments_in_file = seg_X.shape[0]
        # Handle files with no segments
        if n_segments_in_file == 0:
             print(f"[P:{pid}] Info: {fname} contains 0 segments. Skipping feature extraction.", flush=True)
             # Return success but 0 processed/skipped
             return fname, 0, 0, "No segments in file."

        # Lists to store results for this file
        file_features_list = []
        file_labels_list = []

        # --- Process Segments Serially within this Worker ---
        for seg_idx in range(n_segments_in_file):
            # --- Extract Data for Segment ---
            try:
                 # Get data for the current segment
                 segment_data = seg_X[seg_idx, :, :, :] # Explicit slicing
                 label = seg_y[seg_idx]
            except IndexError:
                 # This should not happen if previous checks passed
                 file_skipped_segments += 1
                 continue # Skip processing this segment

            # --- Extract Features ---
            # Call the main feature extraction function for the segment
            features_vector = extract_features_for_segment(segment_data, fs=fs)

            # --- Store Results ---
            if features_vector is not None:
                # Append successful results (features are already float32)
                file_features_list.append(features_vector)
                file_labels_list.append(label)
                file_processed_segments += 1
            else:
                # Increment skipped count if feature extraction failed for the segment
                file_skipped_segments += 1
                # Optionally log which segment failed within the file? Might be too verbose.
                # print(f"[P:{pid}] DEBUG: Segment {seg_idx} failed extraction in {fname}")

        # --- After Processing All Segments for the File ---
        if file_features_list: # Check if any features were successfully extracted
            # Convert lists to numpy arrays: features are rows, labels is a vector
            # Use np.vstack for potentially better memory efficiency than np.array(list)
            features_matrix = np.vstack(file_features_list)
            labels_vector = np.array(file_labels_list, dtype=np.int32)

            # --- Save Results for this File ---
            try:
                # Save features and labels to the dedicated output file
                savemat(output_fpath, {
                    'features': features_matrix, # Shape: (n_processed_segments, EXPECTED_TOTAL_FEATURES)
                    'labels': labels_vector      # Shape: (n_processed_segments,)
                }, do_compression=True) # Use compression
            except Exception as save_err:
                # Log save error but don't overwrite previous processing errors
                save_error_msg = f"Save error for {output_fpath}: {save_err}"
                print(f"[P:{pid}] ERROR {fname}: {save_error_msg}", flush=True)
                # Append save error to any existing error message
                error_message = f"{error_message}; {save_error_msg}" if error_message else save_error_msg

        elif n_segments_in_file > 0: # File had segments, but none were processed successfully
             no_features_msg = "No features extracted successfully from any segment."
             # print(f"[P:{pid}] WARN {fname}: {no_features_msg}", flush=True) # Optional warning
             # Set error message only if no other critical error occurred earlier
             if error_message is None:
                  error_message = no_features_msg
             # No output file will be saved in this case.

    # --- Handle Broad Exceptions (e.g., MemoryError during segment loop) ---
    except MemoryError:
        error_message = f"MemoryError occurred while processing segments in {fname}"
        print(f"[P:{pid}] CRITICAL ERROR: {error_message}", flush=True)
    except Exception as e:
        error_message = f"Unexpected error during segment processing in {fname}: {e}"
        print(f"[P:{pid}] CRITICAL ERROR: {error_message}", flush=True)
        # Optionally include traceback for unexpected errors
        # traceback.print_exc()

    # --- File Processing Conclusion ---
    elapsed_file = time.time() - start_time_file
    # Determine final status based on whether an error occurred
    status_symbol = "OK" if error_message is None else "FAIL"
    segment_info_str = f"{file_processed_segments}/{n_segments_in_file}" if isinstance(n_segments_in_file, int) else "N/A"

    # Print concise summary from worker
    print(f"--- [P:{pid}] {status_symbol} Done: {fname} ({segment_info_str} segs) in {elapsed_file:.2f}s ---", flush=True)
    # Print error details only if an error occurred
    if error_message and status_symbol == "FAIL":
         print(f"    [P:{pid}]   Error details for {fname}: {error_message}", flush=True)

    # Return results for aggregation
    return fname, file_processed_segments, file_skipped_segments, error_message


# --- 主处理流程函数 (并行处理) ---
def run_parallel_processing(input_dir, output_dir_features, num_processes=None):
    """
    Uses multiprocessing Pool to process all input MAT files in parallel.
    """
    start_time_total = time.time() # Start overall timer

    # --- Ensure Output Directory Exists ---
    if not os.path.exists(output_dir_features):
        try:
            os.makedirs(output_dir_features)
            print(f"Created output directory: {output_dir_features}")
        except OSError as e:
            print(f"ERROR: Cannot create output directory '{output_dir_features}': {e}")
            return # Cannot proceed without output directory

    # --- List and Filter Input Files ---
    try:
        # List all items in the directory
        all_items = os.listdir(input_dir)
        # Filter for files ending with .mat
        mat_files = sorted([f for f in all_items if f.endswith('.mat') and
                            os.path.isfile(os.path.join(input_dir, f))])
    except FileNotFoundError:
        print(f"ERROR: Input directory not found: '{input_dir}'")
        return
    except Exception as e:
        print(f"ERROR: Cannot list input directory '{input_dir}': {e}")
        return

    # Check if any .mat files were found
    if not mat_files:
        print(f"ERROR: No .mat files found in '{input_dir}'. Please check the path.")
        return

    total_files = len(mat_files)
    print(f"Found {total_files} .mat files in '{input_dir}'.")
    print(f"Output features will be saved individually to: {output_dir_features}")

    # Create full paths for the files to be processed
    fpaths = [os.path.join(input_dir, fname) for fname in mat_files]

    # --- Determine Number of Parallel Processes ---
    if num_processes is None:
        try:
            # Use psutil for potentially more accurate core count if available
            try:
                 import psutil
                 # Prefer physical cores, fallback to logical if physical fails or not available
                 num_processes = psutil.cpu_count(logical=False)
                 if num_processes is None:
                      num_processes = psutil.cpu_count(logical=True)
            except ImportError:
                 # Fallback to standard library if psutil is not installed
                 num_processes = os.cpu_count()

            # Handle cases where detection might fail
            if num_processes is None:
                 num_processes = 4 # Default fallback
                 print("Could not reliably determine CPU cores, using default: 4 processes")
            else:
                 num_processes = max(1, num_processes) # Ensure at least 1 core
                 print(f"Auto-detected {num_processes} CPU cores to use.")

        except Exception as core_err:
            num_processes = 4 # Generic fallback on any detection error
            print(f"Error detecting CPU cores ({core_err}), using default: 4 processes")
    else:
         # Use user-specified number, ensuring it's at least 1
         num_processes = max(1, int(num_processes))
         print(f"Using specified process count: {num_processes}")

    # Adjust process count: no more processes than files
    num_processes = min(num_processes, total_files)
    # Final check to ensure at least one process
    if num_processes < 1: num_processes = 1
    print(f"Actual number of parallel worker processes to start: {num_processes}")

    # --- Prepare Function for Parallel Execution ---
    # Use functools.partial to pre-fill arguments for the worker function
    process_func_partial = partial(process_single_file,
                                   output_dir_features=output_dir_features,
                                   fs=FS)

    # --- Execute in Parallel using Pool ---
    results = [] # List to store results from worker processes
    print(f"\n--- Starting parallel processing with {num_processes} workers ---", flush=True)
    try:
        # Use a context manager for the pool to ensure proper cleanup
        with multiprocessing.Pool(processes=num_processes) as pool:
             # Use imap_unordered to process tasks and get results as they complete
             # Convert the iterator to a list to force execution and wait for all results
             results = list(pool.imap_unordered(process_func_partial, fpaths))
        print("\n--- Parallel processing pool finished. Aggregating results... ---", flush=True)

    except Exception as pool_err:
        # Catch potential errors during pool creation or execution
        print(f"\n!!! CRITICAL ERROR during multiprocessing pool execution: {pool_err} !!!", flush=True)
        traceback.print_exc() # Print detailed traceback for pool errors
        # Attempt to proceed with any results obtained before the error
        if not results: # If pool failed early, results might be empty
             print("No results were collected due to the pool error.")
             return # Exit if pool failed catastrophically

    # --- Aggregate and Summarize Results ---
    print(f"\n--- Aggregating results for {len(results)} returned tasks (out of {total_files} submitted) ---", flush=True)
    total_processed_segments_all = 0
    total_skipped_segments_all = 0
    successful_files_count = 0
    failed_files_count = 0
    error_summary = {} # Dictionary to store filenames and their errors
    processed_fnames_set = set() # Keep track of files that returned a result

    # Iterate through the results collected from the pool
    for result in results:
        # Check if the result is valid (tuple of 4 elements)
        if isinstance(result, tuple) and len(result) == 4:
            fname, processed_segs, skipped_segs, err_msg = result
            processed_fnames_set.add(fname) # Mark this file as having returned a result
            total_processed_segments_all += processed_segs
            total_skipped_segments_all += skipped_segs
            # Define success: processing completed without an error message returned
            if err_msg is None:
                successful_files_count += 1
            else:
                failed_files_count += 1
                error_summary[fname] = err_msg # Store the error message
        else:
             # Handle cases where a worker might return invalid data (e.g., None)
             failed_files_count += 1
             # Try to identify which file caused this if possible, otherwise log generic error
             error_summary[f"Unknown failure (Invalid Result)"] = f"Pool worker returned unexpected data: {result}"

    # --- Check for Files That Never Returned Results (Potential Crashes) ---
    original_fnames_set = set(mat_files)
    missed_fnames = original_fnames_set - processed_fnames_set
    if missed_fnames:
         print(f"WARNING: {len(missed_fnames)} file(s) were submitted but did not return results, indicating potential worker crashes: {missed_fnames}", flush=True)
         for fname in missed_fnames:
              # Increment failed count for these missing files
              failed_files_count += 1
              error_summary[fname] = "Task did not return a result (process likely crashed)"
         # Recalculate success count based on total submitted minus total failures
         successful_files_count = total_files - failed_files_count

    # Ensure counts are not negative after adjustments
    successful_files_count = max(0, successful_files_count)
    failed_files_count = max(0, failed_files_count)

    # --- Print Final Summary ---
    print("\n--- Final Processing Summary ---", flush=True)
    print(f"Total files attempted: {total_files}")
    print(f"Files processed successfully (returned OK status): {successful_files_count}")
    print(f"Files failed or did not complete: {failed_files_count}")
    print(f"------------------------------------")
    print(f"Total segments processed across all files: {total_processed_segments_all}")
    print(f"Total segments skipped across all files: {total_skipped_segments_all}")

    # Print summary of errors if any occurred
    if error_summary:
        print("\n--- Files with Errors/Failures ---", flush=True)
        # Sort by filename for readability
        for fname in sorted(error_summary.keys()):
            print(f"  - {fname}: {error_summary[fname]}")

    # --- Total Execution Time ---
    elapsed_total = time.time() - start_time_total
    print(f"\n--- Total script execution time: {elapsed_total:.2f} seconds ({elapsed_total/60:.2f} minutes) ---", flush=True)


# --- Script execution entry point ---
if __name__ == "__main__":
    # Add freeze_support() for compatibility when creating executables
    # Especially important on Windows and sometimes macOS
    multiprocessing.freeze_support()

    # --- Configure number of parallel processes ---
    # Set num_parallel_processes = None to auto-detect CPU cores.
    # Set to a specific integer (e.g., 16) to manually limit processes.
    # For your i9-13980HX (24c/32t), starting with None or 16-24 is reasonable.
    num_parallel_processes = None

    # --- Run the main parallel processing function ---
    run_parallel_processing(INPUT_DIR, OUTPUT_DIR_FEATURES,
                            num_processes=num_parallel_processes)

    print("\nScript finished.")