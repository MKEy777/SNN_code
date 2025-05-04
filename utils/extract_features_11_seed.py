import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
import os
import time # Keeping time for basic profiling if needed

FEATURE_NAMES = [
    "Mean PTP", "Mean Square Value", "Variance", "PSD Sum", "Max PSD Value",
    "Freq at Max PSD", "Hjorth Mobility", "Hjorth Complexity", "Shannon Entropy",
    "C0 Complexity (FWHM)", "Power Spectrum Entropy"
]

INPUT_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\PerSession_MAT"
OUTPUT_DIR_FEATURES = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\Individual_Features"
FS = 200
N_FEATURES_PER_COMBINATION = 11
N_BANDS = 4
N_CHANNELS = 62
EXPECTED_TOTAL_FEATURES = N_BANDS * N_CHANNELS * N_FEATURES_PER_COMBINATION

# --- Feature Calculation Functions ---

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

# --- Feature Extraction (Minimal) ---

def extract_features_for_segment(segment_data, fs=FS):
    all_segment_features = []
    bands, channels, window_len = segment_data.shape
    for b in range(bands):
        for c in range(channels):
            ts = segment_data[b, c, :]
            # Basic check for completely invalid input replaced by assuming valid input
            # or letting downstream calculations handle potential NaNs if they can
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

            # No NaN/Inf check, assumes results are finite or handled by np.array conversion
            all_segment_features.extend(features)

    # No final length check
    # Assumes float32 is desired final type
    return np.array(all_segment_features, dtype=np.float32)

# --- File Processing (Minimal) ---

def process_single_file(fpath, output_dir_features, fs=FS):
    fname = os.path.basename(fpath)
    output_fname = os.path.splitext(fname)[0] + "_features_minimal.mat"
    output_fpath = os.path.join(output_dir_features, output_fname)

    # No try-except for loading
    mat_data = loadmat(fpath)
    seg_X = mat_data['seg_X'] # Assumes key exists
    seg_y_raw = mat_data['seg_y'] # Assumes key exists
    seg_y = seg_y_raw.flatten() # Assumes flattenable

    n_segments_in_file = seg_X.shape[0]
    if n_segments_in_file == 0:
        print(f"  Skipping {fname} as it contains no segments.") # Indicate skipped files
        return # Just exit if no segments

    file_features_list = []
    file_labels_list = []

    # --- MODIFICATION START: Print segment progress ---
    print(f"  Processing {n_segments_in_file} segments...")
    for seg_idx in range(n_segments_in_file):
        # Print progress every 10 segments, or for the first and last segment
        if (seg_idx + 1) % 10 == 0 or seg_idx == 0 or seg_idx == n_segments_in_file - 1:
             print(f"    Processing segment {seg_idx + 1}/{n_segments_in_file}...")
        # --- MODIFICATION END ---

        # No try-except for indexing
        segment_data = seg_X[seg_idx, :, :, :]
        label = seg_y[seg_idx]
        # No check for feature extraction result validity
        features_vector = extract_features_for_segment(segment_data, fs=fs)
        # Assumes feature vector is always correct length and valid
        file_features_list.append(features_vector)
        file_labels_list.append(label)

    if file_features_list:
        features_matrix = np.vstack(file_features_list)
        labels_vector = np.array(file_labels_list, dtype=np.int32)
        # No try-except for saving
        savemat(output_fpath, {'features': features_matrix, 'labels': labels_vector}, do_compression=True)
        # --- MODIFICATION START: Print save confirmation ---
        print(f"  Saved features for {fname} to {output_fname}")
        # --- MODIFICATION END ---
    else:
        # --- MODIFICATION START: Print if nothing was saved ---
        print(f"  No features extracted or saved for {fname}.")
        # --- MODIFICATION END ---


# --- Main Execution (Minimal) ---

def run_minimal_processing(input_dir, output_dir_features):
    start_time_total = time.time()

    # No try-except for directory creation/listing
    if not os.path.exists(output_dir_features):
        print(f"Creating output directory: {output_dir_features}")
        os.makedirs(output_dir_features)

    mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat') and os.path.isfile(os.path.join(input_dir, f))])

    if not mat_files:
        print(f"ERROR: No .mat files found in '{input_dir}'.") # Keep minimal error check
        return

    total_files = len(mat_files)
    print(f"Found {total_files} files. Processing...") # Minimal status

    for i, fname in enumerate(mat_files):
        # --- MODIFICATION START: Print file progress ---
        print(f"Processing file {i+1}/{total_files}: {fname}") # Uncommented and kept
        # --- MODIFICATION END ---
        fpath = os.path.join(input_dir, fname)
        process_single_file(fpath, output_dir_features, fs=FS) # No result tracking

    elapsed_total = time.time() - start_time_total
    print(f"\nProcessing finished for {total_files} files in {elapsed_total:.2f} seconds.") # Minimal final status

if __name__ == "__main__":
    run_minimal_processing(INPUT_DIR, OUTPUT_DIR_FEATURES)
    print("Script finished.")