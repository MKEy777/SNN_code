from scipy.io import loadmat, savemat
from scipy import signal
import numpy as np
import os

# 全局参数
FIXED_LEN = 37001   # 200Hz * 185s + 1
FS = 200            # 采样率，Hz
# 指定输出目录
OUTPUT_DIR = r"C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN\\SEED\\PerSubject_MAT"


def load_seed_data(data_dir, fixed_len=FIXED_LEN):
    """
    加载 SEED 数据并 pad/truncate 到 fixed_len
    返回: padded_list, labels_list, subject_list, fixed_len
    """
    lbl = loadmat(os.path.join(data_dir, "label.mat"))["label"][0]
    files = [f for f in os.listdir(data_dir)
             if f not in ("label.mat", "readme.txt")]
    files = sorted(files, key=lambda f: (int(f.split("_")[0]), f))

    padded_list, labels_list, subject_list = [], [], []
    for fname in files:
        subject_id = int(fname.split('_')[0])
        mat = loadmat(os.path.join(data_dir, fname))
        scenes = [k for k in mat.keys() if not k.startswith("__")]
        for i, scene in enumerate(scenes):
            data = mat[scene]  # shape=(62, original_len)
            ch, ln = data.shape
            if ln < fixed_len:
                arr = np.full((ch, fixed_len), np.nan, dtype=data.dtype)
                arr[:, :ln] = data
            else:
                arr = data[:, :fixed_len]
            padded_list.append(arr)
            labels_list.append(int(lbl[i]))
            subject_list.append(subject_id)
    return padded_list, np.array(labels_list, dtype=np.int32), np.array(subject_list, dtype=np.int32), fixed_len


def filter_only(data, fs=FS, fStart=[4, 8, 13, 30], fEnd=[8, 13, 30, 45]):
    """Butterworth 带通滤波，返回四频段数据"""
    bands = []
    for lo, hi in zip(fStart, fEnd):
        b, a = signal.butter(4, [lo/fs, hi/fs], 'bandpass')
        bands.append(signal.filtfilt(b, a, data))
    return np.array(bands, dtype='float32')


def segment_trials(processed_list, labels_list, subject_list,
                   window_s=4, step_s=2, fs=FS):
    """滑动窗口分段，返回 seg_X, seg_y, seg_subj"""
    window_len = int(window_s * fs)
    step = int(step_s * fs)
    segs, seg_labels, seg_subj = [], [], []
    for data, label, subj in zip(processed_list, labels_list, subject_list):
        t_len = data.shape[2]
        for start in range(0, t_len - window_len + 1, step):
            segs.append(data[:, :, start:start + window_len])
            seg_labels.append(label)
            seg_subj.append(subj)
    return (np.array(segs, dtype='float32'),
            np.array(seg_labels, dtype='int32'),
            np.array(seg_subj, dtype='int32'))


def main():
    data_dir = r"C:\\Users\\VECTOR\\Desktop\\DeepLearning\\SNN\\SEED\\SEED\\Preprocessed_EEG"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载并 pad/truncate
    raw_list, labels, subjects, length = load_seed_data(data_dir)
    print(f"Loaded {len(raw_list)} trials for {len(np.unique(subjects))} subjects, length={length}")

    # 2. 滤波
    filtered = [filter_only(x) for x in raw_list]
    print(f"Filtered all trials into {len(filtered)} items")

    # 3. 切割
    seg_X, seg_y, seg_subj = segment_trials(filtered, labels, subjects)
    print(f"Segmented into X={seg_X.shape}, y={seg_y.shape}, subj={seg_subj.shape}")

    # 4. 按被试保存，每个被试 3 个 session 聚合到一个纯 .mat 文件
    for subj in sorted(np.unique(seg_subj)):
        mask = seg_subj == subj
        X_sub = seg_X[mask]
        y_sub = seg_y[mask]
        out_path = os.path.join(OUTPUT_DIR, f"subject_{subj:02d}.mat")
        savemat(out_path, {'seg_X': X_sub, 'seg_y': y_sub})
        print(f"Saved subject {subj:02d}: {X_sub.shape[0]} segments -> {out_path}")

if __name__ == "__main__":
    main()
