from scipy.io import loadmat, savemat
from scipy import signal
import numpy as np
import os

# 全局参数
FIXED_LEN = 36001   # 200Hz * 180s + 1
FS = 200            # 采样率，Hz
WINDOW_S = 4        # 窗口长度（秒）
STEP_S = 2          # 步长（秒）
SKIP_S = 5          # 跳过前 5 秒

# 指定输出目录
OUTPUT_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\PerSession_MAT"

def load_and_pad(data_dir, fixed_len=FIXED_LEN, skip_s=SKIP_S, fs=FS):
    """
    Load each raw .mat trial, skip the first skip_s seconds,
    then pad or truncate to fixed_len.
    Returns: list of filenames, padded data list, label array, subject array
    """
    skip_len = int(skip_s * fs)
    # Load labels from label.mat
    lbl = loadmat(os.path.join(data_dir, "label.mat"))["label"][0]
    # Find all EEG files, skipping label.mat
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.mat') and f != 'label.mat'],
        key=lambda f: (int(f.split('_')[0]), int(f.split('_')[1].split('.')[0]))
    )

    fnames, arrs, labs, subs = [], [], [], []
    for fname in files:
        # Parse subject number
        subj = int(fname.split('_')[0])
        mat = loadmat(os.path.join(data_dir, fname))
        # Exclude metadata keys, get actual data scene names
        scenes = [k for k in mat.keys() if not k.startswith('__')]
        for i, scene in enumerate(scenes):
            data = mat[scene]            # Original shape=(62, original length)
            ch, ln = data.shape
            if ln <= skip_len:
                raise ValueError(f"Trial {fname} {scene} too short: {ln} < {skip_len}")
            # Skip the first skip_len samples
            data_skipped = data[:, skip_len:]
            ln2 = data_skipped.shape[1]
            # Pad or truncate to fixed_len
            if ln2 < fixed_len:
                padded = np.full((ch, fixed_len), np.nan, dtype=data.dtype)
                padded[:, :ln2] = data_skipped
            else:
                padded = data_skipped[:, :fixed_len]

            # Check for NaN in original data
            if np.any(np.isnan(data)):
                print(f"Warning: NaN found in original data of {fname}, scene {scene}")

            fnames.append(fname)
            arrs.append(padded)
            labs.append(int(lbl[i]))
            subs.append(subj)

    return fnames, arrs, np.array(labs, dtype=np.int32), np.array(subs, dtype=np.int32)

def bandpass_filter(trial, fs=FS, bands=[(4,8),(8,13),(13,30),(30,45)]):
    """
    对单个 trial 做 4 个频段的巴特沃斯带通滤波
    输入 trial.shape=(62, T)，输出 shape=(4, 62, T)
    """
    out = []
    for lo, hi in bands:
        b, a = signal.butter(4, [lo/fs, hi/fs], btype='bandpass')
        out.append(signal.filtfilt(b, a, trial))
    return np.array(out, dtype='float32')

def segment_trial(filtered, window_s=WINDOW_S, step_s=STEP_S, fs=FS):
    """
    对单个已滤波 trial 做滑动窗口分段
    输入 filtered.shape=(4, 62, T)，输出 list of segments，每段 shape=(4,62,window_len)
    """
    win = int(window_s * fs)
    step = int(step_s * fs)
    segments = []
    T = filtered.shape[2]
    for start in range(0, T - win + 1, step):
        seg = filtered[:, :, start:start+win]
        segments.append(seg)
    return segments

def main():
    data_dir = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\SEED\Preprocessed_EEG"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载并 pad，跳过前 5 秒
    fnames, raw_list, labels, subjects = load_and_pad(data_dir)
    print(f"Loaded {len(raw_list)} trials across {len(np.unique(subjects))} subjects.")

    # 2. 滤波 + 分段，并按 (subject, session) 分组
    data_by_sess = {}  # key = (被试, session)，value = {'X': [...], 'y': [...]}

    for fname, data, label in zip(fnames, raw_list, labels):
        # 从文件名解析出 session
        base = os.path.splitext(fname)[0]
        subj_str, sess_str = base.split('_')
        subj = int(subj_str)
        sess = int(sess_str)

        # 带通滤波 & 分段
        filt = bandpass_filter(data)       # (4,62,T)
        segs = segment_trial(filt)         # list of (4,62,window_len)

        key = (subj, sess)
        if key not in data_by_sess:
            data_by_sess[key] = {'X': [], 'y': []}
        data_by_sess[key]['X'].extend(segs)
        data_by_sess[key]['y'].extend([label] * len(segs))

    # 3. 按 session 保存成 .mat 文件（共 45 个）
    for (subj, sess), D in sorted(data_by_sess.items()):
        X = np.array(D['X'], dtype='float32')      # (N,4,62,window_len)
        y = np.array(D['y'], dtype=np.int32)
        out_name = f"subject_{subj:02d}_session_{sess}.mat"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        savemat(out_path, {'seg_X': X, 'seg_y': y})
        print(f"Saved subj{subj:02d} sess{sess}: {X.shape[0]} segments -> {out_path}")

if __name__ == "__main__":
    main()
