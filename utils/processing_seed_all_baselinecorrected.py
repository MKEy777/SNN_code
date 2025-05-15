from scipy.io import loadmat, savemat
import numpy as np
import os
import re

# 采样率和窗口参数
FS = 200          # 采样率（Hz）
WINDOW_S = 4      # 滑动窗口长度（秒）
STEP_S = 2        # 窗口步长（秒）
BASELINE_S = 60   # 基线校正时长（秒）

# 输出目录
OUTPUT_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\PerSession_MAT_BaselineCorrected_Variable"

def load_and_baseline_correct_trials(data_dir, baseline_s=BASELINE_S, fs=FS):
    # 计算基线样本数
    baseline_len = int(baseline_s * fs)

    # 加载标签
    lbl = loadmat(os.path.join(data_dir, "label.mat"))['label'].flatten()

    # 获取所有数据文件
    all_files = os.listdir(data_dir)
    data_files = sorted(
        [f for f in all_files if f.lower().endswith('.mat') and f.lower() != 'label.mat'],
        key=lambda f: (int(f.split('_')[0]), f.split('_')[1])
    )

    processed_ids, arrs, labs, subs = [], [], [], []

    for fname in data_files:
        # 打印当前处理文件
        print(f"Processing file: {fname}")
        subj = int(fname.split('_')[0])
        mat = loadmat(os.path.join(data_dir, fname))
        scenes = sorted([k for k in mat.keys() if not k.startswith('__')])
        # 打印文件结构（场景列表）
        print(f"Scenes in {fname}: {scenes}")

        for scene in scenes:
            # 从场景名获取试次编号
            trial_num = int(re.search(r'(\d+)$', scene).group(1))
            idx = trial_num - 1
            data = mat[scene]
            # 检查数据是否足够进行基线校正
            if data.shape[0] != 62 or data.shape[1] < baseline_len:
                continue

            # 计算并减去基线均值
            baseline_mean = np.mean(data[:, :baseline_len], axis=1, keepdims=True)
            corrected = data - baseline_mean
            corrected = np.nan_to_num(corrected.astype(np.float32))

            processed_ids.append(f"{subj}_{fname}_{scene}")
            arrs.append(corrected)
            labs.append(int(lbl[idx]))
            subs.append(subj)

    return processed_ids, arrs, np.array(labs, np.int32), np.array(subs, np.int32), None


def segment_trial(trial_data, window_s=WINDOW_S, step_s=STEP_S, fs=FS):
    # 窗口长度和步长（样本数）
    win = int(window_s * fs)
    step = int(step_s * fs)
    segments = []
    _, T = trial_data.shape
    # 生成所有完整滑动窗口
    if T < win:
        return segments
    num_seg = (T - win) // step + 1
    for i in range(num_seg):
        seg = trial_data[:, i*step : i*step + win]
        segments.append(seg.astype(np.float32))
    return segments


def main():
    data_dir = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\SEED\Preprocessed_EEG"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载并基线校正所有试次
    ids, trials, labels, subs, _ = load_and_baseline_correct_trials(data_dir)

    data_by_session = {}
    for id_str, trial, label, subj in zip(ids, trials, labels, subs):
        # 解析文件名用于会话分组
        parts = id_str.split('_')
        fname_part = next(p for p in parts if p.lower().endswith('.mat'))
        sess_key = (subj, fname_part)

        segs = segment_trial(trial)
        if sess_key not in data_by_session:
            data_by_session[sess_key] = {'X': [], 'y': []}
        data_by_session[sess_key]['X'].extend(segs)
        data_by_session[sess_key]['y'].extend([label]*len(segs))

    saved_count = 0
    # 保存分段数据
    for (subj, fname), D in sorted(data_by_session.items()):
        X = np.array(D['X'], np.float32)
        y = np.array(D['y'], np.int32)
        out_name = f"subject_{subj:02d}_session_{os.path.splitext(fname)[0]}_baseline_varlen.mat"
        savemat(os.path.join(OUTPUT_DIR, out_name), {'seg_X': X, 'seg_y': y}, do_compression=True)
        # 打印保存结果
        print(f"Saved: {out_name}, segments: {X.shape[0]}")
        saved_count += 1

    # 打印总文件数
    print(f"Total files saved: {saved_count}")

if __name__ == '__main__':
    main()