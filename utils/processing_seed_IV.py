from scipy.io import loadmat, savemat
import numpy as np
import os
import re
import time # 引入 time 模块以计时
from multiprocessing import Pool, cpu_count # 引入多进程相关模块

# 采样率和窗口参数 (保持不变)
FS = 200
WINDOW_S = 4
STEP_S = 2
BASELINE_S = 30

# 输出目录 (保持不变)
# 注意：如果原始脚本和修改后脚本的输出目录相同，并行运行时可能会有文件冲突或覆盖。
# 建议为并行版本设置不同的输出目录，或确保文件名唯一性。
OUTPUT_DIR = r"PerSession_MAT_BaselineCorrected_Variable_MP" # 为多进程版本添加 _MP 后缀

# SEED-IV 标签数组 (保持不变)
SESSION_LABELS = {
    '1': [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    '2': [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    '3': [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
}

def segment_trial(trial_data, window_s=WINDOW_S, step_s=STEP_S, fs=FS): #
    """将试次数据分段"""
    win = int(window_s * fs) #
    step = int(step_s * fs) #
    segments = [] #
    _, T = trial_data.shape #
    if T < win: #
        # 为多进程环境减少打印，或使用日志库
        # print(f"PID {os.getpid()}: 跳过分段：数据长度 {T} 小于窗口长度 {win}")
        return segments #
    num_seg = (T - win) // step + 1 #
    for i in range(num_seg): #
        seg = trial_data[:, i*step : i*step + win] #
        segments.append(seg.astype(np.float32)) #
    return segments

def process_session_data(data_file, session_label_list, session_folder_str, baseline_s=BASELINE_S, fs=FS, window_s=WINDOW_S, step_s=STEP_S): #
    """处理单个 .mat 文件 (参数名稍作调整以避免与外部变量混淆)"""
    # 加载 .mat 文件
    pid = os.getpid() # 获取当前进程ID，方便调试打印
    filename = os.path.basename(data_file) #
    try:
        mat = loadmat(data_file) #
    except Exception as e:
        print(f"PID {pid}: 错误：无法加载 {filename}：{e}") #
        return None # 返回None表示处理失败

    # print(f"PID {pid}: {filename} 中的键：{list(mat.keys())}") # 调试信息，可按需保留或移除

    parts = filename.split('_') #
    try:
        subj = int(parts[0]) #
        session_id_str = parts[1].split('.')[0] #
    except (IndexError, ValueError) as e:
        print(f"PID {pid}: 错误：解析文件名 {filename} 失败：{e}") #
        return None

    session_key = session_folder_str #
    if session_key not in SESSION_LABELS: #
        print(f"PID {pid}: 错误：文件夹 {session_key} 不存在于 SESSION_LABELS") #
        return None

    labels_arr = np.array(SESSION_LABELS[session_key], dtype=np.int32) #
    # print(f"PID {pid}: 使用 {session_key} 的标签：{labels_arr}")

    trial_keys = sorted( #
        [k for k in mat.keys() if re.match(r'.*_eeg\d+', k)], #
        key=lambda x: int(re.search(r'\d+', x.split('_eeg')[1]).group()) #
    )

    segments_list = [] #
    seg_labels_list = [] #

    for trial_key in trial_keys: #
        trial_num = int(re.search(r'\d+', trial_key.split('_eeg')[1]).group()) #
        idx = trial_num - 1 #
        if idx >= len(labels_arr): #
            # print(f"PID {pid}: 警告：{filename} 中的试次 {trial_num} 索引超出标签数组长度 {len(labels_arr)}")
            continue
        label_val = labels_arr[idx] #

        data_trial = mat[trial_key] #
        if data_trial.shape[0] != 62: #
            # print(f"PID {pid}: 跳过 {filename} 中的试次 {trial_num}：通道数 {data_trial.shape[0]} 不等于 62")
            continue
        if data_trial.shape[1] < int(baseline_s * fs): #
            # print(f"PID {pid}: 跳过 {filename} 中的试次 {trial_num}：数据长度 {data_trial.shape[1]} 小于基线长度 {int(baseline_s * fs)}")
            continue

        baseline_len = int(baseline_s * fs) #
        baseline_mean_val = np.mean(data_trial[:, :baseline_len], axis=1, keepdims=True) #
        corrected_data = data_trial - baseline_mean_val #
        corrected_data = np.nan_to_num(corrected_data.astype(np.float32)) #

        trial_segments_list = segment_trial(corrected_data, window_s, step_s, fs) #
        # print(f"PID {pid}: 试次 {trial_num} 为文件 {filename} 生成了 {len(trial_segments_list)} 个片段")

        segments_list.extend(trial_segments_list) #
        seg_labels_list.extend([label_val] * len(trial_segments_list)) #

    if not segments_list: # 如果没有有效的片段被处理
        # print(f"PID {pid}: 文件 {filename} 没有生成任何有效片段。")
        return None

    # 将结果打包以便后续保存
    X_arr = np.array(segments_list, dtype=np.float32) #
    y_arr = np.array(seg_labels_list, dtype=np.int32) #
    out_filename_str = f"subject_{subj:02d}_session_{session_id_str}_baseline_varlen.mat" #
    
    return {'output_dir': OUTPUT_DIR, 'out_name': out_filename_str, 'data_X': X_arr, 'data_y': y_arr, 'original_file': filename, 'pid': pid}


def save_processed_data(result):
    """回调函数，用于在主进程中保存由工作进程处理的数据。"""
    if result is None:
        return False # 表示处理失败或没有数据

    try:
        savemat(os.path.join(result['output_dir'], result['out_name']), {'seg_X': result['data_X'], 'seg_y': result['data_y']}, do_compression=True) #
        print(f"PID {result['pid']} (saved by main): 已保存：{result['out_name']} (源文件: {result['original_file']})，片段数：{result['data_X'].shape[0]}") #
        return True
    except Exception as e:
        print(f"PID {result['pid']} (save failed by main): 保存 {result['out_name']} 时出错 (源文件: {result['original_file']})：{e}")
        return False

def main_multiprocess(): #
    """主函数 - 多进程版本"""
    overall_start_time = time.time()
    data_dir_root = r"eeg_raw_data" #
    os.makedirs(OUTPUT_DIR, exist_ok=True) #

    tasks_to_process = []
    for session_folder_name in ['1', '2', '3']: #
        # session_actual_labels = SESSION_LABELS[session_folder_name] # session_label 参数现在直接通过 session_folder_name 传递
        current_session_path = os.path.join(data_dir_root, session_folder_name) #
        if not os.path.exists(current_session_path): #
            print(f"警告：文件夹 {current_session_path} 不存在，跳过") #
            continue

        mat_file_names = [f for f in os.listdir(current_session_path) if f.lower().endswith('.mat')] #
        if not mat_file_names: #
            print(f"警告：在 {current_session_path} 中未找到 .mat 文件") #
            continue

        print(f"\n收集会话文件夹 {session_folder_name} 中的任务，找到 {len(mat_file_names)} 个文件") #
        for mat_file_name_only in mat_file_names: #
            full_file_path = os.path.join(current_session_path, mat_file_name_only) #
            # 将参数打包为元组，传递给 process_session_data
            # 注意：session_label_list 现在应该是 SESSION_LABELS[session_folder_name]，但 process_session_data 内部会通过 session_folder_str 重新获取
            tasks_to_process.append((full_file_path, SESSION_LABELS[session_folder_name], session_folder_name, BASELINE_S, FS, WINDOW_S, STEP_S))

    if not tasks_to_process:
        print("没有找到需要处理的文件。")
        return

    num_processes = cpu_count() -1 if cpu_count() > 1 else 1 # 使用CPU核心数减1，或至少1个进程
    print(f"\n开始使用 {num_processes} 个进程处理 {len(tasks_to_process)} 个文件...")

    successful_saves = 0
    failed_processing_or_save = 0

    # 创建进程池
    # 使用 with 语句确保进程池在使用完毕后能够正确关闭
    with Pool(processes=num_processes) as pool:
        # 使用 starmap 来传递多个参数给 process_session_data
        # starmap 会阻塞，直到所有任务完成
        results_from_workers = pool.starmap(process_session_data, tasks_to_process)

    # 在主进程中串行保存结果，避免多个进程同时写文件（尽管在此场景下输出文件名唯一，但这是个好习惯）
    # 或者，如果 process_session_data 直接保存，则这里收集统计信息
    print("\n所有工作进程已完成，开始保存结果...")
    for result_data in results_from_workers:
        if result_data: # 检查 process_session_data 是否成功返回数据
            if save_processed_data(result_data):
                successful_saves +=1
            else:
                failed_processing_or_save +=1
        else: # process_session_data 返回 None，表示处理失败
            failed_processing_or_save +=1


    overall_end_time = time.time()
    print(f"\n处理完成。总耗时：{overall_end_time - overall_start_time:.2f} 秒")
    print(f"成功保存文件数: {successful_saves}")
    print(f"处理或保存失败/跳过的文件数: {failed_processing_or_save}")


if __name__ == '__main__':

    main_multiprocess()