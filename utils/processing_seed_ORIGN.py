from scipy.io import loadmat, savemat
import numpy as np
import os
import re 

FIXED_LEN = 36001   # 200Hz * 180s + 1
FS = 200            # Sampling rate, Hz
WINDOW_S = 4        # Window length (seconds)
STEP_S = 2          # Step length (seconds)
SKIP_S = 5          # Skip first 5 seconds

# Specify output directory (Adjust if needed)
OUTPUT_DIR = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\PerSession_MAT_NoBandpass_Fixed" 

def load_and_pad(data_dir, fixed_len=FIXED_LEN, skip_s=SKIP_S, fs=FS):
    """
    Load each raw .mat trial, skip the first skip_s seconds,
    then pad or truncate to fixed_len. Matches labels based on scene name trial number.
    Returns: list of unique identifiers (subj_fname_scene), padded data list, label array, subject array, unique filename list
    """
    skip_len = int(skip_s * fs)
    lbl_path = os.path.join(data_dir, "label.mat")
    if not os.path.exists(lbl_path):
        raise FileNotFoundError(f"Label file not found at: {lbl_path}. Ensure 'label.mat' is in the data directory.")
    try:
        # Load labels - Assuming it's a simple array/list of 15 labels
        lbl_data = loadmat(lbl_path)
        # Find the actual key for labels, common is 'label'
        label_key = 'label'
        if label_key not in lbl_data:
             # Try to find a key that looks like labels if 'label' isn't present
             potential_keys = [k for k in lbl_data if not k.startswith('__') and isinstance(lbl_data[k], np.ndarray)]
             if len(potential_keys) == 1:
                 label_key = potential_keys[0]
                 print(f"Info: Using label key '{label_key}' found in {lbl_path}")
             else:
                 raise KeyError(f"Could not find 'label' key or uniquely identify labels in {lbl_path}. Found keys: {list(lbl_data.keys())}")

        lbl = lbl_data[label_key].flatten() # Flatten to ensure 1D array
        label_count = len(lbl)
        print(f"Successfully loaded {label_count} labels from {lbl_path} (using key '{label_key}').")
        # Basic check - SEED usually has 15 trials/labels per session
        if label_count != 15:
            print(f"Warning: Expected 15 labels based on SEED standard, but found {label_count} in {lbl_path}.")

    except Exception as e:
        raise IOError(f"Could not load or parse label file {lbl_path}: {e}")

    # Find all EEG files, skipping label.mat
    try:
        all_files = os.listdir(data_dir)
    except FileNotFoundError:
         raise FileNotFoundError(f"Input data directory not found: {data_dir}. Please check the 'data_dir' path.")
    except Exception as e:
         raise IOError(f"Could not list files in input directory {data_dir}: {e}")

    # Filter for .mat files, exclude label.mat, ensure they are files
    data_files = sorted(
        [f for f in all_files if f.lower().endswith('.mat') and f.lower() != 'label.mat' and os.path.isfile(os.path.join(data_dir, f))],
        key=lambda f: (int(f.split('_')[0]), f.split('_')[1]) # Sort by subject, then date/session part
    )

    unique_fnames, arrs, labs, subs = [], [], [], []
    processed_identifiers = [] # To store subj_fname_scene

    print(f"Found {len(data_files)} potential EEG data files in {data_dir}")
    if not data_files:
         print("Warning: No data files (*.mat excluding label.mat) found.")
         return processed_identifiers, arrs, np.array(labs, dtype=np.int32), np.array(subs, dtype=np.int32), unique_fnames

    for i_file, fname in enumerate(data_files):
        print(f"Processing file {i_file+1}/{len(data_files)}: {fname}")
        try:
            # Parse subject ID from filename
            subj_str = fname.split('_')[0]
            subj = int(subj_str)
            unique_fnames.append(fname) # Store the filename representing the session

            mat_path = os.path.join(data_dir, fname)
            mat = loadmat(mat_path)

        except (IndexError, ValueError):
            print(f"Warning: Could not parse subject ID integer from filename {fname}. Skipping file.")
            continue
        except Exception as e:
            print(f"Warning: Could not load file {mat_path}: {e}. Skipping file.")
            continue

        # Exclude metadata keys, get actual data scene names (like 'djc_eeg1', 'eeg_1', etc.)
        scenes = sorted([k for k in mat.keys() if not k.startswith('__')])
        if not scenes:
            print(f"Warning: No data scenes found in {fname}. Skipping file.")
            continue

        # --- Corrected Label Logic ---
        for scene in scenes:
            # Attempt to extract trial number from scene name using regex
            match = re.search(r'(\d+)$', scene) # Find digits at the end of the string
            if match:
                trial_number_1_based = int(match.group(1))
                current_label_index = trial_number_1_based - 1 # Convert to 0-based index
            else:
                print(f"Warning: Could not extract trial number from scene name '{scene}' in {fname}. Skipping scene.")
                continue

            # Check if label index is valid for the loaded labels
            if not (0 <= current_label_index < label_count):
                print(f"Warning: Calculated label index {current_label_index} (from scene '{scene}') is out of bounds for loaded labels (count: {label_count}) in {fname}. Skipping scene.")
                continue

            # --- Data Loading and Processing (mostly unchanged) ---
            data = mat[scene]
            if data is None or not isinstance(data, np.ndarray) or data.size == 0:
                 print(f"Warning: Invalid or empty data for scene {scene} in {fname}. Skipping scene.")
                 continue
            if data.ndim != 2 or data.shape[0] != 62:
                 print(f"Warning: Unexpected data shape {data.shape} for scene {scene} in {fname}. Expected (62, N). Skipping scene.")
                 continue

            ch, ln = data.shape
            if ln <= skip_len:
                print(f"Info: Scene {scene} in {fname} original length ({ln}) <= skip length ({skip_len}). Skipping scene.")
                continue

            data_skipped = data[:, skip_len:]
            ln2 = data_skipped.shape[1]
            if ln2 == 0:
                 print(f"Info: Scene {scene} in {fname} has zero length after skipping. Skipping scene.")
                 continue

            if ln2 < fixed_len:
                padded = np.full((ch, fixed_len), 0.0, dtype=np.float32)
                padded[:, :ln2] = data_skipped.astype(np.float32)
            else:
                padded = data_skipped[:, :fixed_len].astype(np.float32)

            if not np.all(np.isfinite(padded)):
                print(f"Warning: Non-finite values found in data for {fname}, scene {scene}. Replacing with 0.")
                padded = np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)

            # Append data and labels
            processed_identifiers.append(f"{subj}_{fname}_{scene}") # Unique ID for this piece of data
            arrs.append(padded)
            labs.append(int(lbl[current_label_index])) # Use CORRECT label index
            subs.append(subj)

    print(f"Loaded and padded data for {len(arrs)} total valid scenes from {len(set(unique_fnames))} files.")
    return processed_identifiers, arrs, np.array(labs, dtype=np.int32), np.array(subs, dtype=np.int32), unique_fnames


def segment_trial(trial_data, window_s=WINDOW_S, step_s=STEP_S, fs=FS):
    """
    Segments a single trial data (no band dimension) using a sliding window.
    Input trial_data.shape=(62, T), output list of segments, each shape=(62, window_len)
    """
    win = int(window_s * fs)
    step = int(step_s * fs)
    segments = []
    if not isinstance(trial_data, np.ndarray) or trial_data.ndim != 2:
        print(f"Error: segment_trial expected 2D numpy array, got {type(trial_data)} with shape {getattr(trial_data, 'shape', 'N/A')}")
        return segments # Return empty list on error

    channels, T = trial_data.shape
    if channels != 62: # Add check for expected channel count
         print(f"Warning: segment_trial received data with {channels} channels, expected 62. Proceeding, but check input.")

    if T < win:
         return segments # Not an error, just no segments possible

    num_segments = (T - win) // step + 1

    for i in range(num_segments):
        start = i * step
        end = start + win
        if end > T: break # Safety break
        seg = trial_data[:, start:end]
        if seg.shape == (channels, win):
             segments.append(seg.astype(np.float32))
        else:
             print(f"Warning: Segment created with incorrect shape {seg.shape} at start={start}. Expected ({channels}, {win}). Skipping segment.")

    return segments

def main():
    data_dir = r"C:\Users\VECTOR\Desktop\DeepLearning\SNN\SEED\SEED\Preprocessed_EEG" # Make sure this path is correct!
    # Ensure the output directory name is distinct
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Input data directory: {data_dir}")
    print(f"Output directory for segmented data (no bandpass): {OUTPUT_DIR}")

    # 1. Load and pad, skip first 5 seconds
    try:
        # Returns: processed_ids, data_list, labels_array, subjects_array, unique_filenames
        proc_ids, raw_list, labels, subjects, unique_filenames = load_and_pad(data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return # Exit if data loading fails critically
    except IOError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        return

    if not raw_list:
         print("No data was successfully loaded. Exiting.")
         return

    print(f"Loaded {len(raw_list)} valid trials/scenes across {len(np.unique(subjects))} unique subjects from {len(unique_filenames)} files.")

    # 2. Segment and group by (subject, original_filename)
    # The original filename represents a unique recording session/date
    data_by_session_file = {}  # key = (subject, original_fname), value = {'X': [], 'y': []}

    print("Segmenting trials...")
    if len(proc_ids) != len(raw_list): # Sanity check
        print("Error: Mismatch between identifiers and data length after loading. Aborting.")
        return

    for identifier, data, label, subj in zip(proc_ids, raw_list, labels, subjects):
        # Extract original filename from identifier (e.g., "1_1_20131027.mat_djc_eeg1")
        try:
            parts = identifier.split('_')
            # Find the part ending with .mat
            fname_part = ""
            for i, p in enumerate(parts):
                 if p.lower().endswith('.mat'):
                     fname_part = "_".join(parts[1:i+1]) # Reconstruct fname (e.g., 1_20131027.mat)
                     break
            if not fname_part:
                 print(f"Warning: Could not reconstruct filename from identifier '{identifier}'. Skipping.")
                 continue
            session_key = (subj, fname_part) # Use (subject_id, original_filename) as the key
        except Exception as e:
            print(f"Warning: Error creating session key from identifier '{identifier}'. Skipping. Error: {e}")
            continue

        # Segment the data
        segs = segment_trial(data, window_s=WINDOW_S, step_s=STEP_S, fs=FS)

        if not segs:
            continue # Skip if no segments were generated

        if session_key not in data_by_session_file:
            data_by_session_file[session_key] = {'X': [], 'y': []}

        data_by_session_file[session_key]['X'].extend(segs)
        data_by_session_file[session_key]['y'].extend([label] * len(segs))

    # 3. Save segmented data per original session file
    saved_files_count = 0
    total_segments_saved = 0
    print("\nSaving segmented data per session file...")
    if not data_by_session_file:
        print("No data grouped by session file. Nothing to save.")

    # Sort by subject, then by filename for consistent output order
    sorted_keys = sorted(data_by_session_file.keys(), key=lambda k: (k[0], k[1]))

    for session_key in sorted_keys:
        subj, original_fname = session_key
        D = data_by_session_file[session_key]

        if not D['X']:
             continue # Skip if empty

        try:
            if not D['X'] or not D['y']: continue
            X = np.array(D['X'], dtype=np.float32)
            y = np.array(D['y'], dtype=np.int32)
        except Exception as e:
             print(f"Error converting data to array for subj{subj} file {original_fname}: {e}. Skipping save.")
             continue

        expected_win_len = int(WINDOW_S * FS)
        if X.ndim != 3 or X.shape[1] != 62 or X.shape[2] != expected_win_len or y.ndim != 1 or X.shape[0] != len(y):
             print(f"Error: Data shape mismatch before saving subj{subj} file {original_fname}.")
             print(f"  X shape: {X.shape}, y shape: {y.shape}. Skipping save.")
             continue

        # Create a safe output filename from the original filename
        safe_original_fname = os.path.splitext(original_fname)[0] # Remove .mat extension
        out_name = f"subject_{subj:02d}_session_{safe_original_fname}_no_bandpass.mat"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        try:
            savemat(out_path, {'seg_X': X, 'seg_y': y}, do_compression=True)
            print(f"Saved subj{subj:02d} file {original_fname}: {X.shape[0]} segments -> {out_path}")
            saved_files_count += 1
            total_segments_saved += X.shape[0]
        except Exception as e:
            print(f"Error saving file {out_path}: {e}")


    print(f"\nFinished saving.")
    print(f"Total files saved: {saved_files_count}") # Should match number of unique input .mat files if all had valid data
    print(f"Total segments saved across all files: {total_segments_saved}")

if __name__ == "__main__":
    main()
    print("\nScript finished.")