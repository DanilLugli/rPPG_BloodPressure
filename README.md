# Blood Pressure Estimation from PPG and PPW Signals

This project aims to estimate blood pressure (SBP and DBP) using PPG (Photoplethysmography) and PPW (Pulse Wave Velocity) signals extracted from videos. The project includes various signal processing techniques, interpolation, peak detection, and Pulse Transit Time (PTT) calculation.

## Project Structure

- **Extraction of PPG signals from the forehead and neck**
- **Calculation of PPW signal from the neck**
- **Preprocessing and filtering of signals**
- **Extraction of BVP (Blood Volume Pulse) signal**
- **Peak detection in BVP signals**
- **PTT calculation using cross-correlation and direct methods**
- **Signal interpolation to improve temporal resolution**
- **Calculation of Augmentation Index (AIx)**
- **Blood pressure estimation (SBP and DBP)**
- **Analysis and comparison of blood pressure estimation methods**
- **Visualization and saving of results**

## Requirements

- Python 3.7+
- Python libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`
  - `seaborn`
  - `opencv-python`
  - `mediapipe`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/blood-pressure-estimation.git
    cd blood-pressure-estimation
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Initialize paths and configurations**:
    - Modify the [initialize_paths_and_config()](http://_vscodecontentref_/0) function in the [test_head+neck.py](http://_vscodecontentref_/1) file to specify the paths to your data.

2. **Run the main script**:
    ```bash
    python test_head+neck.py
    ```

3. **Results**:
    - The results will be saved in the specified output directory. The results include filtered signals, peak detection plots, PTT values, blood pressure estimates, and comparison plots.

## Main Functions

### Signal Extraction

- [extract_rgb_trace_MediaPipe(video_path, output_folder)](http://_vscodecontentref_/2): Extracts RGB signals from the forehead using MediaPipe.
- [extract_pixflow_signal_improved(video_path, roi, output_folder, part)](http://_vscodecontentref_/3): Extracts the PPW signal from the neck using optical flow.

### Preprocessing and Filtering

- [preprocess_and_filter_signals(signals, sampling_rate, output_dir, label)](http://_vscodecontentref_/4): Preprocesses and filters the signals, removing any artifacts.

### BVP Extraction

- [cpu_CHROM(signal, output_dir, label)](http://_vscodecontentref_/5): Calculates the BVP signal using the CHROM method.
- [compute_bvp_green_neck(filtered_green_signal, sampling_rate, output_dir, label)](http://_vscodecontentref_/6): Calculates the BVP signal from the green signal of the neck.

### Peak Detection

- [find_peaks_in_signal(ppg_signal, fps, output_dir, label)](http://_vscodecontentref_/7): Detects peaks in the BVP signals.

### PTT Calculation

- [calculate_ptt_cross_correlation(peaks_forehead, peaks_neck, fps, output_dir, label)](http://_vscodecontentref_/8): Calculates PTT using cross-correlation.
- [calculate_ptt_direct(peaks_forehead, peaks_neck, fps, output_dir, label)](http://_vscodecontentref_/9): Calculates PTT using the direct method.

### Signal Interpolation

- [convolution_interpolation(signal, old_rate, new_rate, output_dir, label)](http://_vscodecontentref_/10): Applies cubic convolution interpolation.
- [cubic_spline_interpolation(signal, original_fs, target_fs, output_dir, label)](http://_vscodecontentref_/11): Applies cubic spline interpolation.

### AIx Calculation

- [calculate_aix(bvp_forehead, bvp_neck, peaks_forehead, peaks_neck, fs, output_dir)](http://_vscodecontentref_/12): Calculates the Augmentation Index (AIx).

### Blood Pressure Estimation

- [estimate_systolic_blood_pressure(ptt_values, gender)](http://_vscodecontentref_/13): Estimates systolic blood pressure.
- [estimate_diastolic_blood_pressure(ptt_values, gender)](http://_vscodecontentref_/14): Estimates diastolic blood pressure.
- [estimate_bp_from_aix(aix_values, output_dir)](http://_vscodecontentref_/15): Estimates blood pressure using AIx.

### Analysis and Comparison

- [analyze_bp_methods(estimated_sbp_ppg, estimated_dbp_ppg, estimated_sbp_ppw, estimated_dbp_ppw, estimated_sbp_aix, estimated_dbp_aix, output_dir)](http://_vscodecontentref_/16): Compares different blood pressure estimation methods.
- [compare_all_ptt_methods(ptt_dict, output_dir)](http://_vscodecontentref_/17): Compares PTT values calculated with different methods.

## Contributions

Contributions are welcome! Feel free to open issues or pull requests to improve the project.

