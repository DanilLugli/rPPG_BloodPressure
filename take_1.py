import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import cv2

def select_neck_roi(frame):
    r = cv2.selectROI("Select Neck ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Neck ROI")
    return r

def select_forehead_roi(frame):
    r = cv2.selectROI("Select Forehead ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Forehead ROI")
    return r

def save_roi_dimensions(roi, file_path):
    roi_data = {
        "x": roi[0],
        "y": roi[1],
        "width": roi[2],
        "height": roi[3]
    }
    with open(file_path, 'w') as file:
        json.dump(roi_data, file)

def load_roi_dimensions(file_path):
    with open(file_path, 'r') as file:
        roi_data = json.load(file)
    return (roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height'])

def extract_red_trace(video_path, roi, output_folder, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    red_signal = []
    x, y, w, h = roi

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        roi_frame = image[y:y+h, x:x+w]
        mean_val = cv2.mean(roi_frame)[2]  # Extracting the red channel
        red_signal.append(mean_val)

    cap.release()
    cv2.destroyAllWindows()

    red_signal = np.array(red_signal)
    np.savetxt(f"{output_folder}/ppg_raw_{label}_R.txt", red_signal)
    plt.plot(red_signal)
    plt.title(f'PPG Signal (Raw - {label} - R)')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.savefig(f"{output_folder}/ppg_raw_{label}_R.jpg")
    plt.close()

    return red_signal

def smoothness_priors_detrend(signal, lambda_param=10):
    n = len(signal)
    I = sparse.eye(n)
    D = sparse.eye(n, n, 1) - sparse.eye(n, n)
    D = D[1:]  # Elimina la prima riga di D per ottenere una matrice (n-1) x n

    H = I + lambda_param * D.T @ D
    H = H.tocsc()  # Converti H in formato di matrice sparsa compressa per colonne per l'inversione

    trend = splinalg.spsolve(H, signal)
    detrended_signal = signal - trend

    return detrended_signal

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', output='sos', fs=fs)
    filtered_signal = sosfiltfilt(sos, signal)
    return filtered_signal

def plot_ppg_signal(ppg_signal, label, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(ppg_signal, label=label)
    plt.xlabel('Frame')
    plt.ylabel('Mean Intensity')
    plt.title(f'{label} Signal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{label}_signal.png'))
    plt.close()

def find_peaks_in_signal(ppg_signal):
    peaks, _ = find_peaks(ppg_signal, distance=20)
    return peaks

def plot_peaks(ppg_signal, peaks, label, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(ppg_signal, label=label)
    plt.plot(peaks, np.array(ppg_signal)[peaks], "x", label='Peaks')
    plt.xlabel('Frame')
    plt.ylabel('Mean Intensity')
    plt.title(f'{label} with Peaks')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{label}_peaks.png'))
    plt.close()

def calculate_ptt(peaks_forehead, peaks_neck, fps, output_dir):
    peaks_forehead = np.array(peaks_forehead)
    peaks_neck = np.array(peaks_neck)
    t_ppg = peaks_forehead / fps
    t_ppw = peaks_neck / fps
    ptt_values = []
    for t_f in t_ppg:
        t_n = t_ppw[np.argmin(np.abs(t_ppw - t_f))]
        ptt_values.append(t_f - t_n)
    ptt_values = np.array(ptt_values)
    np.savetxt(os.path.join(output_dir, 'ptt_values.txt'), ptt_values, fmt='%.5f')
    return ptt_values

def plot_ptt_values(ptt_values, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(ptt_values, label='PTT Values')
    plt.xlabel('Measurement Index')
    plt.ylabel('PTT (seconds)')
    plt.title('PTT Values Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ptt_values.png'))
    plt.close()

def estimate_systolic_blood_pressure(ptt_values):
    a = -100
    b = 120
    estimated_sbp = a * ptt_values + b
    return estimated_sbp

def estimate_diastolic_blood_pressure(ptt_values):
    a = -75
    b = 80
    estimated_dbp = a * ptt_values + b
    return estimated_dbp

def plot_estimated_bp(bp_values, label, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(bp_values, label=label)
    plt.xlabel('Measurement Index')
    plt.ylabel('Blood Pressure (mmHg)')
    plt.title(f'Estimated {label} Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'estimated_{label.lower()}.png'))
    plt.close()

def save_estimated_bp(bp_values, label, output_dir):
    np.savetxt(os.path.join(output_dir, f'estimated_{label.lower()}.txt'), bp_values, fmt='%.2f')

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def main():
    
    # Path Subject and Task
    dataset_folder = "/Users/danillugli/Desktop/Boccignone/BP4D+"
    subject = "M001"
    task = "T3/"
    video_path = dataset_folder + f"/{subject}/{task}vid.avi"

    # Path ROI Forehead + Neck
    neck_roi_file_path = f"NIAC/{subject}/{task}neck_roi_dimensions.json"
    forehead_roi_file_path = f"NIAC/{subject}/{task}forehead_roi_dimensions.json"

    # Output Path
    output_dir =  f"NIAC/{subject}/{task}"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    if os.path.exists(neck_roi_file_path):
        neck_roi = load_roi_dimensions(neck_roi_file_path)
    else:
        neck_roi = select_neck_roi(frame)
        save_roi_dimensions(neck_roi, neck_roi_file_path)

    if os.path_exists(forehead_roi_file_path):
        forehead_roi = load_roi_dimensions(forehead_roi_file_path)
    else:
        forehead_roi = select_forehead_roi(frame)
        save_roi_dimensions(forehead_roi, forehead_roi_file_path)

    # Extract Red traces
    red_signal_forehead = extract_red_trace(video_path, forehead_roi, output_dir, "forehead")
    red_signal_neck = extract_red_trace(video_path, neck_roi, output_dir, "neck")

    # Apply Smoothness Priors Detrend filter to forehead signal
    lambda_param = 10
    filtered_red_signal_forehead = smoothness_priors_detrend(red_signal_forehead, lambda_param)

    # Apply Butterworth bandpass filter to neck signal
    low_cutoff_frequency = 0.6
    high_cutoff_frequency = 4.0
    sampling_rate = 30.0  # Assuming a sampling rate of 30 FPS

    filtered_red_signal_neck = butter_bandpass_filter(red_signal_neck, low_cutoff_frequency, high_cutoff_frequency, sampling_rate)

    # Save filtered red signals for forehead
    np.savetxt(f"{output_dir}/ppg_filtered_forehead_R.txt", filtered_red_signal_forehead)
    plt.plot(filtered_red_signal_forehead)
    plt.title('PPG Signal (Filtered - Forehead - R)')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.savefig(f"{output_dir}/ppg_filtered_forehead_R.jpg")
    plt.close()

    # Save filtered red signals for neck
    np.savetxt(f"{output_dir}/ppg_filtered_neck_R.txt", filtered_red_signal_neck)
    plt.plot(filtered_red_signal_neck)
    plt.title('PPG Signal (Filtered - Neck - R)')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.savefig(f"{output_dir}/ppg_filtered_neck_R.jpg")
    plt.close()

    # Find and plot peaks in the filtered red signals
    forehead_peaks = find_peaks_in_signal(filtered_red_signal_forehead)
    neck_peaks = find_peaks_in_signal(filtered_red_signal_neck)

    plot_peaks(filtered_red_signal_forehead, forehead_peaks, 'Forehead PPG (R)', output_dir)
    plot_peaks(filtered_red_signal_neck, neck_peaks, 'Neck PPG (R)', output_dir)

    # Calculate PTT
    ptt = calculate_ptt(forehead_peaks, neck_peaks, sampling_rate, output_dir)
    print("PTT values (in seconds):", ptt)

    # Apply moving average filter to PTT values
    ptt_smoothed = moving_average(ptt, window_size=5)

    # Estimate SBP
    estimated_sbp = estimate_systolic_blood_pressure(ptt_smoothed)
    print("Estimated SBP values (mmHg):", estimated_sbp)

    # Estimate DBP
    estimated_dbp = estimate_diastolic_blood_pressure(ptt_smoothed)
    print("Estimated DBP values (mmHg):", estimated_dbp)

    # Plot and save PTT values and estimated BP values
    plot_ptt_values(ptt_smoothed, output_dir)
    plot_estimated_bp(estimated_sbp, 'SBP', output_dir)
    plot_estimated_bp(estimated_dbp, 'DBP', output_dir)
    save_estimated_bp(estimated_sbp, 'SBP', output_dir)
    save_estimated_bp(estimated_dbp, 'DBP', output_dir)

if __name__ == "__main__":
    main()
