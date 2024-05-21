import os
print("OK_os")
import mediapipe as mp
print("mediapipe importato con successo!")
import numpy as np
print("numpy importato con successo!")
import cv2
print("cv2 importato con successo!")
import matplotlib.pyplot as plt
print("matplotlib importato con successo!")
from scipy.signal import butter, sosfiltfilt, find_peaks
print("scipy importato con successo!")
import heartpy as hp
print("heartpy importato con successo!")


# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Indici M001-T3, dei landmark per la fronte (aggiornati per coprire tutta la fronte, l'ordine dei landmarks ha valore, bisogna seguire il perimetro)
forehead_indices = [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 293, 334, 296, 336, 9, 66, 105, 63, 70]

def get_face(image):
    """
    Funzione per elaborare l'immagine e restituire le coordinate dei landmark della fronte.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    forehead_coords = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            forehead_coords = [(int(face_landmarks.landmark[i].x * image.shape[1]), 
                                int(face_landmarks.landmark[i].y * image.shape[0])) for i in forehead_indices]
    return forehead_coords

def extract_rgb_trace(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    rgb_signals = {'R': [], 'G': [], 'B': []}
    i = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Get forehead coordinates
        forehead_coords = get_face(image)
        
        if forehead_coords:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(forehead_coords, np.int32)], 255)
            mean_val = cv2.mean(image, mask=mask)  # Extract mean values of the RGB channels
            rgb_signals['R'].append(mean_val[2])
            rgb_signals['G'].append(mean_val[1])
            rgb_signals['B'].append(mean_val[0])

            # Draw the forehead ROI
            for idx, coord in enumerate(forehead_coords):
                cv2.putText(image, str(forehead_indices[idx]), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(image, coord, 2, (0, 255, 0), -1)

            forehead_region = np.array(forehead_coords, np.int32)
            cv2.polylines(image, [forehead_region], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.fillPoly(image, [forehead_region], (255, 0, 0, 50))  # Optional: fill the forehead region with a transparent color

            # Display the annotated image
            cv2.imshow('MediaPipe FaceMesh with Forehead Landmarks', image)
            if cv2.waitKey(5) & 0xFF == 27:  # 27 corrisponde al tasto ESC
                break

        i += 1

    cap.release()
    cv2.destroyAllWindows()

    # Convert lists to numpy arrays
    for color in rgb_signals:
        rgb_signals[color] = np.array(rgb_signals[color])

    # Save raw RGB signals
    for color in rgb_signals:
        np.savetxt(f"{output_folder}/ppg_raw_{color}.txt", rgb_signals[color])
        plt.plot(rgb_signals[color])
        plt.title(f'PPG Signal (Raw - {color})')
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        plt.savefig(f"{output_folder}/ppg_raw_{color}.jpg")
        plt.close()

    return rgb_signals

def remove_low_frequency_trends(signal, low_cutoff_frequency, high_cutoff_frequency, sampling_rate):
    # Design a Butterworth filter
    sos_filter = butter(10, [low_cutoff_frequency, high_cutoff_frequency], btype='bp', analog=False, output='sos', fs=sampling_rate)
    # Apply the filter to remove low and high frequency trends
    filtered_signal = sosfiltfilt(sos_filter, signal)
    return filtered_signal

def cpu_CHROM(signal):
    """
    CHROM method on CPU using Numpy.

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. 
    IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """
    X = signal
    print(f"Shape of input signal: {X.shape}")  # Add this line for debugging
    Xcomp = 3*X[:, 0] - 2*X[:, 1]
    Ycomp = (1.5*X[:, 0]) + X[:, 1] - (1.5*X[:, 2])
    print(f"Shape of Xcomp: {Xcomp.shape}, Shape of Ycomp: {Ycomp.shape}")  # Add this line for debugging
    sX = np.std(Xcomp)
    sY = np.std(Ycomp)
    alpha = sX / sY
    bvp = Xcomp - alpha * Ycomp
    return bvp

def calculate_hr_from_bvp(bvp_signal, sampling_rate):
    wd, m = hp.process(bvp_signal, sample_rate=sampling_rate)
    heart_rate = m['bpm']
    return heart_rate, wd

def generate_synthetic_ecg(bvp_peaks, sampling_rate, duration):
    synthetic_ecg = np.zeros(int(duration * sampling_rate))
    for peak in bvp_peaks:
        synthetic_ecg[peak] = 1  # Mettiamo un picco nel segnale sintetico
    return synthetic_ecg

def calculate_ptt(aligned_ecg_signal, aligned_bvp_signal, ecg_peaks, ppg_peaks, sampling_rate):
    ptt_values = []
    for ecg_peak in ecg_peaks:
        # Trova tutti i picchi PPG successivi al picco ECG corrente
        subsequent_ppg_peaks = ppg_peaks[ppg_peaks > ecg_peak]
        if len(subsequent_ppg_peaks) == 0:
            # Se non ci sono picchi PPG successivi, salta questo picco ECG
            continue
        # Trova il primo picco PPG successivo al picco ECG
        ppg_peak = subsequent_ppg_peaks[0]
        ptt = (ppg_peak - ecg_peak) / sampling_rate
        ptt_values.append(ptt)
    return ptt_values

def get_sampling_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def main():
    dataset_folder = "/Users/danillugli/Desktop/Boccignone/BP4D+"
    subject = "F001"
    task = "T1/"
    video_path = dataset_folder + f"/{subject}/{task}vid.avi"
    output_folder = f"NIAC/{subject}/{task}"

    sampling_rate = get_sampling_rate(video_path)
    print(f"Sampling rate: {sampling_rate} FPS")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\nFolder \"{output_folder}\" has been created.")

    # Extract the RGB signals from the video
    rgb_signals = extract_rgb_trace(video_path, output_folder)

    # Combine the RGB signals
    combined_rgb_signals = np.vstack((rgb_signals['R'], rgb_signals['G'], rgb_signals['B'])).T
    print(f"Shape of combined_rgb_signals: {combined_rgb_signals.shape}")  # Add this line for debugging

    # Set the cutoff frequency and apply bandpass filter
    low_cutoff_frequency = 0.7
    high_cutoff_frequency = 4.0

    filtered_rgb_signals = {}
    for color in rgb_signals:
        filtered_rgb_signals[color] = remove_low_frequency_trends(rgb_signals[color], low_cutoff_frequency, high_cutoff_frequency, sampling_rate)

        # Save filtered RGB signals
        np.savetxt(f"{output_folder}/ppg_filtered_{color}.txt", filtered_rgb_signals[color])
        plt.plot(filtered_rgb_signals[color])
        plt.title(f'PPG Signal (Filtered - {color})')
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        plt.savefig(f"{output_folder}/ppg_filtered_{color}.jpg")
        plt.close()

    # Calculate BVP signal using CHROM method
    bvp_signal = cpu_CHROM(combined_rgb_signals)

    # Save BVP signal
    np.savetxt(f"{output_folder}/bvp_signal.txt", bvp_signal)
    plt.plot(bvp_signal)
    plt.title('BVP Signal')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.savefig(f"{output_folder}/bvp_signal.jpg")
    plt.close()

    heart_rate, wd = calculate_hr_from_bvp(bvp_signal, sampling_rate)

    ppg_peaks, _ = find_peaks(bvp_signal, distance=sampling_rate/2)

    duration = len(bvp_signal) / sampling_rate
    synthetic_ecg = generate_synthetic_ecg(ppg_peaks, sampling_rate, duration)

    ecg_peaks = np.where(synthetic_ecg == 1)[0]
    ptt_values = calculate_ptt(synthetic_ecg, bvp_signal, ecg_peaks, ppg_peaks, sampling_rate)

    # Salva i valori PTT
    np.savetxt(f"{output_folder}/ptt_values.txt", ptt_values)
    plt.plot(ptt_values)
    plt.title('PTT Values')
    plt.xlabel('Heartbeat')
    plt.ylabel('PTT (seconds)')
    plt.savefig(f"{output_folder}/ptt_values.jpg")
    plt.close()

    print(f"PTT values calculated and saved in {output_folder}")


    wd, m = hp.process(bvp_signal, sample_rate=sampling_rate)
    hp.plotter(wd, m)

    # Save HeartPy analysis results
    with open(f"{output_folder}/heartpy_results.txt", "w") as f:
        for key, value in m.items():
            f.write(f"{key}: {value}\n")

    print(f"Heart rate: {m['bpm']} BPM")


if __name__ == "__main__":
    main()
