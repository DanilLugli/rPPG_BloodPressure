# --- LIBRARY IMPORTS ---

# --- STANDARD LIBRARIES ---
import os
import json

# --- SCIENTIFIC COMPUTING & DATA ANALYSIS ---
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.signal import butter, sosfiltfilt, filtfilt, find_peaks, resample
from scipy.stats import pearsonr, ttest_ind, iqr
import time

# --- COMPUTER VISION ---
import cv2
import mediapipe as mp

# --- DATA VISUALIZATION ---
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully!")

from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


print("Library imported successfully!")

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Indici dei landmark per la fronte (aggiornati per coprire tutta la fronte)
forehead_indices = [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 293, 334, 296, 336, 9, 66, 105, 63, 70]

# --- MANAGE VIDEO ---
def get_sampling_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_actual_fps(video_path):
    """ Calcola il vero FPS di un video analizzando il tempo tra i frame. """
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("[ERROR] Impossibile aprire il video!")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Numero totale di frame
    nominal_fps = cap.get(cv2.CAP_PROP_FPS)  # FPS dichiarato dal video

    print(f"[INFO] FPS nominale: {nominal_fps} FPS")
    print(f"[INFO] Frame totali: {frame_count}")

    # Calcolo FPS effettivo
    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

    end_time = time.time()
    duration = end_time - start_time
    actual_fps = frame_idx / duration if duration > 0 else 0

    cap.release()

    print(f"[INFO] Durata della lettura: {duration:.2f} secondi")
    print(f"[INFO] FPS effettivo calcolato: {actual_fps:.2f} FPS")

    return actual_fps

def read_blood_pressure_data(file_path):
    """
    Legge i valori di pressione sanguigna da un file di testo.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File non trovato: {file_path}")
        return np.array([])
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        bp_data = np.array([float(line.strip()) for line in lines])
    return bp_data

# --- ROI SELECTION AND FILE MANAGEMENT ---
def select_neck_roi(frame):
    r = cv2.selectROI("Select Neck ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Neck ROI")
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

import cv2
import json
import os

def track_neck_roi(video_path, neck_roi_file_path, output_roi_path):
    """
    Implementa il tracking adattivo della ROI della carotide su un video.
    Se la ROI è già salvata, la carica e applica il tracking. Altrimenti, chiede all'utente di selezionarla.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("[ERROR] Could not read first frame!")

    # Controlla se esiste già una ROI salvata
    if os.path.exists(neck_roi_file_path):
        with open(neck_roi_file_path, 'r') as file:
            roi_data = json.load(file)
            roi = (roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height'])
        print("[INFO] ROI loaded from file:", roi)
    else:
        print("[INFO] Select the initial ROI manually (focus on the left side near the carotid)")
        roi = cv2.selectROI("Select Neck ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        # Salva la ROI selezionata
        roi_data = {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
        with open(neck_roi_file_path, 'w') as file:
            json.dump(roi_data, file)
        print("[INFO] ROI saved:", roi_data)

    # Inizializza il tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, roi)

    roi_updates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, roi = tracker.update(frame)

        if success:
            x, y, w, h = map(int, roi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_updates.append({"x": x, "y": y, "width": w, "height": h})
        else:
            print("[WARNING] Tracking lost in this frame!")

        cv2.imshow("Tracking Neck ROI", frame)
        if cv2.waitKey(20) & 0xFF == 27:  # ESC per uscire
            break

    cap.release()
    cv2.destroyAllWindows()

    # Salva le coordinate aggiornate della ROI per tutti i frame
    with open(output_roi_path, 'w') as file:
        json.dump(roi_updates, file)

    print(f"[INFO] Tracking completed! Updated ROIs saved in {output_roi_path}")
    return roi_updates

# --- FACE SIGNAL RGB EXTRACTION ---
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

def extract_rgb_trace_MediaPipe(video_path, output_folder):
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
        np.savetxt(f"{output_folder}/rgb_raw_{color}_Forehead.txt", rgb_signals[color])
        plt.plot(rgb_signals[color])
        plt.title(f'RGB Signal (Raw - {color})')
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        plt.savefig(f"{output_folder}/rgb_raw_{color}_Forehead.jpg")
        plt.close()

    return rgb_signals

# --- NECK PPW EXTRACTION USING PIXFLOW ---
def extract_pixflow_signal_improved(video_path, roi, output_folder, part, 
                                    fs=30.0, lowcut=0.5, highcut=3.0):
    """
    Estrae il segnale PPW dal collo usando il flusso ottico, ma calcola la magnitudo
    del flusso (invece della sola componente x). Applica un filtro passa-banda.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read the first frame.")

    x, y, w, h = roi
    prev_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    flow_signal = []

    # Parametri Farneback (puoi modificarli per ridurre il rumore)
    pyr_scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    flags = 0  # default

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        curr_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        # Calcolo del flusso ottico (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                            None, pyr_scale, levels, winsize, 
                                            iterations, poly_n, poly_sigma, flags)
        # Ricava la componente orizzontale (u) e verticale (v)
        u = flow[..., 0]
        v = flow[..., 1]
        
        # Calcola la magnitudo del flusso per ogni pixel
        mag = np.sqrt(u**2 + v**2)
        
        # Media della magnitudo su tutta la ROI
        avg_flow_mag = np.mean(mag)
        flow_signal.append(avg_flow_mag)

        prev_gray = curr_gray

    cap.release()

    # Converti in array numpy
    flow_signal = np.array(flow_signal)

    # **Filtraggio passa-banda** per isolare la componente cardiaca
    filtered_signal = bandpass_filter(flow_signal, fs=fs, lowcut=lowcut, highcut=highcut)

    # Salva il segnale grezzo e quello filtrato
    np.savetxt(f"{output_folder}/ppw_signal_{part}_raw.txt", flow_signal)
    np.savetxt(f"{output_folder}/ppw_signal_{part}_filtered.txt", filtered_signal)

    # Plot del segnale grezzo
    plt.figure(figsize=(10,4))
    plt.plot(flow_signal, label='Raw Flow Magnitude')
    plt.title(f'Raw Flow Signal ({part})')
    plt.xlabel('Frame')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.savefig(f"{output_folder}/ppw_signal_{part}_raw.jpg")
    plt.close()

    # Plot del segnale filtrato
    plt.figure(figsize=(10,4))
    plt.plot(filtered_signal, color='orange', label='Filtered Flow Magnitude')
    plt.title(f'Filtered Flow Signal ({part})')
    plt.xlabel('Frame')
    plt.ylabel('Magnitude (filtered)')
    plt.legend()
    plt.savefig(f"{output_folder}/ppw_signal_{part}_filtered.jpg")
    plt.close()

    return filtered_signal

def extract_rgb_trace(video_path, roi, output_folder, part):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Cannot open video file: {video_path}")

    rgb_signals = {'R': [], 'G': [], 'B': []}

    frame_count = 0  # ✅ Contiamo i frame per debug

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        x, y, w, h = roi

        # ✅ Verifica se la ROI è dentro i limiti dell'immagine
        if x + w > image.shape[1] or y + h > image.shape[0]:
            print(f"[ERROR] ROI {roi} out of bounds! Image size: {image.shape}")
            break

        roi_frame = image[y:y+h, x:x+w]

        if roi_frame.size == 0:
            print(f"[ERROR] Empty ROI frame at frame {frame_count}")
            continue

        mean_val = cv2.mean(roi_frame)[:3]  # Extract mean values of the RGB channels
        rgb_signals['R'].append(mean_val[2])
        rgb_signals['G'].append(mean_val[1])
        rgb_signals['B'].append(mean_val[0])

        frame_count += 1  # Incrementa il conteggio dei frame

    cap.release()

    if frame_count == 0:
        print("[ERROR] No frames processed! Check video file or ROI.")
        return {'R': [], 'G': [], 'B': []}

    print(f"[INFO] Processed {frame_count} frames for {part}")

    # Convert lists to numpy arrays
    for color in rgb_signals:
        rgb_signals[color] = np.array(rgb_signals[color])

    # **Salvataggio dei segnali RGB e generazione dei grafici**
    for color in rgb_signals:
        file_path = f"{output_folder}/rgb_raw_{color}_{part}.txt"
        np.savetxt(file_path, rgb_signals[color], fmt="%.2f")

        plt.figure(figsize=(8, 4))
        plt.plot(rgb_signals[color], label=f"RGB {color} ({part})", color=color.lower())
        plt.xlabel("Frame")
        plt.ylabel("Intensity")
        plt.title(f"RGB Signal ({color} - {part})")
        plt.legend()
        plt.savefig(f"{output_folder}/rgb_raw_{color}_{part}.jpg")
        plt.close()

    return rgb_signals

# --- SIGNAL PREPROCESSING AND CLEANING ---
def smoothness_priors_detrend(signal, lambda_param=10):
    n = len(signal)
    if n < 2:
        print("[WARNING] Signal too short for detrending. Returning original signal.")
        return signal
    I = sparse.eye(n)
    D = sparse.eye(n, n, 1) - sparse.eye(n, n)
    D = D[1:]
    H = I + lambda_param * D.T @ D
    H = H.tocsc()
    trend = splinalg.spsolve(H, signal)
    detrended_signal = signal - trend
    return detrended_signal

def butter_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    if len(signal) <= 27:
        print("[WARNING] Signal too short for bandpass filtering. Returning original signal.")
        return signal
    sos = butter(order, [lowcut, highcut], btype='band', output='sos', fs=fs)
    filtered_signal = sosfiltfilt(sos, signal, axis=0)
    return filtered_signal

def butter_lowpass_filter(signal, cutoff=3.0, fs=30.0, order=4):
    nyq = 0.5 * fs  # Frequenza di Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def bandpass_filter(signal, fs=30.0, lowcut=0.5, highcut=3.0, order=4):
    """
    Applica un filtro passa-banda Butterworth al segnale.
    
    :param signal: array monodimensionale con il segnale grezzo
    :param fs: frequenza di campionamento (frame rate del video)
    :param lowcut: frequenza di taglio inferiore (Hz)
    :param highcut: frequenza di taglio superiore (Hz)
    :param order: ordine del filtro
    :return: segnale filtrato
    """
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    filtered_signal = sosfiltfilt(sos, signal)
    return filtered_signal

def validate_ptt(ptt_values):
    lower_limit = 0.08  # Ridotto a 0.08s per non eliminare troppi valori validi
    upper_limit = 0.4   # Aumentato per includere potenziali PTT più realistici

    filtered_ptt = [ptt for ptt in ptt_values if lower_limit <= ptt <= upper_limit]
    removed_outliers = len(ptt_values) - len(filtered_ptt)

    print(f"[INFO] Removed {removed_outliers} outliers from PTT values.")
    print(f"[INFO] Final valid PTT values: {filtered_ptt}")

    return np.array(filtered_ptt)

from scipy.stats import iqr
from scipy.signal import find_peaks
from scipy.stats import iqr
import numpy as np

def find_peaks_in_signal(ppg_signal, fps=30, window_size=5, iqr_factor=1.2, output_dir=None, label="signal"):
    """
    Migliorato il rilevamento e la verifica dei picchi nel segnale BVP.

    - Filtro passa-basso Butterworth per ridurre il rumore senza distorcere i picchi
    - Adattamento della prominenza basato sulla mediana delle prominenze rilevate
    - Regolazione dinamica della distanza minima tra i picchi in base alla frequenza cardiaca
    - Filtro IQR migliorato per ridurre gli outlier
    - **Verifica della distanza tra i picchi** per evitare errori di calcolo
    
    :param ppg_signal: Array con il segnale BVP
    :param fps: Frame rate del video (default: 30 FPS)
    :param window_size: Dimensione della finestra per il filtro passa-basso
    :param iqr_factor: Fattore per il filtro IQR
    :param output_dir: Directory di output per salvare i risultati
    :param label: Etichetta per identificare i file di output
    :return: Indici dei picchi validi nel segnale
    """

    smoothed_signal = butter_lowpass_filter(ppg_signal, cutoff=3.0, fs=fps)

    # 🔹 1. ADATTAMENTO AUTOMATICO DELLA PROMINENZA
    prominence_value = 0.2 * np.std(smoothed_signal)  # Default value
    peaks, properties = find_peaks(smoothed_signal, prominence=prominence_value)

    if len(properties["prominences"]) > 0:
        prominence_value = np.median(properties["prominences"]) * 0.5
        peaks, properties = find_peaks(smoothed_signal, prominence=prominence_value)

    # 🔹 2. DISTANZA MINIMA TRA I PICCHI IN BASE ALLA FREQUENZA CARDIACA
    min_distance = int(fps * 60 / 120)  # 120 BPM
    peaks, properties = find_peaks(smoothed_signal, distance=min_distance, prominence=prominence_value)

    # 🔹 3. FILTRO OUTLIER USANDO IQR
    peak_heights = smoothed_signal[peaks]
    q1, q3 = np.percentile(peak_heights, [25, 75])
    iqr_value = iqr(peak_heights)
    lower_bound = q1 - iqr_factor * iqr_value
    upper_bound = q3 + iqr_factor * iqr_value
    valid_peaks = peaks[(peak_heights >= lower_bound) & (peak_heights <= upper_bound)]

    print(f"[INFO] Rilevati {len(peaks)} picchi iniziali, ridotti a {len(valid_peaks)} dopo il filtro IQR.")

    # 🔹 4. VERIFICA DELLA DISTANZA TRA I PICCHI
    if len(valid_peaks) > 1:
        peak_intervals = np.diff(valid_peaks) / fps  # Converti in secondi
        mean_interval = np.mean(peak_intervals)
        min_interval = np.min(peak_intervals)
        max_interval = np.max(peak_intervals)

        print(f"[DEBUG] Distanza media tra i picchi: {mean_interval:.3f} s")
        print(f"[DEBUG] Distanza minima: {min_interval:.3f} s, massima: {max_interval:.3f} s")

        if mean_interval < 0.4 or mean_interval > 2.0:
            print(f"[WARNING] La distanza media tra i picchi ({mean_interval:.3f} s) è anomala!")

    # 🔹 5. GRAFICO DEI PICCHI SOVRAPPOSTI AL SEGNALE
    if output_dir:
        plt.figure(figsize=(12, 5))
        plt.plot(smoothed_signal, label="BVP Signal", color="blue")
        plt.scatter(valid_peaks, smoothed_signal[valid_peaks], color="red", marker="x", label="Detected Peaks")
        plt.title(f"BVP Signal with Peaks ({label})")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plot_file = os.path.join(output_dir, f"peaks_verification_{label}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"[INFO] Peak verification plot saved in: {plot_file}")

    # 🔹 6. SALVATAGGIO PICCHI SU FILE
    if output_dir:
        output_file = os.path.join(output_dir, f'peaks_{label}.txt')
        np.savetxt(output_file, valid_peaks, fmt='%d')
        print(f"[INFO] Peaks saved in: {output_file}")

    return valid_peaks

# --- FEATURE EXTRACTION FROM CLEANED SIGNAL ---
import os

def cpu_CHROM(signal, output_dir, label):
    """
    CHROM method on CPU using Numpy.
    """
    X = signal
    Xcomp = 3 * X[:, 0] - 2 * X[:, 1]
    Ycomp = 1.5 * X[:, 0] + X[:, 1] - 1.5 * X[:, 2]
    sX = np.std(Xcomp, axis=0)
    sY = np.std(Ycomp, axis=0)
    alpha = sX / sY
    bvp = Xcomp - alpha * Ycomp

    # Salva il BVP su file
    output_file = os.path.join(output_dir, f"bvp_chrom_signal_{label}.txt")
    np.savetxt(output_file, bvp, delimiter=",", header="BVP Signal", comments="")

    # Genera il grafico
    plot_file = os.path.join(output_dir, f"bvp_chrom_plot_{label}.png")
    plt.figure(figsize=(12, 5))
    plt.plot(bvp, label=f"BVP CHROM ({label})", color="blue")
    plt.title(f"BVP Signal (CHROM Method - {label})")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(plot_file)
    plt.close()

    print(f"[INFO] BVP CHROM ({label}) salvato in: {output_file}")
    print(f"[INFO] Grafico BVP CHROM ({label}) salvato in: {plot_file}")

    return bvp

def compute_bvp_fft(signal, sampling_rate, output_dir):
    """
    Calcola il segnale BVP usando la FFT e il filtraggio dei picchi.
    
    Parameters:
        signal (array): Segnale filtrato
        sampling_rate (int): Frequenza di campionamento in Hz
        output_dir (str): Directory di output per salvare i risultati
    
    Returns:
        bvp_signal (array): Segnale BVP ricostruito
        peaks (array): Indici dei picchi rilevati nel segnale BVP
    """
    # Assicurarsi che il segnale sia monodimensionale
    signal = np.asarray(signal).flatten()
    
    # Calcolo della FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
    fft_values = np.fft.rfft(signal)
    
    # Assicurarsi che valid_freqs abbia la stessa lunghezza di fft_values
    valid_freqs = (freqs >= 0.5) & (freqs <= 3.0)
    valid_freqs = valid_freqs[:len(fft_values)]  # Adatta la dimensione per evitare mismatch
    fft_values[~valid_freqs] = 0
    
    # Ricostruzione del segnale con IFFT
    bvp_signal = np.fft.irfft(fft_values, n=n)
    
    # Rilevamento dei picchi
    peaks, _ = find_peaks(bvp_signal, distance=sampling_rate/2.0)  # Almeno mezzo secondo tra i picchi
    
    # Salva il BVP su file
    output_file = f"{output_dir}/bvp_fft.txt"
    np.savetxt(output_file, bvp_signal, delimiter=",", header="BVP Signal", comments="")
    
    # Genera il grafico
    plot_file = f"{output_dir}/bvp_fft_plot.png"
    plt.figure(figsize=(12, 5))
    plt.plot(bvp_signal, label="BVP FFT", color="red")
    plt.scatter(peaks, bvp_signal[peaks], color='black', marker='o', label="Detected Peaks")
    plt.title("BVP Signal (FFT Method)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(plot_file)
    plt.close()
    
    print(f"BVP FFT salvato in: {output_file}")
    print(f"Grafico BVP FFT salvato in: {plot_file}")
    
    return bvp_signal, peaks

def compute_bvp_green_neck(filtered_green_signal, sampling_rate, output_dir, label):
    """
    Calcola il BVP a partire dal segnale verde filtrato del collo.
    
    Il segnale filtrato viene normalizzato (sottraendo la media) per centrarlo attorno a zero.
    Viene poi salvato il segnale BVP e viene generato un grafico.
    Inoltre, si rilevano i picchi (assumendo un intervallo minimo di circa 0.5 secondi tra i battiti)
    per verificare la regolarità della pulsazione.
    
    :param filtered_green_signal: Array 1D contenente il segnale verde filtrato.
    :param sampling_rate: Frequenza di campionamento (Hz).
    :param output_dir: Directory in cui salvare i risultati.
    :param label: Etichetta per identificare i file di output.
    :return: bvp (il segnale BVP) e peaks (gli indici dei picchi rilevati).
    """
    # Normalizza il segnale rimuovendo la media
    bvp = filtered_green_signal - np.mean(filtered_green_signal)
    
    # Salva il segnale BVP in un file di testo
    output_file = os.path.join(output_dir, f"bvp_green_neck_{label}.txt")
    np.savetxt(output_file, bvp, delimiter=",", header="BVP Green Neck Signal", comments="")
    print(f"[INFO] BVP Green Neck ({label}) salvato in: {output_file}")
    
    
    # Genera e salva il grafico del segnale BVP
    plot_file = os.path.join(output_dir, f"bvp_green_neck_{label}.png")
    plt.figure(figsize=(12, 5))
    plt.plot(bvp, label=f"BVP Green Neck ({label})", color="green")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(f"BVP Signal from Filtered Green Neck Signal ({label})")
    plt.legend()
    plt.savefig(plot_file)
    plt.close()
    print(f"[INFO] Grafico BVP Green Neck ({label}) salvato in: {plot_file}")
    
    # Rileva i picchi: assumiamo che il tempo minimo tra i battiti sia 0.5 secondi
    min_distance = int(sampling_rate * 0.5)
    peaks, _ = find_peaks(bvp, distance=min_distance)
    
    # Salva gli indici dei picchi in un file
    peaks_file = os.path.join(output_dir, f"bvp_green_neck_peaks_{label}.txt")
    np.savetxt(peaks_file, peaks, fmt='%d')
    print(f"[INFO] Picchi BVP Green Neck ({label}) salvati in: {peaks_file}")
    
    return bvp, peaks


from scipy.interpolate import interp1d
import numpy as np
def calculate_ptt_cross_correlation(peaks_forehead, peaks_neck, fps, output_dir, label):
    """
    Calcola il Pulse Transit Time (PTT) utilizzando la cross-correlazione tra i segnali binari
    ottenuti dai picchi della fronte e del collo.
    
    :param peaks_forehead: Picchi del segnale della fronte (in frame).
    :param peaks_neck: Picchi del segnale del collo (in frame).
    :param fps: Frame rate del video per la conversione dei lag in secondi.
    :param output_dir: Directory di output per salvare i risultati.
    :param label: Etichetta per identificare i file di output.
    :return: Array contenente il valore PTT stimato (se rientra nel range fisiologico).
    """
    if len(peaks_forehead) == 0 or len(peaks_neck) == 0:
        print("[ERROR] Uno dei due segnali non ha picchi!")
        return np.array([])

    # Definisci la lunghezza del segnale binario: usa il massimo indice di picco +1
    max_length = int(max(np.max(peaks_forehead), np.max(peaks_neck)) + 1)
    binary_forehead = np.zeros(max_length)
    binary_neck = np.zeros(max_length)
    binary_forehead[peaks_forehead] = 1
    binary_neck[peaks_neck] = 1

    # Calcola la cross-correlazione completa
    correlation = np.correlate(binary_forehead, binary_neck, mode='full')
    lags = np.arange(-max_length + 1, max_length)

    print(f"[DEBUG] Cross-correlation (full): {correlation}")
    print(f"[DEBUG] Lags (full): {lags}")

    # Considera solo lag positivi (cioè, i picchi del collo ritardati rispetto a quelli della fronte)
    # e limitiamo il range a [min_lag, max_lag] in frame
    min_lag = int(0.05 * fps)
    max_lag = int(0.8 * fps)
    valid_indices = (lags >= min_lag) & (lags <= max_lag)
    if not np.any(valid_indices):
        print("[WARNING] Nessun lag valido trovato!")
        return np.array([])

    correlation_valid = correlation[valid_indices]
    lags_valid = lags[valid_indices]

    print(f"[DEBUG] Valid Lags: {lags_valid}")
    print(f"[DEBUG] Correlation (valid): {correlation_valid}")

    # Trova il lag con la massima correlazione
    best_lag = lags_valid[np.argmax(correlation_valid)]
    ptt_value = best_lag / fps
    print(f"[DEBUG] Lag with max correlation: {best_lag}")
    print(f"[DEBUG] PTT Value: {ptt_value:.3f} s")

    # Controlla il range fisiologico
    if 0.05 <= ptt_value <= 0.8:
        ptt_values = np.array([ptt_value])
    else:
        ptt_values = np.array([])
        print(f"[WARNING] PTT={ptt_value:.3f}s fuori range fisiologico!")

    if ptt_values.size > 0:
        print(f"[INFO] PTT Mean: {np.mean(ptt_values):.3f}s, Std Dev: {np.std(ptt_values):.3f}s")

    # Salva i valori PTT su file
    output_file = os.path.join(output_dir, f'ptt_values_{label}.txt')
    np.savetxt(output_file, ptt_values, fmt='%.5f')
    print(f"[INFO] PTT values saved in: {output_file}")

    # Genera il grafico dei valori PTT
    plot_file = os.path.join(output_dir, f'ptt_values_{label}.png')
    plt.figure(figsize=(10, 4))
    plt.plot(ptt_values, 'bo-', label=f'PTT Values ({label})')
    plt.xlabel('Measurement Index')
    plt.ylabel('PTT (seconds)')
    plt.title(f'PTT Values ({label}) Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.close()
    print(f"[INFO] PTT plot saved in: {plot_file}")

    return ptt_values

import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_ptt_direct(peaks_forehead, peaks_neck, fps, output_dir, label, signal_neck=None):
    """
    Calcola il Pulse Transit Time (PTT) come differenza tra i tempi dei picchi della fronte e del collo,
    affinando il calcolo tramite interpolazione quadratica attorno al picco del collo (se il segnale
    grezzo 'signal_neck' è fornito).
    
    Per ogni picco della fronte si cerca il primo picco del collo successivo (usando np.searchsorted) e, se disponibile,
    si esegue un'interpolazione quadratica attorno a quel picco per ottenere una stima sub-campionata.
    
    Solo i valori PTT compresi tra 0.1 e 0.5 secondi vengono considerati validi.
    
    :param peaks_forehead: Array dei picchi del segnale della fronte (in frame).
    :param peaks_neck: Array dei picchi del segnale del collo (in frame).
    :param fps: Frame rate del video.
    :param output_dir: Directory in cui salvare i risultati.
    :param label: Etichetta per identificare i file di output.
    :param signal_neck: (Opzionale) Array del segnale grezzo del collo, usato per l'interpolazione.
    :return: Array dei valori PTT (in secondi) validi.
    """
    if len(peaks_forehead) == 0 or len(peaks_neck) == 0:
        print("[ERROR] Uno dei due segnali non ha picchi!")
        return np.array([])

    # Converti i picchi da frame a tempi in secondi
    times_forehead = np.array(peaks_forehead) / fps
    times_neck = np.array(peaks_neck) / fps

    print(f"[DEBUG] Times Forehead: {times_forehead}")
    print(f"[DEBUG] Times Neck (raw): {times_neck}")

    ptt_values = []
    # Per ogni picco della fronte, associa il primo picco del collo che lo segue
    for t_f in times_forehead:
        # Trova l'indice nel vettore dei picchi del collo dove il tempo è maggiore di t_f
        idx = np.searchsorted(times_neck, t_f, side='left')
        if idx < len(times_neck):
            # Se il segnale grezzo del collo è disponibile, esegui l'interpolazione quadratica
            if signal_neck is not None and idx >= 1 and idx < len(signal_neck)-1:
                y0 = signal_neck[idx-1]
                y1 = signal_neck[idx]
                y2 = signal_neck[idx+1]
                denominator = (y0 - 2*y1 + y2)
                if denominator != 0:
                    delta = 0.5 * (y0 - y2) / denominator
                else:
                    delta = 0
                # Calcola il tempo del picco del collo con la stima sub-campionata
                t_n = (peaks_neck[idx] + delta) / fps
            else:
                t_n = times_neck[idx]
                
            ptt = t_n - t_f
            # Modifica del filtro: range accettabile 0.1-0.5 secondi
            if 0.1 <= ptt <= 0.5:
                ptt_values.append(ptt)
            else:
                print(f"[WARNING] PTT={ptt:.3f}s fuori range fisiologico (0.1-0.5 s) per t_f={t_f:.3f}s, t_n={t_n:.3f}s")
        else:
            break  # Non ci sono più picchi del collo dopo questo

    ptt_values = np.array(ptt_values)

    if len(ptt_values) == 0:
        print("[WARNING] Nessun valore PTT rientra nel range fisiologico (0.1-0.5 s)!")
        return np.array([])

    print(f"[INFO] PTT Mean: {np.mean(ptt_values):.3f}s, Std Dev: {np.std(ptt_values):.3f}s")

    # Salva i valori PTT su file
    output_file = os.path.join(output_dir, f'ptt_values_{label}.txt')
    np.savetxt(output_file, ptt_values, fmt='%.5f')
    print(f"[INFO] PTT values saved in: {output_file}")

    # Genera il grafico dei valori PTT
    plot_file = os.path.join(output_dir, f'ptt_values_{label}.png')
    plt.figure(figsize=(10, 4))
    plt.plot(ptt_values, 'bo-', label=f'PTT Values ({label})')
    plt.xlabel('Measurement Index')
    plt.ylabel('PTT (seconds)')
    plt.title(f'PTT Values ({label}) Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.close()
    print(f"[INFO] PTT plot saved in: {plot_file}")

    return ptt_values

def calculate_aix(bvp_forehead, bvp_neck, peaks_forehead, peaks_neck, fs, output_dir):
    """
    Calcola l'Augmentation Index (AIx) utilizzando i segnali BVP della fronte e del collo.

    :param bvp_forehead: Segnale BVP estratto dalla fronte.
    :param bvp_neck: Segnale BVP estratto dal collo.
    :param peaks_forehead: Indici dei picchi nel segnale BVP della fronte.
    :param peaks_neck: Indici dei picchi nel segnale BVP del collo.
    :param fs: Frequenza di campionamento.
    :param output_dir: Directory in cui salvare i risultati.
    :return: Array di valori AIx.
    """
    aix_values = []

    # Assicurati che i segnali abbiano la stessa lunghezza
    min_length = min(len(bvp_forehead), len(bvp_neck))
    bvp_forehead = bvp_forehead[:min_length]
    bvp_neck = bvp_neck[:min_length]

    # Allinea i segnali utilizzando l'interpolazione se necessario
    if len(bvp_forehead) != len(bvp_neck):
        bvp_neck = resample(bvp_neck, len(bvp_forehead))

    for i in range(len(peaks_forehead)-1):
        # Estrai il battito dalla fronte
        start_idx = peaks_forehead[i]
        end_idx = peaks_forehead[i+1]
        beat_forehead = bvp_forehead[start_idx:end_idx]

        if len(beat_forehead) == 0:
            continue  # Salta se il battito è vuoto

        # Trova il picco sistolico (P1) e la pressione diastolica (P_diastolica)
        P1 = np.max(beat_forehead)
        P_diastolic = np.min(beat_forehead)

        # Estrai il battito corrispondente dal collo
        # Trova il picco nel collo che si verifica tra start_idx e end_idx
        neck_peaks_in_beat = peaks_neck[(peaks_neck >= start_idx) & (peaks_neck < end_idx)]
        if len(neck_peaks_in_beat) == 0:
            continue  # Nessun picco nel collo corrispondente
        P2_idx = neck_peaks_in_beat[0]
        P2 = bvp_neck[P2_idx]

        # Calcola PP (Pulse Pressure) usando le ampiezze relative
        PP = P1 - P_diastolic

        # Calcola AIx
        if PP != 0:
            AIx = ((P2 - P1) / PP) * 100
            aix_values.append(AIx)

    aix_values = np.array(aix_values)
    np.savetxt(os.path.join(output_dir, 'aix_values.txt'), aix_values, fmt='%.2f')

    # Plot dei valori di AIx
    plt.figure(figsize=(10, 4))
    plt.plot(aix_values, label='AIx Values')
    plt.xlabel('Beat Index')
    plt.ylabel('AIx (%)')
    plt.title('Augmentation Index Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'aix_values.png'))
    plt.close()

    print(f"[INFO] AIx values calculated and saved: {aix_values}")

    return aix_values

def calculate_aix_red(bvp_forehead_red, bvp_neck, peaks_forehead, peaks_neck, fs, output_dir):
    """
    Calcola l'Augmentation Index (AIx) utilizzando il segnale BVP estratto SOLO dal canale ROSSO della fronte
    e il segnale BVP del collo.

    :param bvp_forehead_red: Segnale BVP della fronte estratto dal canale rosso.
    :param bvp_neck: Segnale BVP estratto dal collo.
    :param peaks_forehead: Indici dei picchi nel segnale BVP della fronte (rosso).
    :param peaks_neck: Indici dei picchi nel segnale BVP del collo.
    :param fs: Frequenza di campionamento.
    :param output_dir: Directory in cui salvare i risultati.
    :return: Array di valori AIx.
    """
    aix_values = []

    # Allinea i segnali se necessario
    min_length = min(len(bvp_forehead_red), len(bvp_neck))
    bvp_forehead_red = bvp_forehead_red[:min_length]
    bvp_neck = bvp_neck[:min_length]

    for i in range(len(peaks_forehead)-1):
        start_idx = peaks_forehead[i]
        end_idx = peaks_forehead[i+1]
        beat_forehead = bvp_forehead_red[start_idx:end_idx]
        if len(beat_forehead) == 0:
            continue
        # Trova il picco sistolico (P1) e il minimo (P_diastolic) nel battito della fronte (rosso)
        P1 = np.max(beat_forehead)
        P_diastolic = np.min(beat_forehead)
        # Nel segnale del collo, per il battito corrente, prendi il primo picco disponibile
        neck_peaks_in_beat = peaks_neck[(peaks_neck >= start_idx) & (peaks_neck < end_idx)]
        if len(neck_peaks_in_beat) == 0:
            continue
        P2_idx = neck_peaks_in_beat[0]
        P2 = bvp_neck[P2_idx]
        PP = P1 - P_diastolic
        if PP != 0:
            AIx = ((P2 - P1) / PP) * 100
            aix_values.append(AIx)
    aix_values = np.array(aix_values)
    np.savetxt(os.path.join(output_dir, 'aix_values_red.txt'), aix_values, fmt='%.2f')

    plt.figure(figsize=(10, 4))
    plt.plot(aix_values, label='AIx Values (Red Channel)')
    plt.xlabel('Beat Index')
    plt.ylabel('AIx (%)')
    plt.title('Augmentation Index (Red Channel Only) Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'aix_values_red.png'))
    plt.close()

    print(f"[INFO] AIx values (red channel) calculated and saved: {aix_values}")
    return aix_values


def calculate_ptt_ppg_forehead_neck(peaks_forehead, peaks_neck_ppg, fps, output_dir):
    """
    Calcola il Pulse Transit Time (PTT) tra il PPG della fronte e il PPG del collo.
    """
    peaks_forehead = np.array(peaks_forehead) / fps
    peaks_neck_ppg = np.array(peaks_neck_ppg) / fps

    if len(peaks_forehead) == 0 or len(peaks_neck_ppg) == 0:
        print("[ERROR] No peaks detected in one of the signals!")
        return np.array([])

    ptt_values = []
    for t_n in peaks_neck_ppg:
        t_f_candidates = peaks_forehead[peaks_forehead > t_n]

        if len(t_f_candidates) > 0:
            t_f = min(t_f_candidates, key=lambda t: abs(t - t_n))
            ptt_value = t_f - t_n
            if 0.1 <= ptt_value <= 0.4:
                ptt_values.append(ptt_value)

    ptt_values = np.array(ptt_values)
    np.savetxt(os.path.join(output_dir, 'ptt_values_PPG_PPG.txt'), ptt_values, fmt='%.5f')
    return ptt_values

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def interpolate_to_length(values, target_length):
    x = np.linspace(0, len(values) - 1, len(values))
    x_new = np.linspace(0, len(values) - 1, target_length)
    interpolated_values = np.interp(x_new, x, values)
    return interpolated_values

# --- BLOOD PRESSURE ESTIMATION ---
def estimate_systolic_blood_pressure(ptt_values, gender):
    if gender == 'male':
        a_sbp = -100  # Coefficiente per la stima della SBP negli uomini
        b_sbp = 120   # Intercetta per la stima della SBP negli uomini
    elif gender == 'female':
        a_sbp = -90   # Coefficiente per la stima della SBP nelle donne
        b_sbp = 110   # Intercetta per la stima della SBP nelle donne
    else:
        raise ValueError("Gender not recognized. Please specify 'male' or 'female'.")
    ptt_values = np.array(ptt_values)  
    estimated_sbp = a_sbp * ptt_values + b_sbp
    return estimated_sbp

def estimate_diastolic_blood_pressure(ptt_values, gender):

    if gender == 'male':
        a_dbp = -75   # Coefficiente per la stima della DBP negli uomini
        b_dbp = 80    # Intercetta per la stima della DBP negli uomini
    elif gender == 'female':
        a_dbp = -70   # Coefficiente per la stima della DBP nelle donne
        b_dbp = 75    # Intercetta per la stima della DBP nelle donne
    else:
        raise ValueError("Gender not recognized. Please specify 'male' or 'female'.")
    ptt_values = np.array(ptt_values)  
    estimated_dbp = a_dbp * ptt_values + b_dbp
    return estimated_dbp

def estimate_bp_from_aix(aix_values, output_dir):
    """
    Stima la pressione sanguigna (SBP e DBP) dall'Augmentation Index (AIx) e salva i risultati in output_dir.
    
    :param aix_values: Array di valori AIx.
    :param output_dir: Directory di output per salvare i risultati.
    :return: Tuple con SBP e DBP stimati.
    """
    # Coefficienti basati su studi medici (da calibrare con dati reali)
    a_sbp, b_sbp = 0.5, 120
    a_dbp, b_dbp = 0.3, 80

    estimated_sbp = a_sbp * aix_values + b_sbp
    estimated_dbp = a_dbp * aix_values + b_dbp

    # Salva i valori stimati in file di testo
    np.savetxt(os.path.join(output_dir, 'estimated_sbp_aix.txt'), estimated_sbp, fmt='%.2f')
    np.savetxt(os.path.join(output_dir, 'estimated_dbp_aix.txt'), estimated_dbp, fmt='%.2f')

    # Genera e salva i grafici dei valori stimati
    plt.figure(figsize=(10, 4))
    plt.plot(estimated_sbp, label='Estimated SBP (AIx)', color='blue')
    plt.xlabel('Measurement Index')
    plt.ylabel('SBP (mmHg)')
    plt.title('Estimated Systolic Blood Pressure (AIx)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'estimated_sbp_aix.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(estimated_dbp, label='Estimated DBP (AIx)', color='red')
    plt.xlabel('Measurement Index')
    plt.ylabel('DBP (mmHg)')
    plt.title('Estimated Diastolic Blood Pressure (AIx)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'estimated_dbp_aix.png'))
    plt.close()

    print(f"[INFO] Estimated SBP and DBP (AIx) saved in {output_dir}")

    return estimated_sbp, estimated_dbp

def analyze_bp_methods(estimated_sbp_ppg, estimated_dbp_ppg, 
                        estimated_sbp_ppw, estimated_dbp_ppw, 
                        estimated_sbp_aix, estimated_dbp_aix, output_dir):
    """
    Confronta le diverse stime della pressione sanguigna ottenute con PPG, PPW e AIx.
    """

    print("[INFO] Performing Statistical Analysis on BP Estimates...")

    # Trova la lunghezza minima tra tutti i dati per uniformare
    min_length = min(len(estimated_sbp_ppg), len(estimated_dbp_ppg), 
                     len(estimated_sbp_ppw), len(estimated_dbp_ppw), 
                     len(estimated_sbp_aix), len(estimated_dbp_aix))

    print(f"[DEBUG] Minimum length of data arrays: {min_length}")

    # Tronca tutte le liste alla lunghezza minima
    estimated_sbp_ppg = estimated_sbp_ppg[:min_length]
    estimated_dbp_ppg = estimated_dbp_ppg[:min_length]
    estimated_sbp_ppw = estimated_sbp_ppw[:min_length]
    estimated_dbp_ppw = estimated_dbp_ppw[:min_length]
    estimated_sbp_aix = estimated_sbp_aix[:min_length]
    estimated_dbp_aix = estimated_dbp_aix[:min_length]

    # Crea il DataFrame con dati uniformati
    data = {
        "SBP_PPG": estimated_sbp_ppg,
        "DBP_PPG": estimated_dbp_ppg,
        "SBP_PPW": estimated_sbp_ppw,
        "DBP_PPW": estimated_dbp_ppw,
        "SBP_AIx": estimated_sbp_aix,
        "DBP_AIx": estimated_dbp_aix,
    }

    df = pd.DataFrame(data)

    print("[INFO] DataFrame created successfully!")

    # Salva i dati in un file CSV
    df.to_csv(os.path.join(output_dir, "blood_pressure_estimates.csv"), index=False)
    print("[INFO] Blood pressure estimates saved to CSV.")

    # 📊 Visualizza le distribuzioni con un pairplot
    plt.figure(figsize=(12, 8))
    sns.pairplot(df)
    plt.suptitle("Pairplot of Blood Pressure Estimates", y=1.02)
    plt.savefig(os.path.join(output_dir, "blood_pressure_comparison_pairplot.png"))
    plt.close()
    print("[INFO] Pairplot saved.")

    # 📉 Boxplot per confrontare le distribuzioni
    plt.figure(figsize=(12, 6))
    df.boxplot()
    plt.xticks(rotation=15)
    plt.title("Comparison of Blood Pressure Estimates")
    plt.savefig(os.path.join(output_dir, "blood_pressure_comparison_boxplot.png"))
    plt.close()
    print("[INFO] Boxplot saved.")

    # 📈 Esegui i test Bland-Altman per ogni confronto
    bland_altman_plot(estimated_sbp_ppg, estimated_sbp_ppw, "SBP PPG vs PPW", f"{output_dir}/bland_altman_sbp_ppg_ppw.png")
    bland_altman_plot(estimated_sbp_ppg, estimated_sbp_aix, "SBP PPG vs AIx", f"{output_dir}/bland_altman_sbp_ppg_aix.png")
    bland_altman_plot(estimated_dbp_ppg, estimated_dbp_ppw, "DBP PPG vs PPW", f"{output_dir}/bland_altman_dbp_ppg_ppw.png")
    bland_altman_plot(estimated_dbp_ppg, estimated_dbp_aix, "DBP PPG vs AIx", f"{output_dir}/bland_altman_dbp_ppg_aix.png")

    print("[INFO] Statistical analysis completed successfully!")

def analyze_blood_pressure_estimations(sbp_ppg, dbp_ppg, sbp_ppw, dbp_ppw, sbp_aix, dbp_aix, output_dir):
    """
    Analizza statisticamente le stime della pressione sanguigna ottenute con PTT-PPG, PTT-PPW e AIx.
    """
    methods = ["PTT-PPG", "PTT-PPW", "AIx"]
    
    # Creiamo un DataFrame con tutte le stime
    df = pd.DataFrame({
        "SBP_PPG": sbp_ppg, "DBP_PPG": dbp_ppg,
        "SBP_PPW": sbp_ppw, "DBP_PPW": dbp_ppw,
        "SBP_AIx": sbp_aix, "DBP_AIx": dbp_aix
    })
    
    # Matrice di correlazione
    correlation_matrix = df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of BP Estimations")
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # Scatter Plot per SBP e DBP
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(sbp_ppg, sbp_ppw, label="PTT-PPG vs PTT-PPW", alpha=0.7)
    plt.scatter(sbp_ppg, sbp_aix, label="PTT-PPG vs AIx", alpha=0.7)
    plt.xlabel("SBP PPG")
    plt.ylabel("SBP Estimations")
    plt.legend()
    plt.title("SBP Estimations Comparison")

    plt.subplot(1, 2, 2)
    plt.scatter(dbp_ppg, dbp_ppw, label="PTT-PPG vs PTT-PPW", alpha=0.7)
    plt.scatter(dbp_ppg, dbp_aix, label="PTT-PPG vs AIx", alpha=0.7)
    plt.xlabel("DBP PPG")
    plt.ylabel("DBP Estimations")
    plt.legend()
    plt.title("DBP Estimations Comparison")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_bp_estimations.png")
    plt.close()
    
    # Bland-Altman Plot

def bland_altman_plot(data1, data2, label, output_path):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff)

    # Plotting del grafico Bland-Altman
    plt.figure(figsize=(6, 5))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md, color='red', linestyle='--', label="Mean Diff")
    plt.axhline(md + 1.96 * sd, color='blue', linestyle='--', label="+1.96 SD")
    plt.axhline(md - 1.96 * sd, color='blue', linestyle='--', label="-1.96 SD")
    plt.xlabel("Mean BP Estimation")
    plt.ylabel("Difference")
    plt.title(f"Bland-Altman Plot ({label})")
    plt.legend()

    # Salva il grafico come immagine
    plt.savefig(output_path)
    plt.close()

    # Salva i risultati in un file di testo
    results_file = output_path.replace('.png', '_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Comparison: {label}\n")
        f.write(f"Mean Difference (MD): {md:.3f}\n")
        f.write(f"Standard Deviation of Difference (SD): {sd:.3f}\n")
        f.write(f"Upper limit (MD + 1.96 * SD): {md + 1.96 * sd:.3f}\n")
        f.write(f"Lower limit (MD - 1.96 * SD): {md - 1.96 * sd:.3f}\n")
    print(f"[INFO] Bland-Altman results saved in {results_file}")

    import pandas as pd

def bland_altman_results(estimated_sbp, real_sbp, estimated_dbp, real_dbp, output_dir):
    """
    Calcola i risultati di Bland-Altman per la stima della pressione sanguigna
    (sia SBP che DBP) e salva i risultati in un file CSV.

    :param estimated_sbp: Valori stimati della pressione sanguigna sistolica (SBP).
    :param real_sbp: Valori reali della pressione sanguigna sistolica (SBP).
    :param estimated_dbp: Valori stimati della pressione sanguigna diastolica (DBP).
    :param real_dbp: Valori reali della pressione sanguigna diastolica (DBP).
    :param output_dir: Cartella di output dove salvare i risultati.
    """

    # Assicuriamoci che le dimensioni siano allineate
    min_length = min(len(estimated_sbp), len(real_sbp), len(estimated_dbp), len(real_dbp))
    estimated_sbp = estimated_sbp[:min_length]
    real_sbp = real_sbp[:min_length]
    estimated_dbp = estimated_dbp[:min_length]
    real_dbp = real_dbp[:min_length]

    # Calcolo della differenza tra stime e valori reali
    sbp_diff = estimated_sbp - real_sbp
    dbp_diff = estimated_dbp - real_dbp

    # Calcolo della media della differenza e della deviazione standard
    sbp_md = np.mean(sbp_diff)
    sbp_sd = np.std(sbp_diff)

    dbp_md = np.mean(dbp_diff)
    dbp_sd = np.std(dbp_diff)

    # Calcolo dei limiti superiori e inferiori per SBP e DBP
    sbp_upper_limit = sbp_md + 1.96 * sbp_sd
    sbp_lower_limit = sbp_md - 1.96 * sbp_sd

    dbp_upper_limit = dbp_md + 1.96 * dbp_sd
    dbp_lower_limit = dbp_md - 1.96 * dbp_sd

    # Salvataggio dei risultati in un file CSV
    results = {
        "Parameter": ["SBP", "DBP"],
        "Mean Difference (MD)": [sbp_md, dbp_md],
        "Standard Deviation (SD)": [sbp_sd, dbp_sd],
        "Upper Limit (MD + 1.96*SD)": [sbp_upper_limit, dbp_upper_limit],
        "Lower Limit (MD - 1.96*SD)": [sbp_lower_limit, dbp_lower_limit]
    }

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "bland_altman_results.csv"), index=False)

    print(f"[INFO] Bland-Altman results saved to {output_dir}/bland_altman_results.csv")
# Modifica della funzione `evaluate_estimations` per includere i risultati nel CSV
def evaluate_estimations(estimated_sbp_ppg, estimated_dbp_ppg, estimated_sbp_ppw, estimated_dbp_ppw, estimated_sbp_aix, estimated_dbp_aix, real_sbp, real_dbp, output_dir):
    """
    Valuta l'accuratezza delle stime della pressione sanguigna rispetto ai valori reali.
    Calcola MAE, RMSE e Pearson correlation e genera il Bland-Altman plot.
    """
    if len(estimated_sbp_ppg) == 0 or len(real_sbp) == 0:
        print("[ERROR] Dati mancanti per la stima SBP!")
        return

    # Assicurarsi che entrambe le liste abbiano la stessa lunghezza
    min_length = min(len(estimated_sbp_ppg), len(real_sbp))
    estimated_sbp_ppg = np.array(estimated_sbp_ppg[:min_length])
    real_sbp = np.array(real_sbp[:min_length])

    # Calcola MAE e RMSE
    mae = np.mean(np.abs(estimated_sbp_ppg - real_sbp))
    rmse = np.sqrt(np.mean((estimated_sbp_ppg - real_sbp) ** 2))

    # Correlazione di Pearson
    correlation, p_value = pearsonr(estimated_sbp_ppg, real_sbp)
    
    # Calcola Bland-Altman e salva i risultati
    bland_altman_results_df = bland_altman_results(estimated_sbp_ppg, real_sbp, estimated_dbp_ppg, real_dbp, output_dir)

    # Stampa i risultati
    print(f"\n[RESULTS] SBP PPG vs Real SBP")
    print(f"MAE: {mae:.2f} mmHg")
    print(f"RMSE: {rmse:.2f} mmHg")
    print(f"Pearson Correlation: {correlation:.2f} (p-value: {p_value:.3f})")

    # Analizza altre tecniche (PPW, AIx)
    # Esegui analisi similari per SBP-PPW, SBP-AIx, DBP-PPW, DBP-AIx

    return bland_altman_results_df


import os
import numpy as np
from scipy.signal import resample_poly
from scipy.interpolate import CubicSpline
from scipy.signal import resample_poly
from fractions import Fraction
import numpy as np
import os
import matplotlib.pyplot as plt

def convolution_interpolation(signal, old_rate, new_rate, output_dir=None, label=""):
    """
    Applica l'interpolazione al segnale usando la tecnica di resampling polinomiale.
    
    :param signal: array 1D del segnale da interpolare.
    :param old_rate: frequenza di campionamento originale.
    :param new_rate: frequenza di campionamento target.
    :param output_dir: (opzionale) directory in cui salvare il segnale interpolato.
    :param label: (opzionale) etichetta per identificare il file di output.
    :return: segnale interpolato.
    """
    # Calcola il rapporto di interpolazione come float
    ratio = new_rate / old_rate
    # Converte il rapporto in una frazione
    frac = Fraction(ratio).limit_denominator()
    up = frac.numerator
    down = frac.denominator

    print(f"[DEBUG] Interpolation factors for {label}: up = {up}, down = {down}")
    
    # Applica l'interpolazione usando resample_poly
    from scipy.signal import resample_poly
    interpolated_signal = resample_poly(signal, up, down)
    
    # Se viene fornita output_dir, salva il segnale interpolato su file
    if output_dir is not None:
        output_file = os.path.join(output_dir, f"interpolated_signal_{label}.txt")
        np.savetxt(output_file, interpolated_signal, fmt="%.5f")
        plt.figure(figsize=(10, 4))
        plt.plot(interpolated_signal, label=f"Interpolated {label}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.title(f"Interpolated Signal ({label})")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"interpolated_signal_{label}.jpg"))
        plt.close()
        print(f"[INFO] Interpolated signal for {label} saved in {output_dir}")
    
    return interpolated_signal

def cubic_spline_interpolation(signal, original_fs, target_fs, output_dir=None, label="interpolated"):
    """
    Interpola il segnale usando l'interpolazione cubica spline.
    
    :param signal: Array del segnale da interpolare.
    :param original_fs: Frequenza di campionamento originale (in Hz).
    :param target_fs: Frequenza di campionamento target (in Hz).
    :param output_dir: Directory in cui salvare il segnale interpolato (opzionale).
    :param label: Etichetta per identificare il file.
    :return: Segnale interpolato.
    """
    # Crea un vettore di tempi originali
    original_times = np.arange(len(signal)) / original_fs
    # Calcola il numero di campioni target
    target_length = int(len(signal) * target_fs / original_fs)
    # Genera il vettore di tempi target
    target_times = np.linspace(original_times[0], original_times[-1], target_length)
    
    # Crea l'interpolatore spline
    cs = CubicSpline(original_times, signal)
    interpolated_signal = cs(target_times)
    
    # Salva il segnale interpolato se viene fornita una directory
    if output_dir:
        output_file = os.path.join(output_dir, f"{label}_spline_interpolated.txt")
        np.savetxt(output_file, interpolated_signal, fmt="%.5f")
        print(f"[INFO] Segnale interpolato con spline salvato in: {output_file}")
    
    return interpolated_signal

# --- DATA VISUALIZATION AND SAVING ---
def plot_ppg_signal(ppg_signal, label, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(ppg_signal, label=label)
    plt.xlabel('Frame')
    plt.ylabel('Signal Amplitude')
    plt.title(f'{label} Signal')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{label}_signal.png'))
    plt.close()

def plot_peaks(ppg_signal, peaks, label, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(ppg_signal, label=label)
    plt.plot(peaks, np.array(ppg_signal)[peaks], "x", label='Peaks')
    plt.xlabel('Frame')
    plt.ylabel('Signal Amplitude')
    plt.title(f'{label} with Peaks')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{label}_peaks.png'))
    plt.close()

def plot_ptt_values(ptt_values, output_dir, label="PTT Values"):
    plt.figure(figsize=(10, 4))
    plt.plot(ptt_values, label=label)
    plt.xlabel('Measurement Index')
    plt.ylabel('PTT (seconds)')
    plt.title(f'{label} Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'ptt_{label.replace("-", "_")}.png'))
    plt.close()

def plot_estimated_bp(bp_values, label, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(bp_values, label=label)
    plt.xlabel('Measurement Index')
    plt.ylabel('Blood Pressure (mmHg)')
    plt.title(f'Estimated {label} Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'estimated_{label.lower()}.png'))
    plt.close()

def plot_estimated_blood_pressure_scatter(estimated_sbp, estimated_dbp, output_dir):
    """
    Crea un grafico scatter con i valori stimati della pressione sanguigna sistolica (SBP) e diastolica (DBP).
    
    :param estimated_sbp: Lista dei valori stimati della pressione sanguigna sistolica.
    :param estimated_dbp: Lista dei valori stimati della pressione sanguigna diastolica.
    :param output_dir: Directory in cui salvare il grafico.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(estimated_sbp)), estimated_sbp, label='Systolic Blood Pressure (SBP)', color='cyan', marker='^', edgecolors='b')
    plt.scatter(range(len(estimated_dbp)), estimated_dbp, label='Diastolic Blood Pressure (DBP)', color='salmon', marker='v', edgecolors='r')
    
    plt.xlabel('Measurement Index')
    plt.ylabel('Blood Pressure (mmHg)')
    plt.title('Estimated Blood Pressure Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Salva il grafico nella directory specificata
    output_path = os.path.join(output_dir, 'estimated_blood_pressure_scatter.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Estimated blood pressure scatter graph saved at: {output_path}")

import matplotlib.pyplot as plt

def plot_peak_synchronization(forehead_peaks, neck_peaks, signal_forehead, signal_neck, fps, output_dir):
    """
    Genera un'immagine per visualizzare se i picchi dei segnali della fronte e del collo sono sincronizzati.

    :param forehead_peaks: Indici dei picchi nel segnale della fronte.
    :param neck_peaks: Indici dei picchi nel segnale del collo.
    :param signal_forehead: Segnale della fronte.
    :param signal_neck: Segnale del collo.
    :param fps: Frame rate del video.
    :param output_dir: Directory di output per salvare l'immagine.
    """
    # Converti i picchi in tempi
    times_forehead = np.array(forehead_peaks) / fps
    times_neck = np.array(neck_peaks) / fps

    # Crea una figura
    plt.figure(figsize=(12, 6))

    # Plot del segnale della fronte con i picchi
    plt.subplot(2, 1, 1)
    plt.plot(signal_forehead, label='Forehead Signal', color='blue')
    plt.scatter(forehead_peaks, signal_forehead[forehead_peaks], color='red', marker='o', label='Forehead Peaks')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.title('Forehead Signal with Peaks')
    plt.legend()

    # Plot del segnale del collo con i picchi
    plt.subplot(2, 1, 2)
    plt.plot(signal_neck, label='Neck Signal', color='green')
    plt.scatter(neck_peaks, signal_neck[neck_peaks], color='orange', marker='o', label='Neck Peaks')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.title('Neck Signal with Peaks')
    plt.legend()

    # Salva l'immagine
    output_path = os.path.join(output_dir, 'peak_synchronization.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"[INFO] Peak synchronization plot saved in: {output_path}")

def save_estimated_bp(bp_values, label, output_dir):
    np.savetxt(os.path.join(output_dir, f'estimated_{label.lower()}.txt'), bp_values, fmt='%.2f')

# --- MAIN --- 
def initialize_paths_and_config():
    print("\n[INFO] Initializing paths and configurations...")
    dataset_folder = "/Volumes/DanoUSB"
    subject = "F005"
    task = "T3"
    gender = "female"
    video_path = f"{dataset_folder}/{subject}/{task}/vid.avi"
    output_dir = f"NIAC/{subject}/{task}"
    neck_roi_file_path = f"{output_dir}/neck_roi_dimensions.json"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    return video_path, output_dir, neck_roi_file_path, gender

def calculate_roi(video_path, neck_roi_file_path, output_roi_path):

    """
    Seleziona o carica la ROI iniziale e poi applica il tracking adattivo per aggiornarla frame dopo frame.
    
    :param video_path: Percorso del video da analizzare.
    :param neck_roi_file_path: Percorso del file JSON in cui salvare/caricare la ROI iniziale.
    :param output_roi_path: Percorso del file JSON in cui salvare/caricare le ROI aggiornate dal tracking.
    :return: Ultima ROI tracciata (x, y, width, height)
    """
    print("\n[STEP 1] Calculating ROI with tracking...")

    # Se esiste già un file con tutto il tracking adattivo, lo carica
    if os.path.exists(output_roi_path):
        with open(output_roi_path, 'r') as file:
            roi_updates = json.load(file)
        print(f"[INFO] Loaded adaptive tracking data from {output_roi_path}")
        last_roi = roi_updates[-1]
        return (last_roi['x'], last_roi['y'], last_roi['width'], last_roi['height'])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read frame from video.")
        return None

    # Se esiste una ROI salvata, la carica, altrimenti chiede la selezione manuale
    if os.path.exists(neck_roi_file_path):
        neck_roi = load_roi_dimensions(neck_roi_file_path)
        print(f"[INFO] Neck ROI loaded from file: {neck_roi}")
    else:
        print("[INFO] Select neck ROI manually.")
        neck_roi = select_neck_roi(frame)
        save_roi_dimensions(neck_roi, neck_roi_file_path)
        print(f"[INFO] Neck ROI saved to file: {neck_roi}")

    # Controlla se la ROI è valida
    if not neck_roi or len(neck_roi) != 4 or neck_roi[2] <= 0 or neck_roi[3] <= 0:
        print("[ERROR] Invalid ROI. Check ROI selection.")
        return None

    # Avvia il tracking adattivo della ROI
    roi_updates = track_neck_roi(video_path, neck_roi_file_path, output_roi_path)

    if not roi_updates:
        print("[ERROR] Tracking failed, no ROIs saved!")
        return None

    # Restituisce l'ultima ROI tracciata
    last_roi = roi_updates[-1]
    return (last_roi['x'], last_roi['y'], last_roi['width'], last_roi['height'])

def process_ppg_forehead(video_path, output_dir):
    print("\n[STEP 2] Processing forehead PPG...")
    rgb_signals = extract_rgb_trace_MediaPipe(video_path, output_dir)
    # Convertiamo i segnali in array numpy separati per ciascun canale
    R = np.array(rgb_signals['R'])
    G = np.array(rgb_signals['G'])
    B = np.array(rgb_signals['B'])
    
    # Verifica che ciascun tracciato abbia almeno 28 campioni
    min_samples = 28
    if R.shape[0] < min_samples:
        needed = min_samples - R.shape[0]
        R = np.concatenate((R, np.repeat(R[-1:], needed, axis=0)), axis=0)
        G = np.concatenate((G, np.repeat(G[-1:], needed, axis=0)), axis=0)
        B = np.concatenate((B, np.repeat(B[-1:], needed, axis=0)), axis=0)
        print(f"[WARNING] Forehead signal had less than 28 samples. Extended to {R.shape[0]} samples.")
    
    print("[INFO] Forehead RGB signals extracted successfully.")
    return R, G, B

def process_ppg_neck(video_path, neck_roi, output_dir):
    print("\n[STEP 3] Processing neck PPG...")

    if neck_roi is None or len(neck_roi) != 4:
        print("[ERROR] Invalid Neck ROI. Check ROI selection.")
        return None
    
    x, y, w, h = neck_roi
    if w <= 0 or h <= 0:
        print(f"[ERROR] Invalid ROI dimensions: {neck_roi}")
        return None

    print(f"[INFO] Using Neck ROI: x={x}, y={y}, w={w}, h={h}")

    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    if not ret:
        print("[ERROR] Could not read the first frame of the video.")
        return None
    cap.release()


    rgb_signals = extract_rgb_trace(video_path, neck_roi, output_dir, "Neck")

    print("[DEBUG] Extracted RGB signals for Neck")

    if len(rgb_signals['R']) == 0 or len(rgb_signals['G']) == 0:
        print("[ERROR] No valid RGB data extracted from Neck.")
        return None

    red_signal = np.array(rgb_signals['R'])
    green_signal = np.array(rgb_signals['G'])

    print(f"[DEBUG] Red signal shape: {red_signal.shape}, Green signal shape: {green_signal.shape}")

    if red_signal.shape[0] < 5 or green_signal.shape[0] < 5:
        print(f"[ERROR] Signals are too short! Red: {red_signal.shape[0]}, Green: {green_signal.shape[0]}")
        return None

    mean_rg_signal = np.vstack((red_signal, green_signal)).T  # Combine R and G into a 2D array

    print(f"[DEBUG] Mean R+G signal shape: {mean_rg_signal.shape}")

    if mean_rg_signal.shape[0] < 28:
        needed = 28 - mean_rg_signal.shape[0]
        last_samples = mean_rg_signal[-1:].repeat(needed, axis=0)
        mean_rg_signal = np.vstack((mean_rg_signal, last_samples))
        print(f"[WARNING] Neck signal had less than 28 samples. Extended to {mean_rg_signal.shape[0]} samples.")

    return mean_rg_signal

def calculate_ppw_neck(video_path, neck_roi, output_dir):
    print("\n[STEP 4] Calculating neck PPW...")
    ppw_signal = extract_pixflow_signal_improved(video_path, neck_roi, output_dir, "Neck")
    print("[INFO] Neck PPW signal extracted successfully.")
    return ppw_signal

def preprocess_and_filter_signals(signals, sampling_rate, output_dir, label):

    """
    Preprocessa e filtra i segnali, salvando i risultati in file .txt e .jpg.
    Rimuove eventuali artefatti finali che superano una soglia di ampiezza.

    :param signals: array monodimensionale con i segnali da preprocessare e filtrare.
    :param sampling_rate: Frequenza di campionamento.
    :param output_dir: Directory di output per salvare i risultati.
    :param label: Etichetta per identificare i file di output.
    :return: Segnali filtrati (eventualmente troncati se rilevato un artefatto).
    """
    # 1. Rimuove le tendenze dal segnale
    detrended_signals = smoothness_priors_detrend(signals, lambda_param=10)
    
    # 2. Applica il filtro passa-banda
    filtered_signals = butter_bandpass_filter(
        detrended_signals,
        lowcut=0.7,
        highcut=3.0,
        fs=sampling_rate
    )

    # 3. Rileva e rimuove un eventuale artefatto finale
    threshold_factor = 4.0  # Puoi regolare questo valore in base alle tue esigenze
    mean_val = np.mean(filtered_signals)
    std_val = np.std(filtered_signals)
    upper_threshold = mean_val + threshold_factor * std_val
    lower_threshold = mean_val - threshold_factor * std_val

    # Cerchiamo gli indici in cui il segnale supera la soglia (in alto o in basso)
    artifact_indices = np.where((filtered_signals > upper_threshold) | (filtered_signals < lower_threshold))[0]
    if len(artifact_indices) > 0:
        # Se troviamo un artefatto, tagliamo il segnale dal primo campione anomalo alla fine
        first_artifact_index = artifact_indices[0]
        filtered_signals = filtered_signals[:first_artifact_index]
        print(f"[INFO] Rilevato artefatto oltre la soglia. Segnale troncato a partire dal campione {first_artifact_index}.")

    # 4. Salva i segnali filtrati su file .txt
    output_file_txt = f"{output_dir}/filtered_signals_{label}.txt"
    np.savetxt(output_file_txt, filtered_signals, delimiter=",", header="Filtered Signals", comments="")
    print(f"Segnale filtrato salvato in: {output_file_txt}")
    
    # 5. Genera il grafico dei segnali filtrati e salva su file .jpg
    output_file_jpg = f"{output_dir}/filtered_signals_{label}.jpg"
    plt.figure(figsize=(10, 4))
    plt.plot(filtered_signals, label="Filtered Signals")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title(f"Filtered Signals ({label})")
    plt.legend()
    plt.savefig(output_file_jpg)
    plt.close()
    print(f"Grafico del segnale filtrato salvato in: {output_file_jpg}")
    
    return filtered_signals

def save_and_visualize_results(bvp_forehead, bvp_neck, forehead_peaks, neck_peaks, ptt, estimated_sbp, estimated_dbp, output_dir, label="PTT"):
    """
    Salva e visualizza i risultati delle elaborazioni.
    """
    print(f"\n[INFO] Saving and visualizing results for {label}...")
    plot_peaks(bvp_forehead, forehead_peaks, f'Forehead PPG ({label})', output_dir)
    plot_peaks(bvp_neck, neck_peaks, f'Neck PPG ({label})', output_dir)
    plot_ptt_values(ptt, output_dir, label=label)
    plot_estimated_bp(estimated_sbp, f'SBP ({label})', output_dir)
    plot_estimated_bp(estimated_dbp, f'DBP ({label})', output_dir)
    save_estimated_bp(estimated_sbp, f'SBP_{label}', output_dir)
    save_estimated_bp(estimated_dbp, f'DBP_{label}', output_dir)
    print(f"[INFO] Results saved and visualized successfully for {label}.")

def calculate_coefficients(ptt_values, bp_values):
    """
    Calcola i coefficienti della regressione lineare che mappa i valori PTT (in secondi)
    ai valori di pressione sanguigna (in mmHg).

    :param ptt_values: Array dei valori PTT.
    :param bp_values: Array dei valori reali di pressione sanguigna.
    :return: Coefficienti a e b tali che BP = a * PTT + b.
    """
    # Assicurati che ptt_values e bp_values abbiano la stessa lunghezza
    min_length = min(len(ptt_values), len(bp_values))
    ptt_values = np.array(ptt_values[:min_length])
    bp_values = np.array(bp_values[:min_length])
    
    # Calcola la regressione lineare (minimi quadrati)
    a, b = np.polyfit(ptt_values, bp_values, 1)
    print(f"[INFO] Coefficienti calcolati: a = {a:.3f}, b = {b:.3f}")
    return a, b


from scipy.stats import pearsonr

def compare_ptt_bp_trends(ptt_dict, bp_systolic, bp_diastolic, output_dir):
    """
    Confronta, per ciascun metodo PTT, la stima rispetto ai dati di pressione arteriosa
    di riferimento (BP). Per ogni metodo calcola:
      - Correlazione di Pearson tra PTT e SBP
      - Correlazione di Pearson tra PTT e DBP
      - MAE e RMSE (calcolati tra le stime BP ottenute tramite regressione lineare sui valori PTT e i BP reali)
      - Plot Bland-Altman per SBP e DBP

    Si assume che per ciascun metodo PTT siano stati calcolati dei valori BP tramite la funzione di stima (ad es. estimate_systolic_blood_pressure).

    :param ptt_dict: Dizionario dei metodi PTT, es. {'Direct_Conv': ptt_direct_conv, 'Direct_Spline': ptt_direct_spline, ...}
    :param bp_systolic: Array dei valori SBP di riferimento.
    :param bp_diastolic: Array dei valori DBP di riferimento.
    :param output_dir: Directory di output.
    """
    import pandas as pd
    comparisons = []
    for method_name, ptt_values in ptt_dict.items():
        ptt_arr = np.array(ptt_values)
        min_len = min(len(ptt_arr), len(bp_systolic), len(bp_diastolic))
        ptt_arr = ptt_arr[:min_len]
        sbp_ref = np.array(bp_systolic)[:min_len]
        dbp_ref = np.array(bp_diastolic)[:min_len]
        
        # Calcola correlazioni
        corr_sbp, p_sbp = pearsonr(ptt_arr, sbp_ref)
        corr_dbp, p_dbp = pearsonr(ptt_arr, dbp_ref)
        
        # Per confronto via regressione, supponiamo di usare una semplice equazione lineare
        # (i coefficienti possono essere quelli definiti nelle funzioni di stima)
        # Qui, ad esempio, stimiamo BP = a * PTT + b, e poi confrontiamo con BP reale.
        # Definiamo coefficienti arbitrari per l'esempio.
        a_sbp, b_sbp = -100, 120
        a_dbp, b_dbp = -75, 80
        est_sbp = a_sbp * ptt_arr + b_sbp
        est_dbp = a_dbp * ptt_arr + b_dbp

        mae_sbp = np.mean(np.abs(est_sbp - sbp_ref))
        rmse_sbp = np.sqrt(np.mean((est_sbp - sbp_ref) ** 2))
        mae_dbp = np.mean(np.abs(est_dbp - dbp_ref))
        rmse_dbp = np.sqrt(np.mean((est_dbp - dbp_ref) ** 2))
        
        comparisons.append({
            "Method": method_name,
            "SBP Pearson r": corr_sbp,
            "SBP p-value": p_sbp,
            "SBP MAE": mae_sbp,
            "SBP RMSE": rmse_sbp,
            "DBP Pearson r": corr_dbp,
            "DBP p-value": p_dbp,
            "DBP MAE": mae_dbp,
            "DBP RMSE": rmse_dbp
        })
        
        # Bland-Altman Plot per SBP
        mean_sbp = (est_sbp + sbp_ref) / 2
        diff_sbp = est_sbp - sbp_ref
        md_sbp = np.mean(diff_sbp)
        sd_sbp = np.std(diff_sbp)
        plt.figure(figsize=(6,5))
        plt.scatter(mean_sbp, diff_sbp, alpha=0.5)
        plt.axhline(md_sbp, color='red', linestyle='--', label="Mean Diff")
        plt.axhline(md_sbp + 1.96 * sd_sbp, color='blue', linestyle='--', label="+1.96 SD")
        plt.axhline(md_sbp - 1.96 * sd_sbp, color='blue', linestyle='--', label="-1.96 SD")
        plt.xlabel("Mean SBP (mmHg)")
        plt.ylabel("Difference (mmHg)")
        plt.title(f"Bland-Altman SBP: {method_name}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"bland_altman_SBP_{method_name}.png"))
        plt.close()
        
        # Bland-Altman Plot per DBP
        mean_dbp = (est_dbp + dbp_ref) / 2
        diff_dbp = est_dbp - dbp_ref
        md_dbp = np.mean(diff_dbp)
        sd_dbp = np.std(diff_dbp)
        plt.figure(figsize=(6,5))
        plt.scatter(mean_dbp, diff_dbp, alpha=0.5)
        plt.axhline(md_dbp, color='red', linestyle='--', label="Mean Diff")
        plt.axhline(md_dbp + 1.96 * sd_dbp, color='blue', linestyle='--', label="+1.96 SD")
        plt.axhline(md_dbp - 1.96 * sd_dbp, color='blue', linestyle='--', label="-1.96 SD")
        plt.xlabel("Mean DBP (mmHg)")
        plt.ylabel("Difference (mmHg)")
        plt.title(f"Bland-Altman DBP: {method_name}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"bland_altman_DBP_{method_name}.png"))
        plt.close()
        
    comp_df = pd.DataFrame(comparisons)
    comp_csv = os.path.join(output_dir, "ptt_bp_comparisons.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"[INFO] PTT vs BP trend comparisons saved in {comp_csv}")

def compare_all_ptt_methods(ptt_dict, output_dir):
    import itertools
    results = []
    pairs = list(itertools.combinations(ptt_dict.keys(), 2))
    for method1, method2 in pairs:
        ptt1 = np.array(ptt_dict[method1])
        ptt2 = np.array(ptt_dict[method2])
        min_len = min(len(ptt1), len(ptt2))
        ptt1 = ptt1[:min_len]
        ptt2 = ptt2[:min_len]

        # Verifica che entrambe le liste abbiano almeno 2 elementi
        if len(ptt1) < 2 or len(ptt2) < 2:
            print(f"[ERROR] Non ci sono abbastanza valori PTT validi per {method1} e {method2}.")
            continue

        # Calcola Pearson, MAE, RMSE
        pearson_r, p_val = pearsonr(ptt1, ptt2)
        mae = np.mean(np.abs(ptt1 - ptt2))
        rmse = np.sqrt(np.mean((ptt1 - ptt2)**2))
        results.append({
            "Method 1": method1,
            "Method 2": method2,
            "Pearson r": pearson_r,
            "p-value": p_val,
            "MAE": mae,
            "RMSE": rmse
        })

        # Plot Bland-Altman
        mean_vals = (ptt1 + ptt2) / 2
        diff_vals = ptt1 - ptt2
        md = np.mean(diff_vals)
        sd = np.std(diff_vals)
        plt.figure(figsize=(6,5))
        plt.scatter(mean_vals, diff_vals, alpha=0.5)
        plt.axhline(md, color='red', linestyle='--', label="Mean Diff")
        plt.axhline(md + 1.96 * sd, color='blue', linestyle='--', label="+1.96 SD")
        plt.axhline(md - 1.96 * sd, color='blue', linestyle='--', label="-1.96 SD")
        plt.xlabel("Mean PTT (s)")
        plt.ylabel("Difference (s)")
        plt.title(f"Bland-Altman: {method1} vs {method2}")
        plt.legend()
        plot_file = os.path.join(output_dir, f"bland_altman_{method1}_vs_{method2}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"[INFO] Bland-Altman plot saved: {plot_file}")

    # Salva i risultati in un file CSV
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, "ptt_comparisons.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"[INFO] PTT comparisons results saved in: {csv_file}")

def compare_ptt_methods(ptt_cross_corr, ptt_direct, output_dir, label):
    """
    Confronta i valori PTT calcolati con il metodo della cross-correlazione e il metodo diretto.
    
    :param ptt_cross_corr: Valori PTT calcolati con la cross-correlazione.
    :param ptt_direct: Valori PTT calcolati con il metodo diretto.
    :param output_dir: Directory di output per salvare i risultati.
    :param label: Etichetta per identificare i file di output.
    """
    if len(ptt_cross_corr) < 2 or len(ptt_direct) < 2:
        print(f"[ERROR] Non ci sono abbastanza valori PTT validi per {label}.")
        return

    # Assicurati che entrambe le liste abbiano la stessa lunghezza
    min_length = min(len(ptt_cross_corr), len(ptt_direct))
    ptt_cross_corr = np.array(ptt_cross_corr[:min_length])
    ptt_direct = np.array(ptt_direct[:min_length])

    # Calcola la correlazione di Pearson
    correlation, p_value = pearsonr(ptt_cross_corr, ptt_direct)
    print(f"[INFO] Correlation between Cross-Correlation and Direct PTT ({label}): {correlation:.3f} (p-value: {p_value:.3f})")

    # Genera un grafico scatter per confrontare i due metodi
    plt.figure(figsize=(10, 6))
    plt.scatter(ptt_cross_corr, ptt_direct, alpha=0.7, color='blue', label=f"{label}")
    plt.xlabel("PTT Cross-Correlation (s)")
    plt.ylabel("PTT Direct (s)")
    plt.title(f"Comparison of PTT Methods ({label})")
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(output_dir, f"compare_ptt_methods_{label.replace(' ', '_')}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Comparison plot saved in {output_path}")

def compare_bp_estimations(estimated_sbp_ppg, estimated_dbp_ppg, estimated_sbp_ppw, estimated_dbp_ppw, estimated_sbp_aix, estimated_dbp_aix, output_dir):
    """
    Confronta le diverse stime della pressione sanguigna (SBP e DBP) calcolando
    il coefficiente di correlazione di Pearson tra PTT-PPG, PTT-PPW e AIx.

    Salva i risultati e genera un grafico della matrice di correlazione.
    """
    print("\n[INFO] Comparing Blood Pressure Estimations with Pearson's Correlation...")

    # Trova la lunghezza minima tra tutti i dati per uniformare
    min_length = min(len(estimated_sbp_ppg), len(estimated_dbp_ppg),len(estimated_sbp_ppw), len(estimated_dbp_ppw),  len(estimated_sbp_aix), len(estimated_dbp_aix))

    # Tronca le liste alla lunghezza minima per allineare i dati
    estimated_sbp_ppg = estimated_sbp_ppg[:min_length]
    estimated_dbp_ppg = estimated_dbp_ppg[:min_length]
    estimated_sbp_ppw = estimated_sbp_ppw[:min_length]
    estimated_dbp_ppw = estimated_dbp_ppw[:min_length]
    estimated_sbp_aix = estimated_sbp_aix[:min_length]
    estimated_dbp_aix = estimated_dbp_aix[:min_length]

    # Creazione DataFrame per analisi
    data = {
        "SBP_PPG": estimated_sbp_ppg,
        "DBP_PPG": estimated_dbp_ppg,
        "SBP_PPW": estimated_sbp_ppw,
        "DBP_PPW": estimated_dbp_ppw,
        "SBP_AIx": estimated_sbp_aix,
        "DBP_AIx": estimated_dbp_aix,
    }

    df = pd.DataFrame(data)

    # Calcola le correlazioni di Pearson
    correlation_results = {}
    comparisons = [
        ("SBP_PPG", "SBP_PPW"),
        ("SBP_PPG", "SBP_AIx"),
        ("SBP_PPW", "SBP_AIx"),
        ("DBP_PPG", "DBP_PPW"),
        ("DBP_PPG", "DBP_AIx"),
        ("DBP_PPW", "DBP_AIx"),
    ]

    for col1, col2 in comparisons:
        r_value, p_value = pearsonr(df[col1], df[col2])
        correlation_results[f"{col1} vs {col2}"] = {"Pearson r": r_value, "p-value": p_value}

    # Converti i risultati in DataFrame e salva
    correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index')
    correlation_df.to_csv(os.path.join(output_dir, "pearson_correlation_results.csv"))
    print("[INFO] Pearson correlation results saved to CSV.")

    # Matrice di correlazione
    correlation_matrix = df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of BP Estimations")
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()
    print("[INFO] Correlation matrix plot saved.")

    print("[INFO] Blood pressure estimation comparison completed.")

def correlate_bp_ptt(ptt_values, bp_values, label):
    """
    Calcola la correlazione di Pearson tra PTT e pressione sanguigna.
    """
    if len(ptt_values) == 0 or len(bp_values) == 0:
        print(f"[ERROR] Nessun valore PTT o BP valido per {label}!")
        return None

    min_length = min(len(ptt_values), len(bp_values))
    ptt_values = ptt_values[:min_length]
    bp_values = bp_values[:min_length]

    # Verifica se gli array sono costanti
    if np.all(ptt_values == ptt_values[0]) or np.all(bp_values == bp_values[0]):
        print(f"[ERROR] Input costante per {label}. La correlazione non è definita.")
        return None

    correlation, p_value = pearsonr(ptt_values, bp_values)
    print(f"[INFO] Correlazione tra PTT e {label}: {correlation:.3f} (p-value: {p_value:.3f})")
    return correlation

def evaluate_estimations(estimated_bp, real_bp, method_name, output_dir):
    """
    Valuta l'accuratezza delle stime della pressione sanguigna rispetto ai valori reali.
    Calcola MAE, RMSE e Pearson correlation e genera il Bland-Altman plot.
    """
    if len(estimated_bp) == 0 or len(real_bp) == 0:
        print(f"[ERROR] Dati mancanti per {method_name}. Salto il calcolo.")
        return

    # Assicurarsi che entrambe le liste abbiano la stessa lunghezza
    min_length = min(len(estimated_bp), len(real_bp))
    estimated_bp = np.array(estimated_bp[:min_length])
    real_bp = np.array(real_bp[:min_length])

    # 1️⃣ MAE - Mean Absolute Error
    mae = np.mean(np.abs(estimated_bp - real_bp))

    # 2️⃣ RMSE - Root Mean Squared Error
    rmse = np.sqrt(np.mean((estimated_bp - real_bp) ** 2))

    # 3️⃣ Pearson Correlation
    correlation, p_value = pearsonr(estimated_bp, real_bp)

    # 4️⃣ Bland-Altman Plot
    mean_bp = (estimated_bp + real_bp) / 2
    diff_bp = estimated_bp - real_bp  # Differenza tra stima e valore reale
    mean_diff = np.mean(diff_bp)
    std_diff = np.std(diff_bp)

    plt.figure(figsize=(8, 5))
    plt.scatter(mean_bp, diff_bp, color='blue', alpha=0.5, label=method_name)
    plt.axhline(mean_diff, color='red', linestyle='--', label="Mean Diff")
    plt.axhline(mean_diff + 1.96 * std_diff, color='black', linestyle='dotted', label="+1.96 SD")
    plt.axhline(mean_diff - 1.96 * std_diff, color='black', linestyle='dotted', label="-1.96 SD")
    plt.xlabel("Mean BP (mmHg)")
    plt.ylabel("Difference (Estimated - Real) (mmHg)")
    plt.title(f"Bland-Altman Plot - {method_name}")
    plt.legend()
    plt.grid()
    
    # Salva il grafico
    output_path = f"{output_dir}/bland_altman_{method_name}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Bland-Altman Plot salvato in {output_path}")

    # Stampa i risultati
    print(f"\n[RESULTS] {method_name}")
    print(f"MAE: {mae:.2f} mmHg")
    print(f"RMSE: {rmse:.2f} mmHg")
    print(f"Pearson Correlation: {correlation:.2f} (p-value: {p_value:.3f})")

    return mae, rmse, correlation, p_value

import matplotlib.pyplot as plt

def plot_ptt_vs_bp(ptt_values, bp_values, label, output_dir):
    """
    Crea un grafico scatter per visualizzare la correlazione tra PTT e pressione sanguigna.
    """
    min_length = min(len(ptt_values), len(bp_values))
    ptt_values = ptt_values[:min_length]
    bp_values = bp_values[:min_length]

    plt.figure(figsize=(8, 5))
    plt.scatter(ptt_values, bp_values, alpha=0.7, color='blue', label=f"{label}")
    plt.xlabel("Pulse Transit Time (PTT) [s]")
    plt.ylabel("Blood Pressure (mmHg)")
    plt.title(f"Correlazione tra PTT e {label}")
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(output_dir, f"ptt_vs_{label.replace(' ', '_')}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Grafico salvato in {output_path}")

def main():
    # Step 0: Initialize paths and configurations
    video_path, output_dir, neck_roi_file_path, gender = initialize_paths_and_config()
    
    # Calcola l'FPS nominale e effettivo
    nominal_fps = get_sampling_rate(video_path)
    actual_fps = get_actual_fps(video_path)
    sampling_rate = actual_fps if actual_fps is not None else nominal_fps
    print(f"[INFO] Using FPS: {sampling_rate} for analysis\n")

    # Percorsi ai file di pressione sanguigna
    bp_systolic_file = "/Volumes/DanoUSB/Physiology/F005/T3/LA Systolic BP_mmHg.txt"
    bp_diastolic_file = "/Volumes/DanoUSB/Physiology/F005/T3/BP Dia_mmHg.txt"

    # Lettura della pressione sanguigna
    bp_systolic = read_blood_pressure_data(bp_systolic_file)
    bp_diastolic = read_blood_pressure_data(bp_diastolic_file)

    print(f"[INFO] Systolic BP: {bp_systolic[:10]}")
    print(f"[INFO] Diastolic BP: {bp_diastolic[:10]}")

    # Step 1: Calculate ROI using tracking
    output_roi_path = os.path.join(output_dir, "tracked_neck_roi.json")
    neck_roi = calculate_roi(video_path, neck_roi_file_path, output_roi_path)
    if neck_roi is None:
        return

    # Step 2: Process forehead PPG
    R_forehead, G_forehead, B_forehead = process_ppg_forehead(video_path, output_dir)

    # Step 3: Process neck PPG
    T_mean_signals_neck = process_ppg_neck(video_path, neck_roi, output_dir)

    # Step 4: Calculate neck PPW
    ppw_neck = calculate_ppw_neck(video_path, neck_roi, output_dir)

    # Step 5: Preprocessa e filtra i segnali
    # Per il fronte, filtriamo separatamente ciascun canale
    filtered_R_forehead = preprocess_and_filter_signals(R_forehead, sampling_rate, output_dir, "forehead_R")
    filtered_G_forehead = preprocess_and_filter_signals(G_forehead, sampling_rate, output_dir, "forehead_G")
    filtered_B_forehead = preprocess_and_filter_signals(B_forehead, sampling_rate, output_dir, "forehead_B")

    neck_green = T_mean_signals_neck[:, 1]
    filtered_neck_green = preprocess_and_filter_signals(neck_green, sampling_rate, output_dir, "neck_green")

    # Per il collo, si assume che T_mean_signals_neck sia già disponibile
    filtered_neck = preprocess_and_filter_signals(T_mean_signals_neck, sampling_rate, output_dir, "neck")

    # Step 6: Estrai il BVP
    print("\n[STEP 6] Extracting BVP...")
    # Per il fronte utilizziamo tutti e tre i canali (R, G, B)
    min_length = min(len(filtered_R_forehead), len(filtered_G_forehead), len(filtered_B_forehead))
    filtered_R_forehead = filtered_R_forehead[:min_length]
    filtered_G_forehead = filtered_G_forehead[:min_length]
    filtered_B_forehead = filtered_B_forehead[:min_length]
    filtered_forehead = np.vstack((filtered_R_forehead, filtered_G_forehead, filtered_B_forehead)).T
    bvp_forehead = cpu_CHROM(filtered_forehead, output_dir, label="forehead")
    # Per il collo, utilizziamo solo il segnale verde
    bvp_neck, neck_peaks = compute_bvp_green_neck(filtered_neck_green, sampling_rate, output_dir, label="neck")      
    print("[INFO] BVP signals extracted successfully.")

    # Step 7: Detect Peaks
    print("\n[STEP 7] Detecting peaks...")
    forehead_peaks = find_peaks_in_signal(bvp_forehead, fps=sampling_rate, output_dir=output_dir, label="forehead")
    neck_peaks_ppg = find_peaks_in_signal(bvp_neck, fps=sampling_rate, output_dir=output_dir, label="neck_ppg")
    neck_peaks_ppw = find_peaks_in_signal(ppw_neck, fps=sampling_rate, output_dir=output_dir, label="neck_ppw")

    # Visualizzazione della sincronizzazione dei picchi
    plot_peak_synchronization(forehead_peaks, neck_peaks_ppg, bvp_forehead, bvp_neck, sampling_rate, output_dir)

    # Step 8: Calculate PTT using Cross-Correlation Method
    print("\n[STEP 8] Calculating PTT using Cross-Correlation Method...")

    if len(bvp_forehead) > 0 and len(bvp_neck) > 0:
        ptt_cross_corr = calculate_ptt_cross_correlation(forehead_peaks, neck_peaks_ppg, sampling_rate, output_dir, label="CROSS_PPG_F_PPG_N")
        ptt_cross_corr = validate_ptt(ptt_cross_corr)
        print(f"[INFO] PTT Forehead-PPG Neck (Cross-Correlation): {ptt_cross_corr}")

        ptt_cross_corr_ppw = calculate_ptt_cross_correlation(forehead_peaks, neck_peaks_ppw, sampling_rate, output_dir, label="CROSS_PPG_F_PPW_N")
        ptt_cross_corr_ppw = validate_ptt(ptt_cross_corr_ppw)
        print(f"[INFO] PTT Forehead-PPW Neck (Cross-Correlation): {ptt_cross_corr_ppw}")
    else:
        print("[ERROR] BVP signals are empty. Cannot calculate PTT.")

    # Step 9: Calculate PTT using Direct Method
    print("\n[STEP 9] Calculating PTT using Direct Method...")

    if len(bvp_forehead) > 0 and len(bvp_neck) > 0:
        ptt_direct = calculate_ptt_direct(forehead_peaks, neck_peaks_ppg, sampling_rate, output_dir, label="DIRECT_PPG_F_PPG_N")
        ptt_direct = validate_ptt(ptt_direct)
        print(f"[INFO] PTT Forehead-PPG Neck (Direct): {ptt_direct}")

        ptt_direct_ppw = calculate_ptt_direct(forehead_peaks, neck_peaks_ppw, sampling_rate, output_dir, label="DIRECT_PPG_F_PPW_N")
        ptt_direct_ppw = validate_ptt(ptt_direct_ppw)
        print(f"[INFO] PTT Forehead-PPW Neck (Direct): {ptt_direct_ppw}")
    else:
        print("[ERROR] BVP signals are empty. Cannot calculate PTT.")

    # --- INTERPOLATION STEP ----
    # Definisci il target di campionamento desiderato (ad esempio 100 fps)
    target_rate = 100

    # Applica l'interpolazione per convoluzione cubica sui segnali filtrati
    interpolated_R_forehead_conv = convolution_interpolation(filtered_R_forehead, sampling_rate, target_rate, output_dir, label="forehead_R_conv")
    interpolated_G_forehead_conv = convolution_interpolation(filtered_G_forehead, sampling_rate, target_rate, output_dir, label="forehead_G_conv")
    interpolated_B_forehead_conv = convolution_interpolation(filtered_B_forehead, sampling_rate, target_rate, output_dir, label="forehead_B_conv")
    interpolated_neck_green_conv = convolution_interpolation(filtered_neck_green, sampling_rate, target_rate, output_dir, label="neck_green_conv")

    # Applica l'interpolazione per spline cubica sui segnali filtrati
    interpolated_R_forehead_spline = cubic_spline_interpolation(filtered_R_forehead, sampling_rate, target_rate, output_dir, label="forehead_R_spline")
    interpolated_G_forehead_spline = cubic_spline_interpolation(filtered_G_forehead, sampling_rate, target_rate, output_dir, label="forehead_G_spline")
    interpolated_B_forehead_spline = cubic_spline_interpolation(filtered_B_forehead, sampling_rate, target_rate, output_dir, label="forehead_B_spline")
    interpolated_neck_green_spline = cubic_spline_interpolation(filtered_neck_green, sampling_rate, target_rate, output_dir, label="neck_green_spline")

    # Ora il nuovo sampling rate è quello target
    sampling_rate = target_rate
    print(f"[INFO] Signals interpolated to {target_rate} fps.")

    # Calcola il BVP per entrambe le interpolazioni
    filtered_forehead_conv = np.vstack((interpolated_R_forehead_conv, interpolated_G_forehead_conv, interpolated_B_forehead_conv)).T
    bvp_forehead_conv = cpu_CHROM(filtered_forehead_conv, output_dir, label="forehead_conv")
    bvp_neck_conv, neck_peaks_conv = compute_bvp_green_neck(interpolated_neck_green_conv, sampling_rate, output_dir, label="neck_conv")

    filtered_forehead_spline = np.vstack((interpolated_R_forehead_spline, interpolated_G_forehead_spline, interpolated_B_forehead_spline)).T
    bvp_forehead_spline = cpu_CHROM(filtered_forehead_spline, output_dir, label="forehead_spline")
    bvp_neck_spline, neck_peaks_spline = compute_bvp_green_neck(interpolated_neck_green_spline, sampling_rate, output_dir, label="neck_spline")

    # Step 7: Detect Peaks for both interpolations
    print("\n[STEP 7] Detecting peaks...")
    forehead_peaks_conv = find_peaks_in_signal(bvp_forehead_conv, fps=sampling_rate, output_dir=output_dir, label="forehead_conv")
    forehead_peaks_spline = find_peaks_in_signal(bvp_forehead_spline, fps=sampling_rate, output_dir=output_dir, label="forehead_spline")

    # Visualizzazione della sincronizzazione dei picchi per entrambe le interpolazioni
    plot_peak_synchronization(forehead_peaks_conv, neck_peaks_conv, bvp_forehead_conv, bvp_neck_conv, sampling_rate, output_dir)
    plot_peak_synchronization(forehead_peaks_spline, neck_peaks_spline, bvp_forehead_spline, bvp_neck_spline, sampling_rate, output_dir)

    # Step 8: Calculate PTT using Cross-Correlation Method for both interpolations
    print("\n[STEP 8] Calculating PTT using Cross-Correlation Method...")

    if len(bvp_forehead_conv) > 0 and len(bvp_neck_conv) > 0:
        ptt_cross_corr_conv = calculate_ptt_cross_correlation(forehead_peaks_conv, neck_peaks_conv, sampling_rate, output_dir, label="CROSS_PPG_F_PPG_N_conv")
        ptt_cross_corr_conv = validate_ptt(ptt_cross_corr_conv)
        print(f"[INFO] PTT Forehead-PPG Neck (Cross-Correlation, Convolution): {ptt_cross_corr_conv}")

        ptt_cross_corr_ppw_conv = calculate_ptt_cross_correlation(forehead_peaks_conv, neck_peaks_ppw, sampling_rate, output_dir, label="CROSS_PPG_F_PPW_N_conv")
        ptt_cross_corr_ppw_conv = validate_ptt(ptt_cross_corr_ppw_conv)
        print(f"[INFO] PTT Forehead-PPW Neck (Cross-Correlation, Convolution): {ptt_cross_corr_ppw_conv}")

    if len(bvp_forehead_spline) > 0 and len(bvp_neck_spline) > 0:
        ptt_cross_corr_spline = calculate_ptt_cross_correlation(forehead_peaks_spline, neck_peaks_spline, sampling_rate, output_dir, label="CROSS_PPG_F_PPG_N_spline")
        ptt_cross_corr_spline = validate_ptt(ptt_cross_corr_spline)
        print(f"[INFO] PTT Forehead-PPG Neck (Cross-Correlation, Spline): {ptt_cross_corr_spline}")

        ptt_cross_corr_ppw_spline = calculate_ptt_cross_correlation(forehead_peaks_spline, neck_peaks_ppw, sampling_rate, output_dir, label="CROSS_PPG_F_PPW_N_spline")
        ptt_cross_corr_ppw_spline = validate_ptt(ptt_cross_corr_ppw_spline)
        print(f"[INFO] PTT Forehead-PPW Neck (Cross-Correlation, Spline): {ptt_cross_corr_ppw_spline}")

    # Step 9: Calculate PTT using Direct Method for both interpolations
    print("\n[STEP 9] Calculating PTT using Direct Method...")

    if len(bvp_forehead_conv) > 0 and len(bvp_neck_conv) > 0:
        ptt_direct_conv = calculate_ptt_direct(forehead_peaks_conv, neck_peaks_conv, sampling_rate, output_dir, label="PPG_F_PPG_N_conv")
        ptt_direct_conv = validate_ptt(ptt_direct_conv)
        print(f"[INFO] PTT Forehead-PPG Neck (Direct, Convolution): {ptt_direct_conv}")

        ptt_direct_ppw_conv = calculate_ptt_direct(forehead_peaks_conv, neck_peaks_ppw, sampling_rate, output_dir, label="PPG_F_PPW_N_conv")
        ptt_direct_ppw_conv = validate_ptt(ptt_direct_ppw_conv)
        print(f"[INFO] PTT Forehead-PPW Neck (Direct, Convolution): {ptt_direct_ppw_conv}")

    if len(bvp_forehead_spline) > 0 and len(bvp_neck_spline) > 0:
        ptt_direct_spline = calculate_ptt_direct(forehead_peaks_spline, neck_peaks_spline, sampling_rate, output_dir, label="PPG_F_PPG_N_spline")
        ptt_direct_spline = validate_ptt(ptt_direct_spline)
        print(f"[INFO] PTT Forehead-PPG Neck (Direct, Spline): {ptt_direct_spline}")

        ptt_direct_ppw_spline = calculate_ptt_direct(forehead_peaks_spline, neck_peaks_ppw, sampling_rate, output_dir, label="PPG_F_PPW_N_spline")
        ptt_direct_ppw_spline = validate_ptt(ptt_direct_ppw_spline)
        print(f"[INFO] PTT Forehead-PPW Neck (Direct, Spline): {ptt_direct_ppw_spline}")
        # Calcola AIx per tutti i BVP calcolati
        print("\n[STEP 10] Calculating AIx...")
        aix_values_conv = calculate_aix(bvp_forehead_conv, bvp_neck, forehead_peaks_conv, neck_peaks_ppg, sampling_rate, output_dir)
        aix_values_spline = calculate_aix(bvp_forehead_spline, bvp_neck, forehead_peaks_spline, neck_peaks_ppg, sampling_rate, output_dir)
        aix_values_red = calculate_aix_red(filtered_R_forehead, bvp_neck, forehead_peaks, neck_peaks_ppg, sampling_rate, output_dir)
        #aix_values_red = calculate_aix_red(filtered_R_forehead, bvp_neck, forehead_peaks, neck_peaks_ppg, sampling_rate, output_dir)
        print(f"[INFO] AIx values calculated successfully.")

    # Stima della pressione sanguigna
    print("\n[STEP 11] Estimating Blood Pressure...")
    estimated_sbp_conv = estimate_systolic_blood_pressure(ptt_direct_conv, gender)
    estimated_dbp_conv = estimate_diastolic_blood_pressure(ptt_direct_conv, gender)
    estimated_sbp_spline = estimate_systolic_blood_pressure(ptt_direct_spline, gender)
    estimated_dbp_spline = estimate_diastolic_blood_pressure(ptt_direct_spline, gender)
    estimated_sbp_aix_conv, estimated_dbp_aix_conv = estimate_bp_from_aix(aix_values_conv, output_dir)
    estimated_sbp_aix_spline, estimated_dbp_aix_spline = estimate_bp_from_aix(aix_values_spline, output_dir)
    estimated_sbp_aix_red, estimated_dbp_aix_red = estimate_bp_from_aix(aix_values_red, output_dir)

    # Salva e visualizza i risultati per entrambe le interpolazioni
    save_and_visualize_results(bvp_forehead_conv, bvp_neck, forehead_peaks_conv, neck_peaks_ppg, ptt_direct_conv, estimated_sbp_conv, estimated_dbp_conv, output_dir, label="Convolution")
    save_and_visualize_results(bvp_forehead_spline, bvp_neck, forehead_peaks_spline, neck_peaks_ppg, ptt_direct_spline, estimated_sbp_spline, estimated_dbp_spline, output_dir, label="Spline")

    # Confronta i metodi di interpolazione
    compare_ptt_methods(ptt_direct_conv, ptt_direct_spline, output_dir, label="Direct Method")
    compare_ptt_methods(ptt_cross_corr_conv, ptt_cross_corr_spline, output_dir, label="Cross-Correlation Method")

    # Step 12: Confronto tra metodi di stima della pressione sanguigna
    print("\n[STEP 12] Analyzing Blood Pressure Estimation Methods...")
    analyze_bp_methods(estimated_sbp_conv, estimated_dbp_conv, estimated_sbp_spline, estimated_dbp_spline, estimated_sbp_aix_conv, estimated_dbp_aix_conv, output_dir)
    analyze_bp_methods(estimated_sbp_conv, estimated_dbp_conv, estimated_sbp_spline, estimated_dbp_spline, estimated_sbp_aix_spline, estimated_dbp_aix_spline, output_dir)

    print("\n[STEP 13] Comparing Blood Pressure Estimation Methods...")
    compare_bp_estimations(estimated_sbp_conv, estimated_dbp_conv, estimated_sbp_spline, estimated_dbp_spline, estimated_sbp_aix_conv, estimated_dbp_aix_conv, output_dir)
    compare_bp_estimations(estimated_sbp_conv, estimated_dbp_conv, estimated_sbp_spline, estimated_dbp_spline, estimated_sbp_aix_spline, estimated_dbp_aix_spline, output_dir)

    print("\n[STEP 14] Computing correlation between PTT and Blood Pressure...\n")
    correlate_bp_ptt(ptt_direct_conv, bp_systolic, "SBP")
    correlate_bp_ptt(ptt_direct_conv, bp_diastolic, "DBP")
    correlate_bp_ptt(ptt_direct_spline, bp_systolic, "SBP")
    correlate_bp_ptt(ptt_direct_spline, bp_diastolic, "DBP")

    ptt_methods = {
    "Direct_Conv": ptt_direct_conv,
    "Direct_Spline": ptt_direct_spline,
    "CrossConv": ptt_cross_corr_conv,
    "CrossSpline": ptt_cross_corr_spline
    }

    compare_all_ptt_methods(ptt_methods, output_dir)

    # Plot dei risultati
    plot_ptt_vs_bp(ptt_direct_conv, bp_systolic, "SBP", output_dir)
    plot_ptt_vs_bp(ptt_direct_conv, bp_diastolic, "DBP", output_dir)
    plot_ptt_vs_bp(ptt_direct_spline, bp_systolic, "SBP", output_dir)
    plot_ptt_vs_bp(ptt_direct_spline, bp_diastolic, "DBP", output_dir)

    # --- Chiamata per calcolare i risultati di Bland-Altman ---
    bland_altman_results(estimated_sbp_conv, bp_systolic, estimated_dbp_conv, bp_diastolic, output_dir)
    bland_altman_results(estimated_sbp_spline, bp_systolic, estimated_dbp_spline, bp_diastolic, output_dir)
    bland_altman_results(estimated_sbp_aix_conv, bp_systolic, estimated_dbp_aix_conv, bp_diastolic, output_dir)
    bland_altman_results(estimated_sbp_aix_spline, bp_systolic, estimated_dbp_aix_spline, bp_diastolic, output_dir)

    print("\n[INFO] All processing steps completed successfully.")

if __name__ == "__main__":
    main()