# --- LIBRARY IMPORTS ---
import json
print("json imported successfully!")

import os
print("os imported successfully!")

import numpy as np
print("numpy imported successfully!")

import mediapipe as mp
print("mediapipe imported successfully!")

import matplotlib.pyplot as plt
print("matplotlib imported successfully!")

from scipy.signal import butter, sosfiltfilt, find_peaks, resample  # Aggiunto 'resample' per l'AIx
print("scipy.signal imported successfully!")

import scipy.sparse as sparse
print("scipy.sparse imported successfully!")

import scipy.sparse.linalg as splinalg
print("scipy.sparse.linalg imported successfully!")

import cv2
print("opencv-python imported successfully!")

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

def segment_signal(signal, window_length_sec, fps):
    """
    Segmenta il segnale in finestre di una lunghezza specifica in secondi.

    :param signal: Segnale da segmentare.
    :param window_length_sec: Lunghezza della finestra in secondi.
    :param fps: Frame per secondo del video.
    :return: Lista di segmenti di segnale.
    """
    window_length = int(window_length_sec * fps)
    segments = [signal[i:i + window_length] for i in range(0, len(signal), window_length)]
    return segments

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

    return rgb_signals

# --- NECK PPW EXTRACTION USING PIXFLOW ---
def extract_pixflow_signal(video_path, roi, output_folder, part):
    """
    Extract PPW signal from neck using PixFlow algorithm.
    Measures pixel displacements due to arterial pulsations in x direction.
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Average x displacement
        avg_flow_x = np.mean(flow[..., 0])
        flow_signal.append(avg_flow_x)
        prev_gray = curr_gray

    cap.release()

    # Save the flow signal
    flow_signal = np.array(flow_signal)
    np.savetxt(f"{output_folder}/flow_signal_{part}.txt", flow_signal)
    plt.plot(flow_signal)
    plt.title(f'Flow Signal ({part})')
    plt.xlabel('Frame')
    plt.ylabel('Average Flow X')
    plt.savefig(f"{output_folder}/flow_signal_{part}.jpg")
    plt.close()

    return flow_signal

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

def find_peaks_in_signal(ppg_signal, fps=30, window_size=5):
    """
    Migliorato il rilevamento dei picchi nel segnale PPG con:
    - Adattamento della prominenza basato sulla deviazione standard del segnale
    - Regolazione dinamica della distanza minima tra i picchi in base al frame rate (FPS)
    - Filtro passa-basso con media mobile per ridurre il rumore
    - Filtro IQR migliorato per ridurre gli outlier
    
    :param ppg_signal: Array con il segnale PPG
    :param fps: Frame rate del video (default: 30 FPS)
    :param window_size: Dimensione della finestra per il filtro passa-basso (default: 5)
    :return: Indici dei picchi validi nel segnale
    """

    # 1️⃣ FILTRO PASSA-BASSO: Media mobile per ridurre il rumore
    smoothed_signal = np.convolve(ppg_signal, np.ones(window_size)/window_size, mode='valid')

    # 2️⃣ ADATTAMENTO AUTOMATICO DELLA PROMINENZA
    prominence_value = 0.15 * np.std(smoothed_signal)

    # 3️⃣ DISTANZA MINIMA TRA I PICCHI IN BASE ALLA FREQUENZA CARDIACA
    # Supponiamo un range realistico di 50-120 BPM → convertiamo in intervalli tra picchi
    min_bpm = 50
    max_bpm = 120
    min_distance = int(fps * 60 / max_bpm)  # Minima distanza in frame (per 120 BPM)
    max_distance = int(fps * 60 / min_bpm)  # Massima distanza in frame (per 50 BPM)

    peaks, properties = find_peaks(smoothed_signal, distance=min_distance, prominence=prominence_value)

    # 4️⃣ FILTRO OUTLIER USANDO IQR
    peak_heights = smoothed_signal[peaks]
    q1, q3 = np.percentile(peak_heights, [25, 75])
    iqr_value = iqr(peak_heights)
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value

    valid_peaks = peaks[(peak_heights >= lower_bound) & (peak_heights <= upper_bound)]

    print(f"[INFO] Rilevati {len(peaks)} picchi iniziali, ridotti a {len(valid_peaks)} dopo il filtro IQR.")
    
    return valid_peaks

# --- FEATURE EXTRACTION FROM CLEANED SIGNAL ---
def cpu_CHROM(signal):
    """
    CHROM method on CPU using Numpy.

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. 
    IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """
    X = signal
    Xcomp = 3 * X[:, 0] - 2 * X[:, 1]
    Ycomp = 1.5 * X[:, 0] + X[:, 1] - 1.5 * X[:, 2]
    sX = np.std(Xcomp, axis=0)
    sY = np.std(Ycomp, axis=0)
    alpha = sX / sY
    bvp = Xcomp - alpha * Ycomp
    return bvp

def cpu_POS(signal):
    """
    Implement the Plane Orthogonal to Skin (POS) algorithm.
    """
    # Remove DC component
    mean_rgb = np.mean(signal, axis=0)
    S = signal - mean_rgb  # S ha dimensione (N, 2) perché usiamo solo R e G

    # Proiezione corretta (2x2 invece di 2x3)
    H = np.array([[1, -1],  
                  [1, 1]])

    # Applica la proiezione
    Xs = H @ S.T  # Ora H (2,2) e S.T (2,N) → Nessun errore
    S1 = Xs[0, :]
    S2 = Xs[1, :]

    alpha = np.std(S1) / np.std(S2)
    bvp = S1 - alpha * S2
    return bvp

from scipy.interpolate import interp1d
import numpy as np

def calculate_ptt(peaks_forehead, peaks_neck, fps, output_dir):
    """
    Calcola il Pulse Transit Time (PTT) basandosi sui picchi rilevati nei segnali della fronte e del collo.
    
    - Ottimizza il matching scegliendo il picco più vicino invece del primo successivo.
    - Filtra automaticamente i PTT fuori dai range fisiologici (0.1s - 0.4s).
    - Evita problemi causati da picchi spuri o errati.
    
    :param peaks_forehead: Indici dei picchi rilevati nel segnale della fronte.
    :param peaks_neck: Indici dei picchi rilevati nel segnale del collo.
    :param fps: Frame rate del video per la conversione dei tempi.
    :param output_dir: Directory di output per salvare i risultati.
    :return: Array dei valori PTT calcolati.
    """
    
    # Convertiamo gli indici dei picchi in tempo (secondi)
    peaks_forehead = np.array(peaks_forehead) / fps
    peaks_neck = np.array(peaks_neck) / fps

    print(f"[DEBUG] Peaks Forehead (time): {peaks_forehead}")
    print(f"[DEBUG] Peaks Neck (time): {peaks_neck}")

    # Se non ci sono picchi in uno dei due segnali, restituiamo un array vuoto
    if len(peaks_forehead) == 0 or len(peaks_neck) == 0:
        print("[ERROR] No peaks detected in one of the signals!")
        return np.array([])

    ptt_values = []
    for t_n in peaks_neck:
        t_f_candidates = peaks_forehead[peaks_forehead > t_n]

        if len(t_f_candidates) > 0:
            # ✅ Trova il picco della fronte più vicino a quello del collo
            t_f = min(t_f_candidates, key=lambda t: abs(t - t_n))

            # ✅ Calcolo PTT (differenza tra i due picchi)
            ptt_value = t_f - t_n

            # ✅ Filtraggio PTT per valori fisiologici (0.1s - 0.4s)
            if 0.1 <= ptt_value <= 0.4:
                ptt_values.append(ptt_value)
            else:
                print(f"[WARNING] PTT={ptt_value:.3f}s fuori range fisiologico!")

    # Convertiamo in numpy array per facilitarne la gestione
    ptt_values = np.array(ptt_values)

    # ✅ Debugging: Stampiamo la media e la deviazione standard dei PTT
    if len(ptt_values) > 0:
        print(f"[INFO] PTT Mean: {np.mean(ptt_values):.3f}s, Std Dev: {np.std(ptt_values):.3f}s")
    
    # ✅ Salvataggio dei valori PTT in un file di testo
    np.savetxt(os.path.join(output_dir, 'ptt_values.txt'), ptt_values, fmt='%.5f')

    print(f"[DEBUG] Computed PTT values: {ptt_values}")  

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

def estimate_bp_from_aix(aix_values):
    """
    Stima la pressione sanguigna (SBP e DBP) dall'Augmentation Index (AIx).
    
    :param aix_values: Array di valori AIx.
    :return: Tuple con SBP e DBP stimati.
    """
    # Coefficienti basati su studi medici (da calibrare con dati reali)
    a_sbp, b_sbp = 0.5, 120
    a_dbp, b_dbp = 0.3, 80

    estimated_sbp = a_sbp * aix_values + b_sbp
    estimated_dbp = a_dbp * aix_values + b_dbp

    return estimated_sbp, estimated_dbp

from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

        plt.figure(figsize=(6, 5))
        plt.scatter(mean, diff, alpha=0.5)
        plt.axhline(md, color='red', linestyle='--', label="Mean Diff")
        plt.axhline(md + 1.96 * sd, color='blue', linestyle='--', label="+1.96 SD")
        plt.axhline(md - 1.96 * sd, color='blue', linestyle='--', label="-1.96 SD")
        plt.xlabel("Mean BP Estimation")
        plt.ylabel("Difference")
        plt.title(f"Bland-Altman Plot ({label})")
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    bland_altman_plot(sbp_ppg, sbp_ppw, "SBP PPG vs PPW", f"{output_dir}/bland_altman_sbp_ppg_ppw.png")
    bland_altman_plot(sbp_ppg, sbp_aix, "SBP PPG vs AIx", f"{output_dir}/bland_altman_sbp_ppg_aix.png")
    bland_altman_plot(dbp_ppg, dbp_ppw, "DBP PPG vs PPW", f"{output_dir}/bland_altman_dbp_ppg_ppw.png")
    bland_altman_plot(dbp_ppg, dbp_aix, "DBP PPG vs AIx", f"{output_dir}/bland_altman_dbp_ppg_aix.png")

    # T-Test per verificare differenze significative
    for (col1, col2) in [("SBP_PPG", "SBP_PPW"), ("SBP_PPG", "SBP_AIx"), ("DBP_PPG", "DBP_PPW"), ("DBP_PPG", "DBP_AIx")]:
        t_stat, p_value = ttest_rel(df[col1], df[col2])
        print(f"T-Test {col1} vs {col2}: t-statistic={t_stat:.3f}, p-value={p_value:.3f}")
    
    print("[INFO] Analisi statistica completata! Grafici salvati.")

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

def save_estimated_bp(bp_values, label, output_dir):
    np.savetxt(os.path.join(output_dir, f'estimated_{label.lower()}.txt'), bp_values, fmt='%.2f')

# --- MAIN --- 
def initialize_paths_and_config():
    print("\n[INFO] Initializing paths and configurations...")
    dataset_folder = "/Volumes/DanoUSB"
    subject = "M007"
    task = "T9"
    gender = "male"
    video_path = f"{dataset_folder}/{subject}/{task}/vid.avi"
    output_dir = f"NIAC/{subject}/{task}"
    neck_roi_file_path = f"{output_dir}/neck_roi_dimensions.json"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    return video_path, output_dir, neck_roi_file_path, gender

def calculate_roi(video_path, neck_roi_file_path):
    print("\n[STEP 1] Calculating ROI...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read frame from video.")
        return None

    if os.path.exists(neck_roi_file_path):
        neck_roi = load_roi_dimensions(neck_roi_file_path)
        print("[INFO] Neck ROI loaded from file.")
    else:
        print("[INFO] Select neck ROI manually.")
        neck_roi = select_neck_roi(frame)
        save_roi_dimensions(neck_roi, neck_roi_file_path)
        print("[INFO] Neck ROI saved to file.")
    return neck_roi

def process_ppg_forehead(video_path, output_dir):
    print("\n[STEP 2] Processing forehead PPG...")
    rgb_signals = extract_rgb_trace_MediaPipe(video_path, output_dir)
    signals = np.vstack((rgb_signals['R'], rgb_signals['G'], rgb_signals['B'])).T
    print("[INFO] RGB signal from forehead extracted successfully.")
    if signals.shape[0] < 28:
        needed = 28 - signals.shape[0]
        last_samples = signals[-1:].repeat(needed, axis=0)
        signals = np.vstack((signals, last_samples))
        print(f"[WARNING] Forehead signal had less than 28 samples. Extended to {signals.shape[0]} samples.")
    return signals

def process_ppg_neck(video_path, neck_roi, output_dir):
    print("\n[STEP 3] Processing neck PPG...")

    # ✅ Controllo validità ROI
    if neck_roi is None or len(neck_roi) != 4:
        print("[ERROR] Invalid Neck ROI. Check ROI selection.")
        return None
    
    x, y, w, h = neck_roi
    if w <= 0 or h <= 0:
        print(f"[ERROR] Invalid ROI dimensions: {neck_roi}")
        return None

    print(f"[INFO] Using Neck ROI: x={x}, y={y}, w={w}, h={h}")

    # ✅ Debug: Prova a caricare un singolo frame prima di processare tutto il video
    cap = cv2.VideoCapture(video_path)
    ret, image = cap.read()
    if not ret:
        print("[ERROR] Could not read the first frame of the video.")
        return None
    cap.release()

    print("[INFO] First frame read successfully.")

    rgb_signals = extract_rgb_trace(video_path, neck_roi, output_dir, "Neck")

    print("[DEBUG] Extracted RGB signals for Neck")

    # ✅ Debug: Verifica se abbiamo dati validi
    if len(rgb_signals['R']) == 0 or len(rgb_signals['G']) == 0:
        print("[ERROR] No valid RGB data extracted from Neck.")
        return None

    red_signal = np.array(rgb_signals['R'])
    green_signal = np.array(rgb_signals['G'])

    print(f"[DEBUG] Red signal shape: {red_signal.shape}, Green signal shape: {green_signal.shape}")

    # ✅ Debug: Controllo su eventuali dati anomali
    if red_signal.shape[0] < 5 or green_signal.shape[0] < 5:
        print(f"[ERROR] Signals are too short! Red: {red_signal.shape[0]}, Green: {green_signal.shape[0]}")
        return None

    # Compute the mean of Red and Green signals
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
    ppw_signal = extract_pixflow_signal(video_path, neck_roi, output_dir, "Neck")
    print("[INFO] Neck PPW signal extracted successfully.")
    return ppw_signal

def preprocess_and_filter_signals(signals, sampling_rate):
    detrended_signals = smoothness_priors_detrend(signals, lambda_param=10)
    filtered_signals = butter_bandpass_filter(
        detrended_signals,
        lowcut=0.7,
        highcut=4.0,
        fs=sampling_rate
    )
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

from scipy.stats import pearsonr

def compare_ptt_methods(ptt_ppw, ptt_ppg):
    """
    Confronta il PTT calcolato con PPG-PPW e PPG-PPG usando la correlazione di Pearson.
    """
    if len(ptt_ppw) == 0 or len(ptt_ppg) == 0:
        print("[ERROR] One of the PTT arrays is empty. Cannot compare.")
        return

    correlation, _ = pearsonr(ptt_ppw, ptt_ppg)
    print(f"[INFO] Correlation between PPG-PPW PTT and PPG-PPG PTT: {correlation:.3f}")

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
    sampling_rate = get_sampling_rate(video_path)
    print(f"[INFO] Sampling rate: {sampling_rate} FPS\n")

        # Percorsi ai file di pressione sanguigna
    bp_systolic_file = "/Volumes/DanoUSB/Physiology/M007/T9/LA Systolic BP_mmHg.txt"
    bp_diastolic_file = "/Volumes/DanoUSB/Physiology/M007/T9/BP Dia_mmHg.txt"

    # Lettura dei dati dai file
    bp_systolic = read_blood_pressure_data(bp_systolic_file)
    bp_diastolic = read_blood_pressure_data(bp_diastolic_file)

    print(f"[INFO] Pressione Sistolica: {bp_systolic[:10]}")
    print(f"[INFO] Pressione Diastolica: {bp_diastolic[:10]}")
    # Step 1: Calculate ROI
    neck_roi = calculate_roi(video_path, neck_roi_file_path)
    if neck_roi is None:
        return

    # Step 2: Process forehead PPG
    T_rgb_signals_forehead = process_ppg_forehead(video_path, output_dir)

    # Step 3: Process neck PPG
    T_mean_signals_neck = process_ppg_neck(video_path, neck_roi, output_dir)


    # Step 4: Calculate neck PPW
    ppw_neck = calculate_ppw_neck(video_path, neck_roi, output_dir)

     # Step 5: Preprocess and filter signals
    filtered_forehead = preprocess_and_filter_signals(T_rgb_signals_forehead, sampling_rate)
    filtered_neck = preprocess_and_filter_signals(T_mean_signals_neck, sampling_rate)


    # Step 6: Extract BVP
    print("\n[STEP 6] Extracting BVP...")
    bvp_forehead = cpu_CHROM(filtered_forehead)  
    bvp_neck = cpu_POS(filtered_neck)  
    print("[INFO] BVP signals extracted successfully.")

    # Step 7: Detect peaks
    print("\n[STEP 7] Detecting peaks...")
    forehead_peaks = find_peaks_in_signal(bvp_forehead, fps=sampling_rate)
    neck_peaks_ppg = find_peaks_in_signal(bvp_neck, fps=sampling_rate)

    neck_peaks_ppw = find_peaks_in_signal(ppw_neck, fps=sampling_rate)
    
    print("\n[STEP 8] Calculating PTT...")

# Assicurati di avere entrambe le variabili correttamente assegnate
    ptt_forehead_ppw_neck = calculate_ptt(forehead_peaks, neck_peaks_ppw, sampling_rate, output_dir)
    ptt_forehead_ppw_neck = validate_ptt(ptt_forehead_ppw_neck)

    ptt_forehead_ppg_neck = calculate_ptt(forehead_peaks, neck_peaks_ppg, sampling_rate, output_dir)
    ptt_forehead_ppg_neck = validate_ptt(ptt_forehead_ppg_neck)

    print(f"[INFO] PTT Forehead-PPW Neck: {ptt_forehead_ppw_neck}")
    print(f"[INFO] PTT Forehead-PPG Neck: {ptt_forehead_ppg_neck}")

    # Verifica che le variabili siano state correttamente assegnate
    if ptt_forehead_ppw_neck.size == 0 or ptt_forehead_ppg_neck.size == 0:
        print("[ERROR] Nessun valore PTT valido calcolato! Interrompo l'esecuzione.")

    # Step 9: Estimate Blood Pressure
    print("\n[STEP 9] Estimating Blood Pressure...")

    estimated_sbp_ppw = estimate_systolic_blood_pressure(ptt_forehead_ppw_neck, gender)
    estimated_dbp_ppw = estimate_diastolic_blood_pressure(ptt_forehead_ppw_neck, gender)

    estimated_sbp_ppg = estimate_systolic_blood_pressure(ptt_forehead_ppg_neck, gender)
    estimated_dbp_ppg = estimate_diastolic_blood_pressure(ptt_forehead_ppg_neck, gender)

    print(f"[INFO] Estimated SBP (PPW): {estimated_sbp_ppw}")
    print(f"[INFO] Estimated DBP (PPW): {estimated_dbp_ppw}")

    print(f"[INFO] Estimated SBP (PPG): {estimated_sbp_ppg}")
    print(f"[INFO] Estimated DBP (PPG): {estimated_dbp_ppg}")

    # Step 10: Calculate AIx
    print("\n[STEP 10] Calculating AIx...")
    aix_values = calculate_aix(bvp_forehead, bvp_neck, forehead_peaks, neck_peaks_ppg, sampling_rate, output_dir)

    print("\n[STEP 11] Estimating Blood Pressure from AIx...")
    estimated_sbp_aix, estimated_dbp_aix = estimate_bp_from_aix(aix_values)

    save_and_visualize_results(bvp_forehead, bvp_neck, forehead_peaks, neck_peaks_ppg, ptt_forehead_ppg_neck, estimated_sbp_ppg, estimated_dbp_ppg, output_dir, label="PPG-PPG")
    save_and_visualize_results(bvp_forehead, bvp_neck, forehead_peaks, neck_peaks_ppw, ptt_forehead_ppw_neck, estimated_sbp_ppw, estimated_dbp_ppw, output_dir, label="PPG-PPW")
    save_and_visualize_results(bvp_forehead, bvp_neck, forehead_peaks, neck_peaks_ppg, aix_values, estimated_sbp_aix, estimated_dbp_aix, output_dir, label="AIx")

    print("\n[STEP 16] Evaluating Estimation Accuracy...\n")
    evaluate_estimations(estimated_sbp_ppg, bp_systolic, "SBP_PPG", output_dir)
    evaluate_estimations(estimated_dbp_ppg, bp_diastolic, "DBP_PPG", output_dir)
    evaluate_estimations(estimated_sbp_ppw, bp_systolic, "SBP_PPW", output_dir)
    evaluate_estimations(estimated_dbp_ppw, bp_diastolic, "DBP_PPW", output_dir)
    evaluate_estimations(estimated_sbp_aix, bp_systolic, "SBP_AIx", output_dir)
    evaluate_estimations(estimated_dbp_aix, bp_diastolic, "DBP_AIx", output_dir)
    # Step 13: Confronto tra metodi di stima della pressione sanguigna
    print("\n[STEP 13] Analyzing Blood Pressure Estimation Methods...")
    analyze_bp_methods(estimated_sbp_ppg, estimated_dbp_ppg, estimated_sbp_ppw, estimated_dbp_ppw, estimated_sbp_aix, estimated_dbp_aix, output_dir)

    print("\n[STEP 14] Comparing Blood Pressure Estimation Methods...")
    compare_bp_estimations(estimated_sbp_ppg, estimated_dbp_ppg, estimated_sbp_ppw, estimated_dbp_ppw, estimated_sbp_aix, estimated_dbp_aix, output_dir)

    print("\n[STEP 15] Computing correlation between PTT and Blood Pressure...\n")
    correlate_bp_ptt(ptt_forehead_ppg_neck, bp_systolic, "SBP")
    correlate_bp_ptt(ptt_forehead_ppg_neck, bp_diastolic, "DBP")
# Plot dei risultati
    plot_ptt_vs_bp(ptt_forehead_ppg_neck, bp_systolic, "SBP", output_dir)
    plot_ptt_vs_bp(ptt_forehead_ppg_neck, bp_diastolic, "DBP", output_dir)

    print("\n[INFO] All processing steps completed successfully.")


if __name__ == "__main__":
    main()

