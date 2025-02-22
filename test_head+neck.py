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

def plot_ptt_values(ptt_values, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(ptt_values, label='PTT Values')
    plt.xlabel('Measurement Index')
    plt.ylabel('PTT (seconds)')
    plt.title('PTT Values Over Time')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ptt_values.png'))
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
    subject = "M001"
    task = "T3"
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

def save_and_visualize_results(bvp_forehead, bvp_neck, forehead_peaks, neck_peaks, ptt, estimated_sbp, estimated_dbp, output_dir):
    """
    Salva e visualizza i risultati delle elaborazioni.
    """
    print("\n[INFO] Saving and visualizing results...")
    plot_peaks(bvp_forehead, forehead_peaks, 'Forehead PPG (CHROM)', output_dir)
    plot_peaks(bvp_neck, neck_peaks, 'Neck PPG (CHROM)', output_dir)
    plot_ptt_values(ptt, output_dir)
    plot_estimated_bp(estimated_sbp, 'SBP', output_dir)
    plot_estimated_bp(estimated_dbp, 'DBP', output_dir)
    save_estimated_bp(estimated_sbp, 'SBP', output_dir)
    save_estimated_bp(estimated_dbp, 'DBP', output_dir)
    print("[INFO] Results saved and visualized successfully.")

def main():
    # Step 0: Initialize paths and configurations
    video_path, output_dir, neck_roi_file_path, gender = initialize_paths_and_config()
    sampling_rate = get_sampling_rate(video_path)
    print(f"[INFO] Sampling rate: {sampling_rate} FPS\n")

    # Step 1: Calculate ROI
    neck_roi = calculate_roi(video_path, neck_roi_file_path)
    if neck_roi is None:
        return

    # Step 2: Process forehead PPG
    T_rgb_signals_forehead = process_ppg_forehead(video_path, output_dir)

    # Step 3: Process neck PPG
    T_mean_signals_neck = process_ppg_neck(video_path, neck_roi, output_dir)


    # Step 4: Calculate neck PPW
    calculate_ppw_neck(video_path, neck_roi, output_dir)

     # Step 5: Preprocess and filter signals
    filtered_forehead = preprocess_and_filter_signals(T_rgb_signals_forehead, sampling_rate)
    filtered_neck = preprocess_and_filter_signals(T_mean_signals_neck, sampling_rate)

    # Step 6: Extract BVP
    print("\n[STEP 6] Extracting BVP...")
    bvp_forehead = cpu_CHROM(filtered_forehead)  # CHROM for forehead
    bvp_neck = cpu_POS(filtered_neck)  # POS for neck
    print("[INFO] BVP signals extracted successfully.")

    # Step 7: Detect peaks and calculate AIx
    print("\n[STEP 7] Detecting peaks and calculating AIx...")
    forehead_peaks = find_peaks_in_signal(bvp_forehead, fps=sampling_rate)
    neck_peaks = find_peaks_in_signal(bvp_neck, fps=sampling_rate)
    calculate_aix(bvp_forehead, bvp_neck, forehead_peaks, neck_peaks, sampling_rate, output_dir)


    # Step 8: Calculate PTT
    print("\n[STEP 8] Calculating PTT...")
    raw_ptt = calculate_ptt(forehead_peaks, neck_peaks, sampling_rate, output_dir)
    ptt = validate_ptt(raw_ptt)
    print(f"[INFO] Filtered PTT values: {ptt}")

    # Step 9: Estimate Blood Pressure
    print("\n[STEP 9] Estimating Blood Pressure...")
    estimated_sbp = estimate_systolic_blood_pressure(ptt, gender)
    estimated_dbp = estimate_diastolic_blood_pressure(ptt, gender)
    print(f"[INFO] Estimated SBP: {estimated_sbp}")
    print(f"[INFO] Estimated DBP: {estimated_dbp}")

    # Step 10: Save and visualize results
    save_and_visualize_results(bvp_forehead, bvp_neck, forehead_peaks, neck_peaks, ptt, estimated_sbp, estimated_dbp, output_dir)

    # Final message
    print("\n[INFO] All processing steps completed successfully.")

if __name__ == "__main__":
    main()