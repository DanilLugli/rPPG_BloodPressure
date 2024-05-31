import json
import os
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import cv2

print("Library imported successfully!")
# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Indici dei landmark per la fronte (aggiornati per coprire tutta la fronte, l'ordine dei landmarks ha valore, bisogna seguire il perimetro)
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

def extract_rgb_trace(video_path, roi, output_folder, part):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    rgb_signals = {'R': [], 'G': [], 'B': []}

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        x, y, w, h = roi
        roi_frame = image[y:y+h, x:x+w]
        mean_val = cv2.mean(roi_frame)[:3]  # Extract mean values of the RGB channels
        rgb_signals['R'].append(mean_val[2])
        rgb_signals['G'].append(mean_val[1])
        rgb_signals['B'].append(mean_val[0])

        # Optional: Draw the ROI on the frame for visualization
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('ROI', image)
        if cv2.waitKey(5) & 0xFF == 27:  # 27 corresponds to the ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert lists to numpy arrays
    for color in rgb_signals:
        rgb_signals[color] = np.array(rgb_signals[color])

    # Save raw RGB signals
    for color in rgb_signals:
        np.savetxt(f"{output_folder}/rgb_raw_{color}_Neck.txt", rgb_signals[color])
        plt.plot(rgb_signals[color])
        plt.title(f'RGB Signal (Raw - {color})')
        plt.xlabel('Frame')
        plt.ylabel('Intensity')
        plt.savefig(f"{output_folder}/rgb_raw_{color}_Neck.jpg")
        plt.close()

    return rgb_signals

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
    if len(signal) <= 27:  # Padlen for the filter is 27
        print("Warning: The length of the input vector is too short for bandpass filtering. Skipping filtering.")
        return signal  # Return the signal unfiltered
    sos = butter(order, [lowcut, highcut], btype='band', output='sos', fs=fs)
    filtered_signal = sosfiltfilt(sos, signal)
    return filtered_signal

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

def estimate_systolic_blood_pressure(ptt_values, gender):
    if gender == 'male':
        a_sbp = -100  # Coefficiente per la stima della SBP negli uomini
        b_sbp = 120   # Intercetta per la stima della SBP negli uomini
    elif gender == 'female':
        a_sbp = -90   # Coefficiente per la stima della SBP nelle donne
        b_sbp = 110   # Intercetta per la stima della SBP nelle donne
    else:
        raise ValueError("Gender not recognized. Please specify 'male' or 'female'.")

    #ptt_values = np.array(ptt_values)  
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
    #ptt_values = np.array(ptt_values)  
    estimated_dbp = a_dbp * ptt_values + b_dbp
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

def interpolate_to_length(values, target_length):
    x = np.linspace(0, len(values) - 1, len(values))
    x_new = np.linspace(0, len(values) - 1, target_length)
    interpolated_values = np.interp(x_new, x, values)
    return interpolated_values
    
def get_sampling_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

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

def divide_and_calculate_means(data, num_groups=6):
    # Assicurati che i dati possano essere divisi equamente nel numero specificato di gruppi
    data_length = len(data)
    group_size = data_length // num_groups
    
    means = []
    for i in range(num_groups):
        group_data = data[i * group_size:(i + 1) * group_size]
        group_mean = np.mean(group_data)
        means.append(group_mean)
    
    return means

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

def main():
    
    # Path Subject and Task
    #dataset_folder = "/Volumes/DanoUSB/"
    #subject = "F001"
    #task = "T1"
    dataset_folder = "/Users/danillugli/Desktop/Boccignone/BP4D+"
    subject = "M001"
    task = "T3/"
    gender = "female" if subject[0] == "F" else "male" if subject[0] == "M" else "unknown"

    video_path = dataset_folder + f"/{subject}/{task}/vid.avi"
    # Output Path
    output_dir =  f"NIAC/{subject}/{task}"
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nFolder \"{output_dir}\" has been created.")

    print(f"\n<-- START - Subject: {subject} - Task: {task} -->\n")

    # Path ROI Forehead + Neck
    neck_roi_file_path = f"NIAC/{subject}/{task}neck_roi_dimensions.json"
    #forehead_roi_file_path = f"NIAC/{subject}/{task}forehead_roi_dimensions.json"

    sampling_rate = get_sampling_rate(video_path)
    print(f"Sampling rate: {sampling_rate} FPS\n")

    # Extract RGB Forehead signal
    print("\n1] Extracting RGB Forehead Signal")
    rgb_signals_forehead = extract_rgb_trace_MediaPipe(video_path, output_dir)
    T_rgb_signal_forehead = np.vstack((rgb_signals_forehead['R'], rgb_signals_forehead['G'], rgb_signals_forehead['B'])).T
    #print(len(rgb_signals_forehead['R']))
    #print(rgb_signals_forehead['R'])
    print("RGB Forehead Signal Extracted\n")

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

    #Extract RGB Neck signal
    print("\n2]Extracting RGB Neck Signal")
    rgb_signal_neck = extract_rgb_trace(video_path, neck_roi, output_dir, "neck")
    T_rgb_signals_neck = np.vstack((rgb_signal_neck['R'], rgb_signal_neck['G'], rgb_signal_neck['B'])).T 
    #print(rgb_signal_neck['R'])
    print("RGB Neck Signal Extracted.\n")


    print("3]Filtering Forehead Signal... \n")
    # Apply Smoothness Priors Detrend filter to forehead signal
    lambda_param = 10
    filtered_rgb_signal_forehead = smoothness_priors_detrend(T_rgb_signal_forehead, lambda_param)
    #filtered_rgb_signal_forehead = {}
    #for color in rgb_signals_forehead:
    #    filtered_rgb_signal_forehead[color] = smoothness_priors_detrend(T_rgb_signal_forehead[color], lambda_param)

    

    print("4]Filtering Neck Signal... \n")
    # Apply Butterworth bandpass filter to neck signal
    low_cutoff_frequency = 0.6
    high_cutoff_frequency = 4.0 
    filtered_rgb_signal_neck= {}
    for color in rgb_signal_neck:
        filtered_rgb_signal_neck[color] = butter_bandpass_filter(rgb_signal_neck[color], low_cutoff_frequency, high_cutoff_frequency, sampling_rate)
    
    filtered_rgb_signal_neck = np.vstack((filtered_rgb_signal_neck['R'], filtered_rgb_signal_neck['G'], filtered_rgb_signal_neck['B'])).T 


    # Save filtered rgb signals for forehead
    np.savetxt(f"{output_dir}/rgb_filtered_forehead_R.txt", filtered_rgb_signal_forehead)
    plt.plot(filtered_rgb_signal_forehead)
    plt.title('RGB Signal (Filtered - Forehead - R)')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.savefig(f"{output_dir}/rgb_filtered_forehead_R.jpg")
    plt.close()

    # Save filtered rgb signals for neck
    np.savetxt(f"{output_dir}/rgb_filtered_neck_R.txt", filtered_rgb_signal_neck)
    plt.plot(filtered_rgb_signal_neck)
    plt.title('RGB Signal (Filtered - Neck - R)')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.savefig(f"{output_dir}/rgb_filtered_neck_R.jpg")
    plt.close()

    print("5]cpu_CHROM Applicatin\n")
    #Get BVP_Signal_Forehead
    bvp_signal_forehead = cpu_CHROM(filtered_rgb_signal_forehead)
    #Get BBVP_Signal_Forehead
    bvp_signal_neck = cpu_CHROM(filtered_rgb_signal_neck)

    print("6] Saving Forehead BVP Signal")
    #Save BVP Signal Forehead
    np.savetxt(f"{output_dir}/bvp_signal_forehead.txt", bvp_signal_forehead)
    plt.plot(bvp_signal_forehead)
    plt.title('BVP Signal_Forehead')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.savefig(f"{output_dir}/bvp_signal_forehead.jpg")
    plt.close()
    
    print("7] Saving Neck BVP Signal")
    #Save BVP Signal Neck
    np.savetxt(f"{output_dir}/bvp_signal_neck.txt", bvp_signal_neck)
    plt.plot(bvp_signal_neck)
    plt.title('BVP Signal_Neck')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.savefig(f"{output_dir}/bvp_signal_neck.jpg")
    plt.close()

    # Find and plot peaks in the filtered red signals
    forehead_peaks = find_peaks_in_signal(bvp_signal_forehead)
    neck_peaks = find_peaks_in_signal(bvp_signal_neck)

    plot_peaks(bvp_signal_forehead, forehead_peaks, 'Forehead PPG (R)', output_dir)
    plot_peaks(bvp_signal_neck, neck_peaks, 'Neck PPG (R)', output_dir)

    # Calculate PTT
    ptt = calculate_ptt(forehead_peaks, neck_peaks, 50, output_dir)
    print("PTT values (in seconds):", ptt)

    # Estimate SBP and DBP every 30 seconds
    estimated_sbp = estimate_systolic_blood_pressure(ptt, gender)
    print("Estimated SBP values (mmHg): ", estimated_sbp)

    estimated_dbp = estimate_diastolic_blood_pressure(ptt, gender)
    print("Estimated DBP values (mmHg): ", estimated_dbp)

    plot_estimated_blood_pressure_scatter(estimated_sbp, estimated_dbp, output_dir)

    # Interpolate DBP values to match the length of the reference file
    target_length = 1000  # Replace with the desired length
    interpolated_dbp = interpolate_to_length(estimated_dbp, target_length)
    interpolated_sbp = interpolate_to_length(estimated_sbp, target_length)

    #plot_estimated_blood_pressure_scatter(interpolated_sbp, interpolated_dbp, output_dir)

    #print(interpolated_dbp)
    #print(interpolated_sbp)
    data_dbp = np.loadtxt('/Users/danillugli/Desktop/Boccignone/Project/NIAC/M001/T3/estimated_dbp.txt')
    data_sbp = np.loadtxt('/Users/danillugli/Desktop/Boccignone/Project/NIAC/M001/T3/estimated_sbp.txt') 

    dbp_media = divide_and_calculate_means(data_dbp)
    sbp_media = divide_and_calculate_means(data_sbp)

    plt.plot(dbp_media)
    plt.title('DBP Media')
    plt.xlabel('Frame')
    plt.ylabel('DBP')
    plt.savefig(f"{output_dir}/dbp_media.jpg")
    plt.close()

    plt.plot(sbp_media)
    plt.title('SBP Media')
    plt.xlabel('Frame')
    plt.ylabel('SBP')
    plt.savefig(f"{output_dir}/sbp_media.jpg")
    plt.close()

    # Plot and save PTT values and estimated BP values
    plot_ptt_values(ptt, output_dir)
    plot_estimated_bp(estimated_sbp, 'SBP', output_dir)
    plot_estimated_bp(interpolated_dbp, 'DBP', output_dir)
    save_estimated_bp(estimated_sbp, 'SBP', output_dir)
    save_estimated_bp(estimated_dbp, 'DBP', output_dir)

    print("\n\nAll processing steps completed successfully.")

if __name__ == "__main__":
    main()
