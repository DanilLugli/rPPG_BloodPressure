import os
import numpy as np
import scipy.signal
from scipy.stats import pearsonr


def estimate_bp_using_phase_diff_segments(ppw_signal, bvp_signal, sampling_rate, num_segments=6):
    """
    Suddivide i segnali in num_segments e per ogni segmento calcola SBP e DBP.

    Args:
        ppw_signal (numpy array): Segnale PPW dal collo.
        bvp_signal (numpy array): Segnale BVP dal fronte o dal collo.
        sampling_rate (float): Frequenza di campionamento.
        num_segments (int): Numero di segmenti da generare.

    Returns:
        dict: Due liste di 6 valori rispettivamente per SBP e DBP.
    """
    # Fai in modo che i segnali abbiano la stessa lunghezza
    min_len = min(len(ppw_signal), len(bvp_signal))
    ppw_signal = ppw_signal[:min_len]
    bvp_signal = bvp_signal[:min_len]
    
    # Suddividi i segnali in num_segments parti
    ppw_segments = np.array_split(ppw_signal, num_segments)
    bvp_segments = np.array_split(bvp_signal, num_segments)
    
    sbp_list = []
    dbp_list = []
    
    for seg_ppw, seg_bvp in zip(ppw_segments, bvp_segments):
        # Calcola la FFT per il segmento
        fft_ppw = np.fft.fft(seg_ppw)
        fft_bvp = np.fft.fft(seg_bvp)
        
        # Calcola la fase
        phase_ppw = np.angle(fft_ppw)
        phase_bvp = np.angle(fft_bvp)
        
        # Differenza di fase
        phase_diff = np.abs(phase_ppw - phase_bvp)
        
        # Frequenze associate al segmento
        freqs = np.fft.fftfreq(len(seg_ppw), d=1/sampling_rate)
        valid_indices = np.where((freqs > 0.5) & (freqs < 3.0))[0]
        
        # Se non ci sono indici validi, evita divisioni per zero
        if valid_indices.size > 0:
            mean_phase_diff = np.mean(phase_diff[valid_indices])
        else:
            mean_phase_diff = 0.0
        
        # Applica il modello empirico
        sbp = 120 - (3 * mean_phase_diff)
        dbp = 80 - (1.5 * mean_phase_diff)
        
        sbp_list.append(round(sbp, 2))
        dbp_list.append(round(dbp, 2))
    
    return {"Systolic BP (SBP)": sbp_list, "Diastolic BP (DBP)": dbp_list}


def read_blood_pressure_data(file_path):
    """
    Legge i dati della pressione da file.
    Assumiamo che il file contenga un'intestazione da saltare.
    """
    return np.loadtxt(file_path, skiprows=1)


if __name__ == "__main__":
    # Percorsi dei file per i segnali PPG/BVP
    base_path = "/Users/danillugli/Desktop/Boccignone/Project/NIAC/F001/T3"
    ppw_file = os.path.join(base_path, "bvp_green_neck_neck.txt")
    bvp_file = os.path.join(base_path, "bvp_chrom_signal_forehead_conv.txt")
    
    # Carica i segnali, saltando l'intestazione se presente
    ppw_signal = np.loadtxt(ppw_file, skiprows=1)
    bvp_signal = np.loadtxt(bvp_file, skiprows=1)
    
    sampling_rate = 100  # Ad esempio, 100 Hz

    bp_estimates = estimate_bp_using_phase_diff_segments(ppw_signal, bvp_signal, sampling_rate, num_segments=6)
    print("Valori stimati (per segmento):")
    print(bp_estimates)
    
    # Calcola la media dei valori stimati su tutti i segmenti
    mean_sbp_estimated = np.mean(bp_estimates["Systolic BP (SBP)"])
    mean_dbp_estimated = np.mean(bp_estimates["Diastolic BP (DBP)"])
    
    print(f"[INFO] Mean Estimated Systolic BP: {mean_sbp_estimated:.2f}")
    print(f"[INFO] Mean Estimated Diastolic BP: {mean_dbp_estimated:.2f}")
    
    # ------------------ Nuova sezione per i dati di pressione di riferimento ------------------
    bp_systolic_file = "/Volumes/DanoUSB/Physiology/F001/T3/LA Systolic BP_mmHg.txt"
    bp_diastolic_file = "/Volumes/DanoUSB/Physiology/F001/T3/BP Dia_mmHg.txt"

    bp_systolic = read_blood_pressure_data(bp_systolic_file)
    bp_diastolic = read_blood_pressure_data(bp_diastolic_file)

    def calculate_windowed_mean(data, num_windows):
        window_size = len(data) // num_windows
        windowed_means = [np.mean(data[i * window_size:(i + 1) * window_size]) for i in range(num_windows)]
        return np.array(windowed_means)

    # Calcola i valori medi suddivisi in 6 finestre
    num_windows = 6
    bp_systolic_mean = calculate_windowed_mean(bp_systolic, num_windows)
    bp_diastolic_mean = calculate_windowed_mean(bp_diastolic, num_windows)

    print(f"[INFO] Systolic BP (mean of 6 windows): {bp_systolic_mean}")
    print(f"[INFO] Diastolic BP (mean of 6 windows): {bp_diastolic_mean}")

    # ------------------ Calcola la correlazione di Pearson ------------------
    r_sbp, p_sbp = pearsonr(bp_estimates["Systolic BP (SBP)"], bp_systolic_mean)
    r_dbp, p_dbp = pearsonr(bp_estimates["Diastolic BP (DBP)"], bp_diastolic_mean)

    print("Systolic BP - Pearson correlation:", r_sbp, "p-value:", p_sbp)
    print("Diastolic BP - Pearson correlation:", r_dbp, "p-value:", p_dbp)