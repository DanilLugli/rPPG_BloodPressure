import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import os


def load_aix_values(filepath):
    """Carica i valori AIx da un file .txt contenente una sola colonna."""
    Aix = np.loadtxt(filepath).reshape(-1, 1)
    return Aix


def read_blood_pressure_data(file_path):
    """Legge i dati della pressione da file, assumendo che ci sia una riga di intestazione."""
    return np.loadtxt(file_path, skiprows=1)


def calculate_coefficients(X, Y):
    """Calcola i coefficienti del modello lineare."""
    model = LinearRegression()
    model.fit(X, Y)
    a = model.coef_[0]
    b = model.intercept_
    return a, b, model


def estimate_bp(model, AIx_values):
    """Esegue la stima della pressione sanguigna usando il modello lineare."""
    return model.predict(AIx_values)


def average_bp_in_windows(bp_values, num_windows):
    """Divide i valori di BP in finestre e calcola la media per ciascuna finestra."""
    window_size = len(bp_values) // num_windows
    return np.array([np.mean(bp_values[i*window_size:(i+1)*window_size]) for i in range(num_windows)])


def main():
    aix_file = "/Users/danillugli/Desktop/Boccignone/Project/NIAC/M047/T3/aix_values_red.txt"
    bp_systolic_file = "/Volumes/DanoUSB/Physiology/M047/T3/LA Systolic BP_mmHg.txt"
    bp_diastolic_file = "/Volumes/DanoUSB/Physiology/M047/T3/BP Dia_mmHg.txt"

    # Carica dati
    AIx = load_aix_values(aix_file)
    SBP = read_blood_pressure_data(bp_systolic_file)
    DBP = read_blood_pressure_data(bp_diastolic_file)

    num_windows = len(AIx)
    window_size_sbp = len(SBP) // num_windows
    window_size_dbp = len(DBP) // num_windows

    SBP_mean = np.array([np.mean(SBP[i*window_size_sbp:(i+1)*window_size_sbp]) for i in range(num_windows)])
    DBP_mean = np.array([np.mean(DBP[i*window_size_sbp:(i+1)*window_size_sbp]) for i in range(num_windows)])

    # Calcola coefficienti
    a_sbp, b_sbp, model_sbp = calculate_coefficients(AIx, SBP_mean)
    a_dbp, b_dbp, model_dbp = calculate_coefficients(AIx, DBP_mean)

    print(f'Coefficienti SBP: a = {a_sbp:.4f}, b = {b_sbp:.4f}')
    print(f'Coefficienti DBP: a = {a_dbp:.4f}, b = {b_dbp:.4f}')

    # Stima valori BP da AIx nuovi
    AIx_new = np.linspace(np.min(AIx), np.max(AIx), 8).reshape(-1, 1)
    SBP_estimated = estimate_bp(model_sbp, AIx_new)
    DBP_estimated = estimate_bp(model_dbp, AIx_new)

    print("\nStime della pressione sanguigna (su nuovi valori AIx):")
    for i, aix in enumerate(AIx_new):
        print(f'AIx: {aix[0]:.2f} -> SBP stimato: {SBP_estimated[i]:.2f}, DBP stimato: {DBP_estimated[i]:.2f}')

    # Predizioni sui dati originali e calcolo degli errori
    SBP_pred = estimate_bp(model_sbp, AIx)
    DBP_pred = estimate_bp(model_dbp, AIx)

    sbp_rmse = np.sqrt(np.mean((SBP_mean - SBP_pred)**2))
    dbp_rmse = np.sqrt(np.mean((DBP_mean - DBP_pred)**2))
    sbp_r2 = model_sbp.score(AIx, SBP_mean)
    dbp_r2 = model_dbp.score(AIx, DBP_mean)

    print("\nMetriche di valutazione:")
    print(f"SBP - RMSE: {sbp_rmse:.2f}, R²: {sbp_r2:.4f}")
    print(f"DBP - RMSE: {dbp_rmse:.4f}, R²: {dbp_r2:.4f}")

    # Calcola la correlazione di Pearson sui dati originali
    r_sbp, p_sbp = pearsonr(SBP_mean, SBP_pred)
    r_dbp, p_dbp = pearsonr(DBP_mean, DBP_pred)

    print("\nCorrelazioni di Pearson:")
    print(f"SBP - Pearson correlation: {r_sbp:.4f}, p-value: {p_sbp:.4f}")
    print(f"DBP - Pearson correlation: {r_dbp:.4f}, p-value: {p_dbp:.4f}")


if __name__ == '__main__':
    main()