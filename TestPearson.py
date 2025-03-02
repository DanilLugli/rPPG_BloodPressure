import pandas as pd
import os

def analyze_general_data(folder_path):
    # Lista dei file CSV nella cartella specificata
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    all_results = []  # Lista per raccogliere i risultati
    
    for file in files:
        # Carica ogni file CSV
        file_path = os.path.join(folder_path, file)
        
        # Leggi il CSV, separando correttamente i dati (utilizzando la virgola come delimitatore)
        df = pd.read_csv(file_path, header=None)
        
        # Imposta manualmente le colonne
        df.columns = ['Pair', 'Pearson r', 'p-value']
        
        # Rimuovi eventuali righe vuote
        df.dropna(inplace=True)
        
        # Aggiungi una colonna per il nome del file
        df['File'] = file
        
        # Forza la conversione dei valori in 'Pearson r' e 'p-value' in numerico (forzando errori a NaN)
        df['Pearson r'] = pd.to_numeric(df['Pearson r'], errors='coerce')
        df['p-value'] = pd.to_numeric(df['p-value'], errors='coerce')
        
        # Aggiungi il dataframe alla lista dei risultati
        all_results.append(df)
    
    # Verifica se ci sono file da concatenare
    if not all_results:
        print("Nessun file contiene le colonne 'Pearson r' e 'p-value'.")
        return None

    # Unisci tutti i DataFrame in uno solo
    combined_df = pd.concat(all_results, ignore_index=True)

    # Calcola la media dei valori 'Pearson r' e 'p-value' per ogni combinazione
    summary = combined_df.groupby('Pair').agg({
        'Pearson r': 'mean',  # Media dei valori Pearson r per ciascuna combinazione
        'p-value': 'mean'      # Media dei valori p-value per ciascuna combinazione
    }).reset_index()

    # Salva i risultati di sintesi in un file CSV
    output_path = os.path.join(folder_path, 'general_analysis_summary.csv')
    summary.to_csv(output_path, index=False)
    
    print(f"Analisi generale completata. I risultati sono stati salvati in {output_path}")
    return summary

# Usa la funzione con il percorso della cartella contenente i file CSV
folder_path = '/Users/danillugli/Desktop/Boccignone/Project/Pearson_Value'
general_stats_df = analyze_general_data(folder_path)

# Se i dati sono stati analizzati correttamente, stampiamo i risultati
if general_stats_df is not None:
    print(general_stats_df)