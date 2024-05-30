import pandas as pd
import matplotlib.pyplot as plt

# Carica i dati dal file
file_path = '/Users/danillugli/Desktop/Boccignone/F001/T1/BP_mmHg.txt'
data = pd.read_csv(file_path, header=None, names=['BP_mmHg'])

# Calcola statistiche descrittive
mean_bp = data['BP_mmHg'].mean()
median_bp = data['BP_mmHg'].median()
std_bp = data['BP_mmHg'].std()
min_bp = data['BP_mmHg'].min()
max_bp = data['BP_mmHg'].max()

print(f"Media della pressione sanguigna: {mean_bp:.2f} mmHg")
print(f"Mediana della pressione sanguigna: {median_bp:.2f} mmHg")
print(f"Deviazione standard della pressione sanguigna: {std_bp:.2f} mmHg")
print(f"Pressione sanguigna minima: {min_bp:.2f} mmHg")
print(f"Pressione sanguigna massima: {max_bp:.2f} mmHg")

# Aggiungi una linea di tendenza (media mobile)
data['Trend'] = data['BP_mmHg'].rolling(window=50).mean()

# Crea il grafico principale con la linea di tendenza
plt.figure(figsize=(12, 6))
plt.plot(data['BP_mmHg'], label='Pressione Sanguigna (mmHg)')
plt.plot(data['Trend'], label='Trend (Media Mobile)', color='red', linewidth=2)
plt.xlabel('Misurazione')
plt.ylabel('BP (mmHg)')
plt.title('Andamento della pressione sanguigna nel tempo')
plt.legend()
plt.show()

# Segmenta i dati nei primi e ultimi 1000 punti
segment1 = data['BP_mmHg'].iloc[:1000]
segment2 = data['BP_mmHg'].iloc[-1000:]

# Crea un grafico per i segmenti iniziali e finali
plt.figure(figsize=(12, 6))
plt.plot(segment1, label='Segmento Iniziale')
plt.plot(segment2.reset_index(drop=True), label='Segmento Finale', color='orange')
plt.xlabel('Misurazione')
plt.ylabel('BP (mmHg)')
plt.title('Segmenti Iniziali e Finali della pressione sanguigna')
plt.legend()
plt.show()
