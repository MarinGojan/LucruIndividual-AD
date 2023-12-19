import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Specificați calea către fișierul CSV
file_path ="C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv"

# Citirea datelor din fișierul CSV
data = pd.read_csv(file_path)

# Verificare duplicații în întreg DataFrame
duplicate_rows = data[data.duplicated()]

# Afișare rânduri duplicate (dacă există)
print("Rânduri duplicate:")
print(duplicate_rows)

# Eliminare rânduri duplicate (păstrează doar prima apariție)
data = data.drop_duplicates()

# Verificare dacă există valori lipsă în fiecare coloană
missing_values = data.isnull().sum()

# Afișare numărul de valori lipsă pentru fiecare coloană
print("Numărul de valori lipsă în fiecare coloană:")
print(missing_values)

# Analiza univariată pentru variabilele numerice
numeric_vars = data.select_dtypes(include=['number'])

# Histogramă pentru fiecare variabilă numerică
for column in numeric_vars.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], bins=30, kde=True, color='blue')
    plt.title(f'Histograma pentru {column}')
    plt.xlabel('Valori')
    plt.ylabel('Frecvență')
    plt.show()

    # Diagrama cutiei (Box Plot) pentru fiecare variabilă numerică
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[column], color='green')
    plt.title(f'Diagrama cutiei pentru {column}')
    plt.xlabel(column)
    plt.show()

# Analiza bivariată pentru variabilele de interes
# Exemplu: Relația dintre 'neighbourhood_group' și 'price'
plt.figure(figsize=(12, 8))
sns.boxplot(x='neighbourhood_group', y='price', data=data)
plt.title('Relația dintre neighbourhood_group și preț')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Preț')
plt.show()
