import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Citirea datelor din fișierul CSV
file_path = "C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv"
data = pd.read_csv(file_path)

# Afișarea primelor câteva rânduri ale setului de date
print(data.head())

# Selectarea coloanelor categorice pentru analiză
categorical_columns = ['neighbourhood_group', 'room_type', 'neighbourhood']

# Configurarea aspectului vizual al graficului
sns.set(style="whitegrid")

# Afișarea graficului pentru fiecare coloană categorică
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=data, hue=column, palette="viridis", legend=False)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
