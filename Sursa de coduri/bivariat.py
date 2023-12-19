import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specificați calea către fișierul CSV
file_path = "C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv"
data = pd.read_csv(file_path)

# Analize bivariate pentru variabilele de interes

# Relația dintre 'neighbourhood_group' și 'room_type'
plt.figure(figsize=(12, 8))
sns.countplot(x='neighbourhood_group', hue='room_type', data=data)
plt.title('Relația dintre neighbourhood_group și room_type')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Număr de Închirieri')
plt.show()

# Relația dintre 'room_type' și 'minimum_nights'
plt.figure(figsize=(12, 8))
sns.boxplot(x='room_type', y='minimum_nights', data=data)
plt.title('Relația dintre room_type și minimum_nights')
plt.xlabel('Room Type')
plt.ylabel('Număr Minim de Nopți')
plt.show()

# Relația dintre 'neighbourhood' și 'price'
plt.figure(figsize=(15, 8))
sns.boxplot(x='neighbourhood', y='price', data=data)
plt.title('Relația dintre neighbourhood și price')
plt.xlabel('Neighbourhood')
plt.ylabel('Preț')
plt.xticks(rotation=45, ha='right')
plt.show()

# Relația dintre 'neighbourhood_group' și 'reviews_per_month'
plt.figure(figsize=(12, 8))
sns.histplot(data=data, x='reviews_per_month', hue='neighbourhood_group', kde=True, bins=30)
plt.title('Distribuția reviews_per_month în funcție de neighbourhood_group')
plt.xlabel('Reviews per Month')
plt.ylabel('Frecvență')
plt.show()

# Analiza bivariată prin grafic de perechi
sns.pairplot(data)
plt.show()
