import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setul de date
data = pd.read_csv("C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv")

# Exclude non-numeric columns and irrelevant variables before calculating the correlation matrix
numeric_data = data.select_dtypes(include=['number']).drop(columns=['host_id', 'latitude', 'longitude'])
correlation_matrix = numeric_data.corr()

# Crearea unui heat map cu matricea de corelație
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Heat Map - Matrice de Corelație pentru Închirierea Airbnb în New York')
plt.show()
