import pandas as pd
import numpy as np

# Încărcați setul de date
data = pd.read_csv("C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv")

# Identificați și afișați numărul de valori lipsă pentru fiecare coloană numerică
missing_values = data.select_dtypes(include=['number']).isnull().sum()
print("Numărul de valori lipsă în fiecare coloană numerică:")
print(missing_values)

# Exemplu pentru imputarea valorilor lipsă în coloana 'price' (puteți aplica aceeași logică pentru alte coloane)
average_price = np.mean(data['price'])
data['price'].fillna(average_price, inplace=True)

# Verificați dacă valorile lipsă au fost imputate cu succes
missing_values_after_imputation = data.select_dtypes(include=['number']).isnull().sum()
print("\nNumărul de valori lipsă după imputare:")
print(missing_values_after_imputation)
