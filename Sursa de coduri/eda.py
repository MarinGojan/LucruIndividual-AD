import pandas as pd

# Specificați calea către fișierul CSV
file_path = "C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv"

# Citirea datelor din fișierul CSV
data = pd.read_csv(file_path)

# Setarea opțiunilor pentru a afișa toate coloanele și toate rândurile
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Afișarea primelor câteva și ultimelor câteva rânduri ale datelor pentru verificare
print("Primele câteva rânduri:")
print(data.head())

print("\nUltimele câteva rânduri:")
print(data.tail())

# Resetarea opțiunilor la valorile implicite (opțional)
pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')
# Informații despre setul de date (inclusiv tipurile de date și valorile lipsă)
print("\nInformații despre setul de date:")
print(data.info())

# Verificare duplicații în întreg DataFrame
duplicate_rows = data[data.duplicated()]

# Afișare rânduri duplicate (dacă există)
print("\nRânduri duplicate:")
print(duplicate_rows)

# Număr de valori unice în fiecare coloană
unique_counts = data.nunique()

# Afișare număr de valori unice pentru fiecare coloană
print("\nNumăr de valori unice în fiecare coloană:")
print(unique_counts)

# Eliminare rânduri duplicate (păstrează doar prima apariție)
data = data.drop_duplicates()

# Verificare dacă există valori lipsă în fiecare coloană
missing_values = data.isnull().sum()

# Afișare numărul de valori lipsă pentru fiecare coloană
print("\nNumărul de valori lipsă în fiecare coloană:")
print(missing_values)

# Rezumatul statisticilor pentru variabilele numerice
print("\nRezumatul statisticilor pentru variabilele numerice:")
print(data.describe().T)

# Identificarea variabilelor numerice și categorice
numeric_vars = data.select_dtypes(include=['number'])
categorical_vars = data.select_dtypes(include=['object'])

# Afișează numele variabilelor numerice
print("\nVariabile numerice:")
print(numeric_vars.columns)

# Afișează numele variabilelor categorice
print("\nVariabile categorice:")
print(categorical_vars.columns)
