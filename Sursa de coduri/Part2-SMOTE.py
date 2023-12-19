import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Citirea datelor din fișierul CSV
file_path = "C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv"
data = pd.read_csv(file_path)

# Afișarea primelor câteva rânduri ale setului de date
print(data.head())

# Eliminarea coloanelor nedorite sau non-numerice
data_numeric = data.select_dtypes(include=[float, int])

# Separarea setului de date în variabila de caracteristici (X) și variabila țintă (y)
X = data_numeric.drop('price', axis=1)
y = data_numeric['price']

# Imputarea valorilor lipsă
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Crearea unui nou DataFrame după imputare
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Convertirea variabilei țintă la 0 și 1 (1 pentru valori non-NaN, 0 pentru NaN)
y = ~y.isna()

# Impărțirea datelor în set de antrenare și set de testare
X_train, X_test, y_train, y_test = train_test_split(X_imputed_df, y, test_size=0.2, random_state=42)

# Verificăm dacă avem cel puțin două clase diferite în variabila țintă
if y_train.nunique() > 1:
    # Aplicarea SMOTE pe setul de antrenare doar dacă avem cel puțin două clase
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Restul codului pentru afișarea rezultatelor
    class_counts_before = pd.Series(y_train.value_counts(), name='Before SMOTE')
    class_counts_after = pd.Series(y_resampled.value_counts(), name='After SMOTE')

    # Afișarea graficului de bare pentru comparație
    plt.figure(figsize=(10, 6))
    class_counts_before.plot(kind='bar', color='lightblue', position=0, width=0.4, label='Before SMOTE')
    class_counts_after.plot(kind='bar', color='orange', position=1, width=0.4, label='After SMOTE')
    plt.title('Distribution of Classes Before and After SMOTE')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.show()
else:
    print("Nu se poate aplica SMOTE deoarece există o singură clasă în variabila țintă.")
