import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Citirea setului de date
data = pd.read_csv("C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv")

# Eliminarea coloanelor nedorite sau non-numerice
data = data.select_dtypes(include=[float, int])

# Separarea setului de date în variabila de caracteristici (X) și variabila de rezultat (y)
X = data.drop('price', axis=1)
y = data['price']

# Tratarea valorilor lipsă folosind SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standardizarea caracteristicilor
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_imputed)

# Împărțirea setului de date în setul de antrenare și setul de testare
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# Inițializarea modelului Elastic Net
model = ElasticNet(alpha=1, l1_ratio=0.5)

# Antrenarea modelului
model.fit(X_train, y_train)

# Obținerea coeficienților importanți
coefficients = model.coef_

# Afișarea coeficienților într-o diagramă
plt.figure(figsize=(10, 6))
plt.bar(X.columns, coefficients)
plt.title('Importanța caracteristicilor folosind Elastic Net')
plt.xlabel('Caracteristici')
plt.ylabel('Coeficienti')
plt.xticks(rotation=45)
plt.show()
